"""
All tunable values in one place, loaded from a TOML file.

Defaults started with the original bot's values; HUD health and safety settings
are calibrated separately. Original locations are noted where they aren't obvious.
"""
from __future__ import annotations

import dataclasses
import os
import tomllib
import typing
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    # Roboflow model ids. Keys are read from environment variables, never stored here.
    scene: str = "albion_detector-l4o1a/5"
    obstacles: str = "obstacle_instance_detection/5"
    spells: str = "spell_detector/4"
    loading_bar: str = "loading_bar/1"
    health: str = "low_detector/1"
    # which environment variable holds the key for each model
    scene_key_env: str = "ROBOFLOW_KEY2"
    obstacles_key_env: str = "ROBOFLOW_KEY2"
    spells_key_env: str = "ROBOFLOW_KEY1"
    loading_bar_key_env: str = "ROBOFLOW_KEY2"
    health_key_env: str = "ROBOFLOW_KEY2"


@dataclass
class DetectionConfig:
    # minimum confidence per class. The original used .65 for everything except
    # silver and cup of piss, which also needed .55 at pickup time
    min_confidence: float = 0.65
    min_obstacle_confidence: float = 0.5
    class_confidence: dict[str, float] = field(default_factory=dict)
    # which detector classes mean what
    mob_classes: list[str] = field(default_factory=lambda: ["mob"])
    threat_classes: list[str] = field(default_factory=lambda: ["red"])
    farm_classes: list[str] = field(default_factory=lambda: ["hide", "fiber"])
    pickup_classes: list[str] = field(default_factory=lambda: ["silver", "cup of piss"])
    wisp_classes: list[str] = field(default_factory=lambda: ["wisp caged"])
    player_class: str = "player"
    # where the player usually is, relative to the window centre, used when no
    # player is detected (objectFinder.mainPlayerLocation: center + (2, -95))
    player_offset: tuple[int, int] = (2, -95)
    # a mob counts as close within this fraction of the window height (helper.is_mob_close)
    close_range: float = 0.3


@dataclass
class KeyConfig:
    # action bar slots, left to right, as the spell model sees them
    slots: list[str] = field(default_factory=lambda: ["q", "w", "e", "r", "d", "f"])
    food: str = "2"
    stop: str = "s"  # pressed before each click, as in helper.clickOn
    close_menus: str = "esc"
    attack_after_click: str = "space"
    # mouse button that walks the character (wander used right click)
    move_button: str = "right"
    zone_map: str = "tab"  # opens the zone map; hold-and-steer travel reads the player arrow on it
    # bot control hotkeys, global
    pause: str = "f8"
    quit: str = "f9"
    screenshot: str = "right shift"


@dataclass
class CastStep:
    slot: str            # slot key, e.g. "e"
    delay: float = 0.5   # seconds to wait after pressing
    repeat: int = 1


@dataclass
class CombatConfig:
    # cast before engaging if ready (attackMob cast R first)
    openers: list[CastStep] = field(default_factory=lambda: [CastStep("r", 0.5)])
    # first ready step is cast. Default is the warbow rotation
    rotation: list[CastStep] = field(default_factory=lambda: [CastStep("e", 0.5), CastStep("w", 0.5)])
    # pressed when nothing in the rotation is ready
    fallback: CastStep = field(default_factory=lambda: CastStep("q", 0.5))
    shield_slot: str = "d"
    shield_wait: float = 2.5
    leash_slot: str = "f"
    leash_recast_after: float = 3.0  # retained for old configs; Ethereal Path has no recast
    escape_distance: float = 300.0
    heal_max_wait: float = 60.0
    heal_settle: float = 5.0


@dataclass
class BehaviorConfig:
    # checked in this order every tick; the first that applies runs.
    # this is the order of the if-chain in the original main.py
    priority: list[str] = field(default_factory=lambda: [
        "shield", "heal", "fight_close", "travel", "pickup", "farm", "fight_far", "eat", "wander",
    ])
    disabled: list[str] = field(default_factory=list)
    travel_goal: str = ""  # empty disables travel; otherwise portal:NE or map x,y
    travel_click_interval: float = 1.0    # seconds between forward travel clicks
    travel_give_up_seconds: float = 30.0  # no move for this long clears the goal; 0 never gives up
    eat_every_minutes: float = 30.0
    eat_wait: float = 2.0
    pickup_wait: float = 2.0
    wisp_wait: float = 5.5
    farm_walk_timeout: float = 3.0
    farm_max_seconds: float = 30.0
    # ticks with nothing to do before wandering (main.py: bored_counter > 10)
    bored_ticks: int = 10
    idle_sleep: float = 0.15
    wander_range: float = 0.25
    wander_min_distance: float = 0.15
    wander_clicks: int = 4
    wander_click_interval: float = 1.5
    # nothing detected for this long: press esc, then pause if still nothing
    lost_seconds: float = 3.0
    # stuck detection: acting on the same screen spot this many times in a row
    # blocks that spot for blacklist_seconds
    stuck_repeats: int = 3
    stuck_radius: float = 40.0
    blacklist_seconds: float = 15.0
    resource_blacklist_seconds: float = 180.0
    resource_blacklist_durations: list[float] | None = None  # omitted: [legacy seconds, 600, 1800]
    resource_blacklist_radius: float = 60.0
    # detours around obstacles when a target's straight line is blocked
    detours: bool = True
    detour_margin: float = 25.0
    walk_speed_px_per_s: float = 250.0
    max_detour_walk: float = 2.5


@dataclass
class PerceptionConfig:
    capture_backend: str = "mss"   # "mss" (fast) or "pyautogui" (what the original used)
    scene_workers: int = 2         # globals.GENERAL_SCAN_PROCESSES used 2 threads
    status_hz: float = 5.0         # spells + health; the original ran roughly 1 to 2 per second
    stale_after: float = 2.0       # a scene older than this pauses decisions


@dataclass
class HealthConfig:
    # Window-relative pixels; recalibrate both size and rectangles for another HUD scale.
    window_size: tuple[int, int] = (2240, 1400)
    bar: tuple[int, int, int, int] = (105, 56, 245, 68)
    energy_bar: tuple[int, int, int, int] = (105, 72, 245, 80)
    low_threshold: float = 0.4
    model_fallback: bool = False
    # The green gathering/skinning progress bar under the player; window pixels at 2240x1400.
    gather_bar: tuple[int, int, int, int] = (1015, 1024, 1280, 1037)
    gather_bar_min_pixels: int = 30
    gather_bar_min_run: int = 25       # an unbroken green run this long in a row...
    gather_bar_left_slack: int = 20    # ...starting this close to the bar's left end
    gather_bar_model: bool = False   # the old loading_bar model; it fires on health bars


@dataclass
class SafetyConfig:
    hud_size: tuple[int, int] = (2240, 1400)
    hud_rects: list[tuple[int, int, int, int]] = field(default_factory=lambda: [
        (1820, 1030, 2240, 1400), (1880, 1340, 2240, 1400), (0, 0, 520, 200),
        (1660, 0, 2240, 90), (2140, 80, 2240, 260), (680, 1300, 1580, 1400),
    ])
    minimap_recovery: bool = True
    minimap_panel_rect: tuple[int, int, int, int] = (1820, 1030, 2240, 1400)
    minimap_toggle: tuple[int, int] = (2197, 1290)
    minimap_missing_seconds: float = 5.0
    minimap_toggle_interval: float = 60.0
    minimap_toggle_attempts: int = 3
    zone_guard: bool = True
    zone_label_rect: tuple[int, int, int, int] = (2008, 1352, 2146, 1376)  # excludes player count and clock
    zone_label_size: tuple[int, int] = (2240, 1400)  # calibration size; rect scales with captures
    zone_label_threshold: float = 15.0
    zone_label_seconds: float = 2.0
    zone_loading_mean: float = 15.0
    death_limit: int = 2
    death_window_seconds: float = 1800.0
    zero_health: float = 0.01
    minimum_health: float = 0.05
    minimum_health_seconds: float = 3.0
    recovery_health: float = 0.1
    recovery_seconds: float = 2.0
    travel_abort_health: float = 0.25
    attack_health_drop: float = 0.03
    attack_window_seconds: float = 2.0
    attack_hold_seconds: float = 3.0
    retreat_calm_seconds: float = 4.0
    critical_retreat_seconds: float = 8.0
    retreat_drop_epsilon: float = 0.005
    retreat_dash_cooldown: float = 60.0


@dataclass
class NavigationConfig:
    zone: str = "fractured_ground"
    assets_folder: str = "assets/zones"
    window_size: tuple[int, int] = (2240, 1400)
    minimap_rect: tuple[int, int, int, int] = (1840, 1040, 2240, 1400)
    player_screen: tuple[int, int] = (1125, 560)
    # Identity is a direction-only guess. Live tasks refuse clicks until calibrated.
    calibrated: bool = False
    minimap_to_screen: list[list[float]] = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    map_speed_px_per_s: float = 6.0
    arrow_min_value: int = 150
    arrow_min_saturation: int = 60
    arrow_min_mean_value: float = 170.0
    arrow_min_dark_border: float = 0.3
    clearance: float | None = None  # zone default when omitted; 2.0 for zones without one
    snap_radius: int = 8
    lookahead: float = 8.0
    arrival_tolerance: float = 3.0
    waypoint_tolerance: float = 2.0
    click_min: float = 250.0
    click_max: float = 400.0
    precision_click_max: float = 150.0  # road and nearby-waypoint clicks; never extend past target
    near_waypoint_distance: float = 3.0
    road_retry_attempts: int = 2  # bounded short probes before backtracking at a disconnecting road stall
    # A road stall is usually a prop (a wagon, a barrel) far smaller than one map
    # pixel (~370 screen px). Step around it in screen space before learning an edge.
    road_sidestep_attempts: int = 2
    road_sidestep_pixels: float = 350.0
    safe_screen_rect: tuple[int, int, int, int] = (400, 200, 1800, 1100)
    stuck_seconds: float = 4.0
    progress_pixels: float = 0.8
    side_step_seconds: float = 1.0
    stuck_block_radius: int = 3
    max_replans: int = 8  # per spot, not a budget shared by the whole journey
    learned_edge_length: float = 6.0
    zero_motion_seconds: float = 3.0
    zero_motion_pixels: float = 0.3
    backtrack_distance: float = 3.0
    backtrack_attempts: int = 2
    backtrack_seconds: float = 20.0
    trail_spacing: float = 0.5
    trail_limit: int = 128
    retreat_tolerance: float = 0.35
    attack_repeat_radius: float = 3.0
    attack_block_radius: int = 3


@dataclass
class DataConfig:
    folder: str = "training_data"
    # chance per tick of saving a screenshot, as 1 in N. 0 turns it off
    sample_main: int = 200
    sample_bored: int = 20
    sample_nothing_detected: int = 5


@dataclass
class DebugConfig:
    # an OpenCV window showing detections, obstacles and the chosen path.
    # put it on a second monitor: if it takes focus, the bot pauses
    view: bool = False
    log_file: str = "albot.log"
    log_level: str = "INFO"


@dataclass
class Config:
    window_title: str = "Albion Online Client"
    dry_run: bool = False
    models: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    keys: KeyConfig = field(default_factory=KeyConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def key_for_model(self, model: str) -> str | None:
        return os.environ.get(getattr(self.models, f"{model}_key_env"))

    def confidence_for(self, class_name: str) -> float:
        return self.detection.class_confidence.get(class_name, self.detection.min_confidence)


def _build(cls, data: dict[str, Any], path: str):
    """Make a dataclass from a dict, recursing into nested dataclasses."""
    known = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        if name not in known:
            raise ValueError(f"unknown setting '{path}{name}' in config")
        ftype = hints[name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = _build(ftype, value, f"{path}{name}.")
        elif name in ("openers", "rotation"):
            value = [CastStep(**v) for v in value]
        elif name == "fallback":
            value = CastStep(**value)
        elif name == "hud_rects":
            value = [tuple(rect) for rect in value]
        elif isinstance(value, list) and name in ("player_offset", "window_size", "bar", "energy_bar", "gather_bar",
                                                  "minimap_rect", "player_screen", "safe_screen_rect",
                                                  "zone_label_rect", "zone_label_size", "hud_size",
                                                  "minimap_panel_rect", "minimap_toggle"):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load(path: str | None) -> Config:
    if path is None or not os.path.exists(path):
        return Config()
    with open(path, "rb") as f:
        return _build(Config, tomllib.load(f), "")
