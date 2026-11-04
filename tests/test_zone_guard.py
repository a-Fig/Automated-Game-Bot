import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from albot.actuator import DryRunInput, Input
from albot.behaviors import Farm, Fight, Wander
from albot.brain import Context
from albot.config import Config
from albot.geometry import Point
from albot.navigation import ExitGuard, Location, Zone
from albot.safety import SafetyStop, ZoneGuard
from albot.state import Status
from albot.vision import Perceiver, build_scene
from albot.window import FixedWindow
from albot.workers import Workers
from fakes import World, pred


MATRIX = np.array([[373.34, 274.96], [-59.97, 341.67]])
PLAYER = Point(1125, 560)


def world_frame(label=100):
    image = np.full((1400, 2240, 3), 80, np.uint8)
    image[1365:1395, 1960:2160] = label
    image[56:68, 105:245] = (25, 30, 180)
    image[72:80, 105:245] = (180, 30, 25)
    return image


def reading(frame, at):
    return Status(at, (), True, False, health_fraction=1, frame=frame)


def test_zone_label_change_holds_for_two_seconds_then_latches():
    guard = ZoneGuard(Config())
    frame = world_frame()
    assert not guard.observe(reading(frame, 0), 0)
    frame[1365:1395, 1960:2160] = 132  # mutate source: startup crop must be a copy
    assert not guard.observe(reading(frame, 1), 1)
    assert not guard.observe(reading(frame, 2), 2)
    assert "zone label difference 32.0" in guard.observe(reading(frame, 3.1), 3.1)
    assert guard.observe(reading(world_frame(), 4), 4)  # restored label cannot clear the latch


def test_zone_guard_ignores_noise_transients_and_stale_samples():
    guard = ZoneGuard(Config())
    assert not guard.observe(reading(world_frame(), 0), 0)
    for at in (1, 2, 3):
        assert not guard.observe(reading(world_frame(103), at), at)
    assert not guard.observe(reading(world_frame(132), 4), 4)
    assert not guard.observe(reading(world_frame(), 5), 5)
    assert not guard.observe(reading(world_frame(132), 6), 6)
    assert not guard.observe(reading(world_frame(132), 6), 10)  # stale image cannot confirm a hold
    assert not guard.observe(reading(world_frame(), 10), 10)


def test_zone_label_guard_is_independent_of_numeric_health_reading():
    guard = ZoneGuard(Config())
    for at, label in ((0, 100), (1, 132), (2, 132)):
        status = Status(at, (), False, True, frame=world_frame(label))
        assert not guard.observe(status, at)
    status = Status(3.1, (), False, True, frame=world_frame(132))
    assert "zone label" in guard.observe(status, 3.1)


@pytest.mark.parametrize("value", [0, 14])
def test_loading_frame_stops_immediately_even_before_startup_reference(value):
    guard = ZoneGuard(Config())
    assert "loading frame" in guard.observe(reading(np.full((1400, 2240, 3), value, np.uint8), 0), 0)


def test_label_rect_scales_and_pil_rgb_matches_numpy_bgr():
    cfg = Config()
    cfg.safety.zone_label_size = (100, 50)
    cfg.safety.zone_label_rect = (10, 20, 50, 30)
    guard = ZoneGuard(cfg)
    frame = np.full((50, 100, 3), 80, np.uint8)
    frame[20:30, 10:50] = (70, 120, 180)
    assert not guard.observe(reading(frame, 0), 0)
    large = cv2.resize(frame, (200, 100), interpolation=cv2.INTER_NEAREST)
    assert not guard.observe(reading(Image.fromarray(large[:, :, ::-1]), 1), 1)
    large[40:60, 20:100] = (100, 150, 210)
    for at in (2, 3):
        assert not guard.observe(reading(large, at), at)
    assert guard.observe(reading(large, 4.1), 4.1)


def test_farming_zone_stop_releases_input_and_cannot_auto_resume(caplog):
    w = World(zone_guard=True)
    w.see([])
    w.status(frame=world_frame())
    w.ctx.walk_to(Point(700, 400))
    assert not w.config.behavior.travel_goal  # this guard also protects ordinary farming
    w.clock.sleep(0.1)
    w.status(frame=np.zeros((1400, 2240, 3), np.uint8))
    w.input.actions.clear()
    with pytest.raises(SafetyStop):
        w.ctx.press("q")
    assert w.brain.quit.is_set()
    assert w.input.actions == [("press", "s"), ("release",)]
    assert any(r.levelname == "CRITICAL" and "zone changed" in r.message for r in caplog.records)
    w.status(frame=world_frame())
    with pytest.raises(SafetyStop):
        w.ctx.walk_to(Point(700, 400))
    assert len(w.input.actions) == 2


def test_loading_latches_before_models_and_survives_latest_frame_replacement():
    cfg = Config()
    model = SimpleNamespace(infer=lambda frame: pytest.fail("loading must stop before model inference"))
    p = Perceiver(cfg, SimpleNamespace(spells=model), None, FixedWindow(2240, 1400))
    # Use the current clock so the raw frame is fresh, without taking a screenshot.
    import time
    p.grab = lambda: (np.zeros((1400, 2240, 3), np.uint8), (2240, 1400), time.monotonic(), 0.0)
    workers = Workers(p, cfg)
    raw = p.scan_status(False)
    assert workers.zone_guard.reason and raw.health_fraction is None
    workers.statuses.publish(reading(world_frame(), time.monotonic()))
    ctx = Context(cfg, DryRunInput(), workers, p.window)
    with pytest.raises(SafetyStop):
        ctx.check_safety()
    assert ctx.quit.is_set()  # the overwritten black capture was not forgotten


def test_real_release_uses_only_key_up_and_mouse_up_with_fake_devices(monkeypatch):
    import sys
    actions = []
    monkeypatch.setitem(sys.modules, "keyboard", SimpleNamespace(release=lambda key: actions.append(("up", key))))
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(mouseUp=lambda button: actions.append(("mouse_up", button))))
    Input(FixedWindow(2240, 1400, active=False), Config()).release()
    assert ("up", "q") in actions and ("up", "space") in actions
    assert ("mouse_up", "left") in actions and ("mouse_up", "right") in actions


def exit_world(point):
    w = World(zone_guard=True)
    scene = build_scene(w.clock.now(), (2240, 1400), [pred("player", *PLAYER)], [], w.config)
    w.workers.scenes.publish(scene)
    w.status(frame=world_frame())
    # Use real zone metadata but a controlled fix, keeping this independent of locator tests.
    w.ctx.exit_guard.metadata = Zone.load("fractured_ground", learned=False).metadata
    w.ctx.exit_guard.locator = SimpleNamespace(locate=lambda frame, at: Location(*point, None, 0.95))
    return w


def target(delta):
    return Point(*(np.asarray(PLAYER)+MATRIX@np.asarray(delta)))


def test_wander_picks_another_direction_and_farm_chase_and_direct_clicks_refuse_exit():
    w = exit_world((194.1, 96.8))
    choices = iter((math.pi, 0.7, 0, 0.7))
    w.ctx.random = SimpleNamespace(uniform=lambda *args: next(choices))
    point = Wander.pick_point(w.ctx, w.ctx.scene(), attempts=2)
    assert point is not None and point.x > PLAYER.x
    inward = target((-1, 0))
    assert "193.10,96.80" in ("" if w.ctx.movement_allowed(inward) else w.ctx.exit_guard.reason)
    for behavior in (Farm(), Fight("fight_close", True)):
        name = "fiber" if behavior.name == "farm" else "mob"
        scene = build_scene(w.clock.now(), (2240, 1400), [pred("player", *PLAYER), pred(name, *inward)], [], w.config)
        assert behavior.plan(w.ctx, scene, w.ctx.status()) is None
    for send in (w.ctx.walk_to, w.ctx.click, w.ctx.move_to):
        with pytest.raises(SafetyStop):
            send(inward)
    assert not any(a[0] in ("walk", "click", "move") for a in w.input.actions)


def test_inside_buffer_allows_only_monotonically_outward_motion_and_logs_use(caplog):
    w = exit_world((186.0, 96.8))
    assert not w.ctx.movement_allowed(target((-0.5, -1)))  # toward gate
    assert not w.ctx.movement_allowed(target((0, 0)))
    with caplog.at_level("INFO"):
        w.ctx.walk_to(target((0, 1)))  # still in the buffer, but farther from both centres
    assert any(a[0] == "walk" for a in w.input.actions)
    assert "exit buffer escape: moving strictly outward from NW" in caplog.text


def test_only_active_travel_goal_portal_is_exempt():
    w = exit_world((271.7, 124.2))
    w.config.behavior.travel_goal = "portal:NE"
    assert not w.ctx.movement_allowed(target((1, 0)))  # farming/chase still protected
    w.ctx.active_behavior = "travel"
    assert w.ctx.movement_allowed(target((1, 0)))
    w.config.behavior.travel_goal = "portal:NW"
    assert not w.ctx.movement_allowed(target((1, 0)))


def test_missing_fix_skips_exit_check_and_wrong_size_warns_once(caplog):
    guard = ExitGuard(Config().navigation)
    assert guard.allowed(world_frame(), 0, 0, target((-1, 0)))  # no minimap arrow
    small = np.full((700, 1120, 3), 80, np.uint8)
    for at in (1, 2, 3):
        assert guard.allowed(small, at, at, target((-1, 0)))
    assert len([r for r in caplog.records if "exit projection skipped" in r.message]) == 1


def test_zone_label_guard_still_stops_on_an_unsupported_exit_projection_size():
    w = World(zone_guard=True)
    small = cv2.resize(world_frame(), (1120, 700), interpolation=cv2.INTER_NEAREST)
    w.status(frame=small)
    assert w.ctx.movement_allowed(PLAYER)  # projection skipped, not guessed
    w.ctx.check_safety()
    changed = cv2.resize(world_frame(132), (1120, 700), interpolation=cv2.INTER_NEAREST)
    for _ in range(2):
        w.clock.sleep(1)
        w.status(frame=changed)
        w.ctx.check_safety()
    w.clock.sleep(1.1)
    w.status(frame=changed)
    with pytest.raises(SafetyStop):
        w.ctx.check_safety()
    assert w.brain.quit.is_set()


def test_zone_stop_stays_latched_and_releases_even_if_stop_key_fails():
    w = World(zone_guard=True)
    w.status(frame=np.zeros((1400, 2240, 3), np.uint8))
    w.input.press = lambda key: (_ for _ in ()).throw(RuntimeError("fake device failure"))
    with pytest.raises(SafetyStop):
        w.ctx.check_safety()
    assert w.brain.quit.is_set() and w.input.actions == [("release",)]


def test_every_saved_border_signpost_has_an_exit_disc():
    zone = Zone.load("fractured_ground", learned=False)
    assert {d["portal"] for d in zone.metadata["exit_no_go"]} == {"NW", "NE", "SE", "SW"}
    for name, portal in zone.metadata["portals"].items():
        assert any(d["portal"] == name and d["center"] == portal["marker"]
                   and d["radius"] >= 8 for d in zone.metadata["exit_no_go"])
