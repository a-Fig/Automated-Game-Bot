import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from albot.actuator import FocusLost, Input
from albot.behaviors import Heal
from albot.config import Config
from albot.datacollect import Saver
from albot.geometry import Point
from albot.safety import DeathGuard, SafetyStop
from albot.state import Status
from albot.vision import Models, Perceiver, read_health
from albot.window import FixedWindow
from fakes import PLAYER, World, poly, pred


def hud(fraction=1.0):
    frame = np.zeros((1400, 2240, 3), dtype=np.uint8)
    frame[56:68, 105:245] = (30, 30, 30)
    frame[56:68, 105:105 + round(140 * fraction)] = (25, 30, 180)
    frame[72:80, 105:245] = (180, 30, 25)
    return frame


def model(predictions):
    return SimpleNamespace(infer=lambda frame: [SimpleNamespace(predictions=predictions)])


def perceiver(config=None, health=None):
    models = SimpleNamespace(spells=model([]), health=health, loading_bar=model([]))
    return Perceiver(config or Config(), models, None, FixedWindow(2240, 1400))


@pytest.mark.parametrize("fraction", [0.0, 0.2, 0.4, 0.8, 1.0])
def test_health_fill_and_low_threshold_from_fake_frames(fraction):
    frame = hud(fraction)
    status = perceiver().status_from(frame, 1, False)
    assert status.health_fraction == pytest.approx(fraction)
    assert status.low_health == (fraction <= 0.4)
    assert status.frame is frame


def test_health_ignores_white_text_and_supports_pil_rgb():
    frame = hud(0.5)
    frame[62, 110:120] = (240, 240, 240)
    frame[62, 220:230] = (240, 240, 240)
    assert read_health(frame, Config()) == 0.5
    assert read_health(Image.fromarray(frame[:, :, ::-1]), Config()) == 0.5


@pytest.mark.parametrize("frame", [np.zeros((800, 1000, 3), dtype=np.uint8),
                                 np.zeros((1400, 2240, 3), dtype=np.uint8),
                                 np.full((1400, 2240, 3), 255, dtype=np.uint8), object()])
def test_unreadable_hud_is_unknown_not_death(frame):
    status = perceiver().status_from(frame, 1, False)
    assert status.health_fraction is None
    assert status.low_health
    guard = DeathGuard(Config())
    assert not guard.observe(status, 1)
    assert not guard.deaths


def test_health_coordinates_and_threshold_are_configurable():
    cfg = Config()
    cfg.health.window_size = (200, 100)
    cfg.health.bar = (10, 10, 110, 20)
    cfg.health.energy_bar = (10, 20, 110, 30)
    cfg.health.low_threshold = 0.6
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:20, 10:60] = (0, 0, 150)
    status = perceiver(cfg).status_from(frame, 1, False)
    assert status.health_fraction == 0.5 and status.low_health
    cfg.health.bar = (-1, 10, 110, 20)
    assert read_health(frame, cfg) is None


def test_health_model_fallback_is_opt_in_and_only_used_for_unknown_hud():
    calls = []
    health = SimpleNamespace(infer=lambda image: calls.append(image) or
                             [SimpleNamespace(predictions=[SimpleNamespace(class_name="good")])])
    cfg = Config()
    p = perceiver(cfg, health)
    assert p.status_from(object(), 1, False).low_health
    assert not calls
    cfg.health.model_fallback = True
    assert not p.status_from(object(), 2, False).low_health
    assert len(calls) == 1
    assert p.status_from(hud(0.2), 3, False).low_health
    assert len(calls) == 1


def test_disabled_health_fallback_does_not_load_the_model(monkeypatch):
    import sys
    calls = []
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(
        get_model=lambda name, key: calls.append(name) or object()))
    cfg = Config()
    assert Models(cfg).health is None
    assert cfg.models.health not in calls
    cfg.health.model_fallback = True
    assert Models(cfg).health is not None
    assert calls[-1] == cfg.models.health


def reading(timestamp, health):
    return Status(timestamp, (), True, health is None or health <= 0.4, health_fraction=health)


def test_death_episode_is_counted_once_and_recovery_must_be_sustained():
    guard = DeathGuard(Config())
    assert guard.observe(reading(1, 0), 1)
    for t, health in [(2, 0), (3, None), (4, 0.8), (5, 0), (6, 0.8), (7, 0.8)]:
        assert not guard.observe(reading(t, health), t)
        assert guard.unsafe
    assert not guard.observe(reading(8, 0.8), 8)
    assert not guard.unsafe
    assert guard.observe(reading(9, 0), 9)
    assert guard.deaths == [1, 9]


def test_minimum_health_requires_fresh_continuous_samples():
    guard = DeathGuard(Config())
    for t in [1, 2, 3]:
        assert not guard.observe(reading(t, 0.03), t)
        assert not guard.unsafe  # healing/escape is allowed until death is confirmed
    assert guard.observe(reading(4, 0.03), 4)
    guard = DeathGuard(Config())
    assert not guard.observe(reading(1, 0.03), 1)
    assert not guard.observe(reading(1, 0.03), 5)  # stale is not a four-second hold
    assert not guard.observe(reading(5, 0.03), 5)
    assert not guard.deaths
    assert not guard.observe(reading(6, 0.8), 6)
    assert not guard.unsafe


def test_deaths_expire_outside_configurable_window():
    cfg = Config()
    cfg.safety.death_window_seconds = 5
    guard = DeathGuard(cfg)
    assert guard.observe(reading(1, 0), 1)
    for t in [2, 3, 4]:
        guard.observe(reading(t, 1), t)
    assert guard.observe(reading(10, 0), 10)
    assert guard.deaths == [10]


def test_death_guard_checks_inside_healing_wait_and_quits_on_second_death(tmp_path, caplog):
    w = World()
    w.ctx.saver = Saver(str(tmp_path), lambda: hud(0))
    w.see([])
    w.status(low=True, health=0.3)
    w.clock.at(0.5, lambda: w.status(health=0))
    w.tick()
    assert len(w.ctx.death_guard.deaths) == 1 and not w.brain.quit.is_set()
    assert w.clock.now() < 1001
    assert w.input.actions[0][0] == "move"
    assert w.input.actions.count(("press", "f")) == 1
    actions = list(w.input.actions)
    for _ in range(10):
        w.tick()
    assert w.input.actions == actions  # no clicks or keys on the death screen
    w.status(low=False, health=1)
    # The tick loop keeps observing recovery while input is suspended.
    for _ in range(24):
        w.tick()
    assert not w.ctx.death_guard.dead
    w.status(health=0)
    w.clock.sleep(0.02)
    w.tick()
    assert w.brain.quit.is_set()
    records = [json.loads(line) for line in (tmp_path / "deaths" / "deaths.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(r["timestamp"] for r in records)
    assert len(list((tmp_path / "deaths").glob("*.png"))) == 2
    assert "quitting" in caplog.text


def test_minimum_health_can_escape_before_death_is_confirmed():
    w = World()
    w.see([])
    w.status(low=True, health=0.03)
    w.clock.at(0.5, lambda: w.status(low=False, health=1))
    w.tick()
    assert ("press", "f") in w.input.actions
    assert not w.ctx.death_guard.deaths


def test_death_screenshot_is_triggering_frame_not_later_capture(tmp_path):
    frame = hud(0)
    saver = Saver(str(tmp_path), lambda: pytest.fail("must use triggering frame"))
    saver.death(0, frame)
    screenshot = next((tmp_path / "deaths").glob("*.png"))
    assert np.array_equal(np.asarray(Image.open(screenshot)), frame[:, :, ::-1])


def test_fake_frame_flows_from_health_sensor_to_quit_and_exact_death_record(tmp_path):
    cfg = Config()
    cfg.safety.death_limit = 1
    w = World(cfg)
    w.see([pred("mob", PLAYER[0] + 20, PLAYER[1])])
    frame = hud(0)
    status = perceiver(cfg).status_from(frame, w.clock.now(), False)
    w.workers.statuses.publish(status)
    w.ctx.saver = Saver(str(tmp_path), lambda: pytest.fail("must use status frame"))
    w.tick()
    assert w.brain.quit.is_set() and not w.input.actions
    record = json.loads((tmp_path / "deaths" / "deaths.jsonl").read_text())
    assert record["health_fraction"] == 0
    saved = Image.open(tmp_path / "deaths" / record["screenshot"])
    assert np.array_equal(np.asarray(saved), frame[:, :, ::-1])


@pytest.mark.parametrize("method", ["wait", "wait_until"])
def test_death_interrupts_gather_and_other_waits(method):
    w = World()
    w.see([])
    w.clock.at(0.1, lambda: w.status(health=0))
    with pytest.raises(SafetyStop):
        if method == "wait":
            w.ctx.wait(10)
        else:
            w.ctx.wait_until(lambda: False, 10)
    assert w.ctx.death_guard.dead
    assert w.clock.now() < 1001


def test_death_guard_runs_even_without_a_scene_and_death_limit_is_configurable():
    cfg = Config()
    cfg.safety.death_limit = 1
    w = World(cfg)
    w.status(health=0)
    w.tick()
    assert w.brain.quit.is_set() and not w.input.actions


def test_unknown_and_stale_health_suspend_actions():
    w = World()
    w.see([pred("mob", PLAYER[0] + 20, PLAYER[1])])
    w.status(health=None)
    w.tick()
    assert not w.input.actions
    w.status(health=1)
    w.clock.on_sleep = None
    w.clock.sleep(3)
    w.tick()
    assert not w.input.actions and not w.ctx.death_guard.deaths


@pytest.mark.parametrize("change", [{"death_limit": 0}, {"minimum_health_seconds": 0},
                                    {"recovery_health": 0.01}])
def test_invalid_safety_config_is_rejected(change):
    cfg = Config()
    for key, value in change.items():
        setattr(cfg.safety, key, value)
    with pytest.raises(ValueError, match="death guard"):
        DeathGuard(cfg)


def test_escape_aims_away_from_nearby_centroid_and_casts_once():
    w = World()
    w.config.behavior.priority = ["heal"]
    scene = w.see([pred("mob", PLAYER[0] + 100, PLAYER[1] - 50),
                   pred("red", PLAYER[0] + 100, PLAYER[1] + 50),
                   pred("mob", 0, 0)])  # distant detection must not skew the centroid
    w.status(low=True, health=0.3)
    plan = Heal().plan(w.ctx, scene, w.ctx.status())
    Heal().run(w.ctx, plan)
    assert w.input.actions == [("move", round(PLAYER[0] - 300), round(PLAYER[1])), ("press", "f")]


def test_escape_without_enemies_chooses_open_ground():
    w = World()
    x, y = PLAYER
    wall = poly([(x + 80, y - 70), (x + 120, y - 70),
                 (x + 120, y + 70), (x + 80, y + 70)])
    scene = w.see([], [wall])
    point = Heal.escape_point(scene, w.config)
    assert point is not None and scene.reachable(point)
    assert point.dist(scene.player) == pytest.approx(300)


def test_escape_near_window_edge_stays_in_bounds():
    w = World()
    scene = replace(w.see([]), player=Point(20, 20))
    point = Heal.escape_point(scene, w.config)
    assert point is not None
    assert 0 <= point.x < scene.window_size[0] and 0 <= point.y < scene.window_size[1]


def test_real_aim_refuses_mouse_movement_without_focus(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(
        moveTo=lambda *args, **kwargs: pytest.fail("must not move mouse")))
    with pytest.raises(FocusLost):
        Input(FixedWindow(10, 10, active=False), Config()).move_to(Point(5, 5))


def test_death_or_quit_prevents_further_inputs():
    w = World()
    w.see([])
    w.status(health=0)
    for action in [lambda: w.ctx.press("q"), lambda: w.ctx.move_to(Point(1, 1)),
                   lambda: w.ctx.walk_to(Point(1, 1)), lambda: w.ctx.click(Point(1, 1))]:
        with pytest.raises(SafetyStop):
            action()
    assert not w.input.actions
    w = World()
    w.see([])
    w.brain.quit.set()
    with pytest.raises(SafetyStop):
        w.ctx.wait(1)


def test_gather_bar_is_read_from_green_pixels_not_the_model():
    from albot.vision import read_gather_bar
    frame = hud(1.0)
    assert read_gather_bar(frame, Config()) is False
    frame[1024:1037, 1020:1110] = (40, 200, 60)  # BGR green fill, as in a live gather
    assert read_gather_bar(frame, Config()) is True
    assert read_gather_bar(Image.fromarray(frame[:, :, ::-1]), Config()) is True
    assert read_gather_bar(np.zeros((10, 10, 3), dtype=np.uint8), Config()) is None
    status = perceiver().status_from(frame, 1, True)
    assert status.loading_bar is True
