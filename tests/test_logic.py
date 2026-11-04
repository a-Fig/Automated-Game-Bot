import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from albot import config as config_mod  # noqa: E402
from albot.actuator import FocusLost, Input  # noqa: E402
from albot.behaviors import Wander  # noqa: E402
from albot.config import Config  # noqa: E402
from albot.geometry import Point, Polygon, find_detour, find_path, path_clear  # noqa: E402
from albot.spells import SpellSlotMapper  # noqa: E402
from albot.state import Latest, Status  # noqa: E402
from albot.vision import choose_player  # noqa: E402
from albot.window import FixedWindow  # noqa: E402

from fakes import PLAYER, World, poly, pred  # noqa: E402


# geometry ------------------------------------------------------------------

def square(x0=10, y0=10, x1=20, y1=20):
    return Polygon((Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)))


def test_path_blocked_and_clear():
    s = square()
    assert s.blocks(Point(0, 15), Point(30, 15))
    assert not s.blocks(Point(0, 0), Point(30, 0))


def test_closing_edge_is_checked():
    # crosses only the edge from the last point back to the first
    assert square().blocks(Point(0, 15), Point(12, 15))


def test_detour_goes_around_an_obstacle():
    wall = square(40, 0, 60, 80)
    start, goal = Point(0, 40), Point(100, 40)
    assert not path_clear(start, goal, [wall])
    # a thick wall needs two corners: no single waypoint sees both ends
    path = find_path(start, goal, [wall], margin=5, bounds=(200, 200))
    assert path is not None and len(path) == 2
    legs = [start] + path + [goal]
    assert all(path_clear(a, b, [wall]) for a, b in zip(legs, legs[1:]))
    assert find_detour(start, goal, [wall], margin=5, bounds=(200, 200)) == path[0]


def test_no_detour_when_goal_is_walled_in():
    box = [square(40, 40, 60, 42), square(40, 58, 60, 60), square(40, 40, 42, 60), square(58, 40, 60, 60)]
    assert find_detour(Point(0, 0), Point(50, 50), box, margin=1) is None


# spells ----------------------------------------------------------------------

class P:
    def __init__(self, x, c="up", conf=0.9):
        self.x, self.class_name, self.confidence = x, c, conf


def test_spell_slots_hold_position_when_one_is_missed():
    m = SpellSlotMapper()
    assert m.map([P(1), P(2)]) == ("cooldown",) * 6       # not calibrated: hold spells
    m.map([P(x) for x in (100, 150, 200, 250, 300, 350)])
    result = m.map([P(100), P(200), P(250), P(300, "up"), P(350, "cooldown")])
    assert result == ("up", "cooldown", "up", "up", "up", "cooldown")


# state -----------------------------------------------------------------------

def test_latest_drops_older_items():
    latest = Latest()
    assert latest.publish(Status(2.0, (), True, False))
    assert not latest.publish(Status(1.0, (), True, True))
    assert latest.get().timestamp == 2.0 and latest.dropped == 1


def test_player_is_the_detection_nearest_the_usual_spot():
    cfg = Config()
    other = pred("player", 100, 100, confidence=0.99)
    me = pred("player", PLAYER[0] + 5, PLAYER[1] - 3, confidence=0.7)
    p, detected = choose_player([other, me], cfg, (1000, 800))
    assert detected and p == Point(me.x, me.y)


# config ------------------------------------------------------------------------

def test_config_loads_nested_values_and_rejects_typos():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.toml")
        with open(path, "w") as f:
            f.write('[combat]\nrotation = [{slot = "w", delay = 0.3}]\n[detection]\nplayer_offset = [0, -80]\n')
        cfg = config_mod.load(path)
        assert cfg.combat.rotation[0].slot == "w" and cfg.combat.rotation[0].delay == 0.3
        assert cfg.detection.player_offset == (0, -80)
        with open(path, "w") as f:
            f.write("[combat]\nshield_wiat = 2\n")
        with pytest.raises(ValueError, match="shield_wiat"):
            config_mod.load(path)


# behaviors -------------------------------------------------------------------

def close_mob():
    return pred("mob", PLAYER[0] + 60, PLAYER[1])


def test_threat_with_shield_ready_shields_first():
    w = World()
    w.see([pred("red", 300, 300), close_mob()])
    assert w.tick()[0] == ("press", "d")


def test_threat_without_shield_falls_through_to_fighting():
    w = World()
    w.spells = ["up", "up", "up", "up", "cooldown", "up"]
    w.see([pred("red", 300, 300), close_mob()])
    actions = w.tick()
    assert ("press", "d") not in actions
    assert ("press", "r") in actions                       # opener
    assert any(a[0] == "click" for a in actions)           # clicked the mob
    assert ("press", "e") in actions                       # warbow rotation


def test_heal_waits_until_health_recovers():
    w = World()
    w.see([], player=True)
    w.see([pred("silver", 50, 50, 0.3)])   # something on screen, nothing actionable
    w.status(low=True)
    w.clock.at(12, lambda: w.status(low=False))
    start = w.clock.now()
    actions = w.tick()
    assert actions[0][0] == "move" and actions[1] == ("press", "f")
    assert actions.count(("press", "f")) == 1
    # waiting for recovery at 12s + 5s settle
    assert w.clock.now() - start >= 17


def test_heal_wait_ends_early_on_a_threat():
    w = World()
    w.see([pred("silver", 50, 50, 0.3)])
    w.status(low=True)
    w.clock.at(6, lambda: w.see([pred("red", 300, 300)]))
    start = w.clock.now()
    w.tick()
    assert w.clock.now() - start < 8


def test_blocked_mob_gets_a_detour_walk():
    w = World()
    mob = (PLAYER[0] + 200, PLAYER[1])
    wall = [(PLAYER[0] + 80, PLAYER[1] - 40), (PLAYER[0] + 120, PLAYER[1] - 40),
            (PLAYER[0] + 120, PLAYER[1] + 40), (PLAYER[0] + 80, PLAYER[1] + 40)]
    w.see([pred("mob", *mob)], [poly(wall)])
    actions = w.tick()
    assert actions[0][0] == "walk"
    assert not any(a[0] == "click" for a in actions)


def test_farm_node_that_never_gathers_gets_blacklisted():
    w = World()
    node = pred("fiber", PLAYER[0] + 150, PLAYER[1] + 50)
    w.see([node])
    w.tick()
    assert len(w.ctx.resources.failed) == 1
    w.input.actions.clear()
    w.see([pred("fiber", *PLAYER)])  # the camera followed the approach to the node
    w.tick()
    assert not any(a[0] == "click" for a in w.input.actions)


def test_farm_succeeds_when_the_loading_bar_comes_and_goes():
    w = World()
    w.see([pred("hide", PLAYER[0] + 150, PLAYER[1])])
    w.clock.at(1, lambda: w.status(loading=True))
    w.clock.at(6, lambda: w.status(loading=False))
    w.clock.at(6.1, lambda: w.see([]))
    w.tick()
    assert w.brain.counts["farm"] == 1


def test_nothing_detected_presses_esc_then_pauses():
    w = World()
    for _ in range(40):
        w.see([], player=False)
        w.tick()
        if w.brain.paused.is_set():
            break
    assert ("press", "esc") in w.input.actions
    assert w.brain.paused.is_set() and w.workers.paused


def test_waits_stop_when_focus_is_lost():
    w = World()
    w.see([])
    w.clock.at(1, lambda: setattr(w.window, "active", False))
    with pytest.raises(FocusLost):
        w.ctx.wait(5)


def test_real_input_refuses_to_type_without_focus():
    inp = Input(FixedWindow(10, 10, active=False), Config())
    with pytest.raises(FocusLost):
        inp.press("q")


def test_wander_points_are_reachable_and_not_too_close():
    w = World()
    scene = w.see([pred("silver", 50, 50, 0.3)])
    for _ in range(20):
        p = Wander.pick_point(w.ctx, scene)
        assert p is not None
        assert not scene.is_close(p, w.config.behavior.wander_min_distance)
        assert scene.reachable(p)


def test_wander_only_after_being_bored():
    w = World()
    w.see([pred("silver", 50, 50, 0.3)])
    for _ in range(w.config.behavior.bored_ticks + 1):
        w.tick()
    assert not any(a[0] == "walk" for a in w.input.actions)
    w.see([pred("silver", 50, 50, 0.3)])
    w.tick()
    assert any(a[0] == "walk" for a in w.input.actions)


# end to end: real worker threads and brain loop, fake models and capture ----------

def test_threads_brain_and_workers_run_together():
    import threading
    import time
    from types import SimpleNamespace

    from albot.actuator import DryRunInput
    from albot.brain import Brain, Context
    from albot.vision import Perceiver
    from albot.workers import Workers

    def model(preds):
        return SimpleNamespace(infer=lambda image: [SimpleNamespace(predictions=preds)])

    slots = [pred("up", x, 700) for x in (300, 350, 400, 450, 500, 550)]
    calls = {"scene": 0}

    def scene_infer(image):
        calls["scene"] += 1
        if calls["scene"] == 3:
            raise RuntimeError("one bad frame")   # workers must survive this
        return [SimpleNamespace(predictions=[pred("player", *PLAYER), pred("mob", PLAYER[0] + 50, PLAYER[1])])]

    models = SimpleNamespace(
        scene=SimpleNamespace(infer=scene_infer), obstacles=model([]), spells=model(slots),
        loading_bar=model([]), health=model([SimpleNamespace(class_name="high")]),
    )
    cfg = Config()
    cfg.health.model_fallback = True  # this test uses opaque frames, not HUD pixels
    window = FixedWindow(1000, 800)
    capture = SimpleNamespace(grab=lambda rect: object())
    perceiver = Perceiver(cfg, models, capture, window)
    workers = Workers(perceiver, cfg)
    inp = DryRunInput()
    brain = Brain(Context(cfg, inp, workers, window))
    workers.start()
    t = threading.Thread(target=brain.run, daemon=True)
    t.start()
    time.sleep(1.5)
    brain.quit.set()
    workers.stop()
    t.join(timeout=5)

    assert workers.errors == 1
    assert brain.errors == 0
    assert brain.counts["fight_close"] >= 1
    assert ("press", "r") in inp.actions and any(a[0] == "click" for a in inp.actions)


def test_example_config_matches_the_defaults():
    path = os.path.join(os.path.dirname(__file__), "..", "albot.example.toml")
    assert config_mod.load(path) == Config()
