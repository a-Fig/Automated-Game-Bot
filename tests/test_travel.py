from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from albot.behaviors import Plan, Travel
from albot.config import Config
from albot.geometry import Point
from albot.navigation import Location, NavigationStep, WalkGrid
from albot.safety import SafetyStop, TravelGuard
from albot.state import Status
from albot.vision import build_scene
from albot.window import FixedWindow
from fakes import World, pred


PLAYER = (1125, 560)
TRAVEL_CLICK = Point(1450, 560)


def travel_world(calibrated=True):
    cfg = Config()
    cfg.behavior.travel_goal = "portal:NE"
    cfg.navigation.calibrated = calibrated
    cfg.navigation.minimap_to_screen = [[373.34, 274.96], [-59.97, 341.67]]
    cfg.navigation.map_speed_px_per_s = 0.529
    w = World(cfg)
    w.window._rect = FixedWindow(2240, 1400).rect()
    crop = cv2.imread(str(Path(__file__).parent / "fixtures/navigation/frame_minimap_1.png"))
    frame = np.zeros((1400, 2240, 3), np.uint8)
    frame[1040:, 1840:] = crop
    w.status(frame=frame)
    see(w)
    return w


def see(w, predictions=()):
    scene = build_scene(w.clock.now(), (2240, 1400),
                        [*predictions, pred("player", *PLAYER)], [], w.config)
    w.workers.scenes.publish(scene)
    w.status()
    return scene


def fake_navigation(monkeypatch, steps):
    calls = []
    class FakeNavigate:
        def __init__(self, goal, config):
            calls.append(("init", goal, config.calibrated))
            self.position = Location(235, 145, None, 0.9)
            self.locator = SimpleNamespace(locate=lambda frame, now: self.position)
            self.grid = WalkGrid(np.ones((360, 400), bool))
            self.route = []
        def tick(self, frame, now):
            calls.append(("tick", now))
            step = steps.pop(0)
            result = step() if callable(step) else step
            return replace(result, location=self.position) if result.location is None else result
    monkeypatch.setattr("albot.navigation.NavigateTo", FakeNavigate)
    return calls


def seed_route(w):
    travel = next(b for b in w.brain.behaviors if b.name == "travel")
    travel._start_task(w.ctx, w.config.behavior.travel_goal)
    travel.trail = [Point(220, 145), Point(225, 145), Point(230, 145)]
    return travel


def test_travel_off_by_default_and_priority_after_combat():
    cfg = Config()
    assert cfg.behavior.travel_goal == ""
    names = cfg.behavior.priority
    assert names.index("fight_close") < names.index("travel") < names.index("farm") < names.index("wander")
    w = World(cfg)
    scene = w.see([])
    assert Travel().plan(w.ctx, scene, w.ctx.status()) is None


def test_old_priority_list_inserts_travel_but_unsafe_combat_configuration_is_rejected():
    cfg = Config()
    cfg.behavior.priority.remove("travel")
    cfg.behavior.travel_goal = "portal:NE"
    w = World(cfg)
    names = [b.name for b in w.brain.behaviors]
    assert names.index("travel") == names.index("fight_close")+1
    cfg.behavior.disabled = ["fight_close"]
    with pytest.raises(ValueError, match="travel requires"):
        World(cfg)
    cfg = Config()
    cfg.behavior.priority = ["farm"]
    World(cfg)  # travel off preserves custom legacy priorities


def test_travel_yields_to_close_mob_and_cancels_previous_walk(monkeypatch):
    fake_navigation(monkeypatch, [])  # any navigator invocation would fail this test
    w = travel_world()
    w.ctx.travel_moving = True
    mob = pred("mob", PLAYER[0]+60, PLAYER[1])
    see(w, [mob])
    w.tick()
    assert w.input.actions[0] == ("press", "s")
    assert ("click", round(mob.x), round(mob.y)) in w.input.actions
    assert w.brain.counts["fight_close"] == 1 and not w.ctx.travel_moving
    assert w.ctx.active_behavior == ""


def test_combat_published_during_navigation_preempts_before_click(monkeypatch):
    w = travel_world()
    mob = pred("mob", PLAYER[0]+60, PLAYER[1])
    def plan_then_enemy():
        see(w, [mob])
        return NavigationStep("move", click=TRAVEL_CLICK)
    fake_navigation(monkeypatch, [plan_then_enemy])
    w.tick()
    assert ("click", *TRAVEL_CLICK) not in w.input.actions
    assert ("click", round(mob.x), round(mob.y)) in w.input.actions
    assert w.brain.counts["fight_close"] == 1


def reading(timestamp, hp):
    return Status(timestamp, (), True, hp is None or hp < 0.4, health_fraction=hp)


@pytest.mark.parametrize("hp,elapsed,trigger", [(0.96, 1, True), (0.97, 1, False),
                                               (0.99, 1, False), (0.9, 2.1, False)])
def test_under_attack_drop_threshold_and_window(hp, elapsed, trigger):
    guard = TravelGuard(Config())
    assert not guard.observe(reading(10, 1), 10, enabled=True, fighting=False)
    assert guard.observe(reading(10+elapsed, hp), 10+elapsed, True, False) == trigger
    assert guard.under_attack(10+elapsed) == trigger


def test_damage_during_active_fight_is_not_a_new_attack_signal():
    guard = TravelGuard(Config())
    guard.observe(reading(10, 1), 10, True, False)
    assert not guard.observe(reading(11, 0.8), 11, True, True)
    assert not guard.observe(reading(11, 0.8), 11, True, True)  # same sample during cast wait
    assert not guard.observe(reading(12, 0.79), 12, True, False)
    assert not guard.under_attack(12)


def test_attack_signal_requires_fresh_health_and_clears_when_travel_disabled():
    guard = TravelGuard(Config())
    guard.observe(reading(10, 1), 10, True, False)
    assert not guard.observe(reading(10.5, 0.5), 14, True, False)
    assert not guard.observe(reading(14, 0.5), 14, True, False)
    assert guard.observe(reading(15, 0.4), 15, True, False)
    assert not guard.under_attack(19)
    guard.observe(reading(15, 0.4), 15, False, False)
    assert not guard.under_attack(15)


def test_under_attack_engages_nearest_mob_without_red_label_or_close_range(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    w.status(spells=["up", "up", "up", "up", "cooldown", "up"])
    w.ctx.check_safety()  # establish undamaged baseline outside a fight
    w.ctx.current_plan = Plan("fight_close", "old fight")  # previous plan is not active combat
    w.clock.sleep(0.5)
    far, nearest = pred("mob", 1900, 560), pred("mob", 1600, 560)
    see(w, [far, nearest])
    w.status(health=0.95)
    w.ctx.travel_moving = True
    w.tick()
    assert not w.ctx.scene().threats
    assert w.ctx.under_attack() and w.ctx.danger(count_low_health=False)
    assert w.input.actions[0] == ("press", "s")
    assert ("click", 1600, 560) in w.input.actions
    assert w.brain.counts["fight_close"] == 1


def test_under_attack_is_also_a_shield_threat(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    w.ctx.check_safety()
    w.clock.sleep(0.5)
    see(w, [pred("mob", 1900, 560)])
    w.status(health=0.95)
    w.clock.on_sleep = lambda: see(w, [pred("mob", 1900, 560)])
    w.tick()
    assert ("press", "d") in w.input.actions
    assert w.brain.counts["shield"] == 1


def test_attack_without_mob_detection_retreats_instead_of_farming_or_forward_travel(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(spells=["up", "up", "up", "up", "cooldown", "up"])
    w.ctx.check_safety()
    w.clock.sleep(0.5)
    see(w, [pred("fiber", 1200, 700)])
    w.status(health=0.95)
    w.ctx.travel_moving = True
    w.tick()
    assert w.ctx.under_attack() and w.ctx.travel_guard.retreat_started is not None
    assert w.input.actions[0] == ("press", "s")
    assert ("press", "f") in w.input.actions
    clicks = [action for action in w.input.actions if action[0] == "click"]
    assert len(clicks) == 1 and clicks[0][1] < PLAYER[0]
    assert w.brain.counts["travel"] == 1 and not w.brain.counts["farm"]


def test_arrival_clears_goal_without_any_more_input_that_tick(monkeypatch, caplog):
    calls = fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK), NavigationStep("arrived")])
    w = travel_world()
    see(w, [pred("fiber", 1200, 700)])
    w.tick()
    assert ("click", *TRAVEL_CLICK) in w.input.actions and w.ctx.travel_moving
    w.input.actions.clear()
    w.clock.sleep(0.3)
    see(w, [pred("fiber", 1200, 700)])
    with caplog.at_level("INFO"):
        w.tick()
    assert w.config.behavior.travel_goal == ""
    assert not w.input.actions and not w.ctx.travel_moving
    assert "travel arrived" in caplog.text and len(calls) == 3


def test_uncalibrated_travel_sends_no_click_and_does_not_fall_through_to_farm():
    w = travel_world(calibrated=False)
    see(w, [pred("fiber", 1200, 700)])
    w.tick()
    assert not w.input.actions
    assert w.brain.counts["travel"] == 1 and not w.brain.counts["farm"]
    travel = next(b for b in w.brain.behaviors if b.name == "travel")
    assert travel.last_state == "uncalibrated"


def test_same_status_frame_is_not_used_for_duplicate_travel_clicks(monkeypatch):
    calls = fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK)])
    w = travel_world()
    w.clock.on_sleep = None
    w.tick()
    w.tick()
    assert len(calls) == 2
    assert w.input.actions.count(("click", *TRAVEL_CLICK)) == 1


def test_lost_fix_stops_previous_travel_motion(monkeypatch):
    fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK), NavigationStep("lost")])
    w = travel_world()
    w.tick()
    w.input.actions.clear()
    see(w)
    w.tick()
    assert w.input.actions == [("press", "s")]
    assert not w.ctx.travel_moving


def test_critical_health_retreats_before_heal_then_quits_at_deadline(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(health=0.249, low=True)
    w.tick()
    assert not w.brain.quit.is_set() and w.ctx.travel_guard.critical_since is not None
    assert ("press", "f") in w.input.actions
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)
    w.clock.sleep(8.1)
    with pytest.raises(SafetyStop):
        w.ctx.check_safety()
    assert w.brain.quit.is_set()


def test_low_hp_interrupts_fight_wait_then_dispatches_critical_retreat(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(health=0.8)
    see(w, [pred("mob", PLAYER[0]+60, PLAYER[1])])
    w.clock.at(0.2, lambda: w.status(health=0.24, low=True))
    w.tick()
    assert not w.brain.quit.is_set() and w.ctx.travel_guard.critical_since is not None
    assert w.input.actions == [("press", "r")]  # aborted opener wait, no attack click afterward
    assert w.ctx.active_behavior == ""
    w.input.actions.clear()
    w.tick()
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)
    assert not w.brain.quit.is_set()


def test_hp_published_during_path_planning_retreats_instead_of_dispatching_forward(monkeypatch):
    w = travel_world()
    def suddenly_low():
        w.status(health=0.2, low=True)
        return NavigationStep("move", click=TRAVEL_CLICK)
    fake_navigation(monkeypatch, [suddenly_low])
    seed_route(w)
    w.tick()
    assert not w.brain.quit.is_set() and w.ctx.travel_guard.critical_since is not None
    assert ("click", *TRAVEL_CLICK) not in w.input.actions
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)


def test_critical_retreat_cancels_forward_walk_and_does_not_apply_without_travel():
    w = travel_world()
    w.ctx.travel_moving = True
    w.status(health=0.24)
    w.ctx.check_safety()
    assert w.input.actions == [("press", "s")] and not w.brain.quit.is_set()
    assert w.ctx.travel_guard.critical_since is not None
    w = World()
    w.see([])
    w.status(health=0.24)
    w.ctx.check_safety()
    assert not w.brain.quit.is_set()
    guard = TravelGuard(Config())
    assert not guard.abort(reading(10, 0.25), 10)


def test_zero_hp_preserves_death_detection_and_critical_deadline():
    w = travel_world()
    w.status(health=0)
    w.tick()
    assert w.ctx.death_guard.dead and not w.input.actions
    assert not w.brain.quit.is_set()
    w.clock.sleep(8.1)
    w.tick()
    assert w.brain.quit.is_set() and not w.input.actions


@pytest.mark.parametrize("stale", [False, True])
def test_missing_numeric_or_stale_health_stops_travel_even_with_model_fallback(stale):
    w = travel_world()
    w.config.health.model_fallback = True
    w.ctx.travel_moving = True
    if stale:
        w.clock.on_sleep = None
        w.clock.sleep(3)
    else:
        w.status(health=None)
    w.tick()
    assert w.input.actions == [("press", "s")]
    assert not w.ctx.travel_moving and not w.brain.quit.is_set()


def test_damage_during_navigation_planning_yields_to_shield_before_movement(monkeypatch):
    w = travel_world()
    see(w, [pred("mob", 1900, 560)])
    w.clock.on_sleep = lambda: see(w, [pred("mob", 1900, 560)])
    def hit_while_planning():
        w.clock.sleep(0.1)
        w.status(health=0.95)
        return NavigationStep("move", click=TRAVEL_CLICK)
    fake_navigation(monkeypatch, [hit_while_planning])
    w.tick()
    assert w.brain.counts["shield"] == 1
    assert not any(action[0] == "click" for action in w.input.actions)


def test_enabled_travel_rejects_priority_ahead_of_combat():
    cfg = Config()
    cfg.behavior.travel_goal = "portal:NE"
    cfg.behavior.priority.remove("travel")
    cfg.behavior.priority.insert(0, "travel")
    with pytest.raises(ValueError, match="travel requires"):
        World(cfg)


def start_unseen_attack(w):
    w.ctx.check_safety()
    w.clock.sleep(0.2)
    w.status(health=0.95)
    see(w)
    w.tick()


def test_retreat_aims_boots_back_along_visited_route_and_does_not_recast_stale_ready(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    travel = seed_route(w)
    start_unseen_attack(w)
    aim = next(action for action in w.input.actions if action[0] == "move")
    click = next(action for action in w.input.actions if action[0] == "click")
    assert aim[1:] == click[1:] and aim[1] < PLAYER[0]
    for _ in range(3):
        w.clock.sleep(0.2)
        see(w)
        w.tick()
    assert w.input.actions.count(("press", "f")) == 1
    assert travel.retreat_index == 2


def test_retreat_waits_for_four_seconds_without_further_hp_drops_then_resumes(monkeypatch):
    fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK)])
    w = travel_world()
    travel = seed_route(w)
    start_unseen_attack(w)
    travel.task.position = Location(225, 145, None, 0.9)  # two old points reached by the dash
    w.clock.sleep(2)
    w.status(health=0.94)
    see(w)
    w.tick()
    w.clock.sleep(2.1)
    see(w)
    w.tick()
    assert w.ctx.travel_guard.retreat_started is not None  # only ~2s since the last hit
    w.clock.sleep(2)
    see(w)
    w.tick()
    assert w.ctx.travel_guard.retreat_started is None
    assert travel.trail == [Point(220, 145), Point(225, 145)]  # don't later backtrack toward the attacker
    assert travel.task.route == []
    w.input.actions.clear()
    see(w)
    w.tick()
    assert ("click", *TRAVEL_CLICK) in w.input.actions


def test_second_attack_near_same_spot_blocks_grid_and_replans_around_it(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    travel = seed_route(w)
    start_unseen_attack(w)
    travel.task.position = Location(225, 145, None, 0.9)
    w.clock.sleep(1)
    see(w)
    w.tick()
    w.clock.sleep(4.1)
    see(w)
    w.tick()
    assert w.ctx.travel_guard.retreat_started is None
    travel.task.position = Location(236, 145, None, 0.9)
    w.clock.sleep(0.2)
    w.status(health=0.9)
    see(w)
    w.tick()
    assert len(travel.attack_spots) == 2
    assert not travel.task.grid.mask[145, 236]
    route = travel.task.grid.plan((220, 145), (250, 145))
    assert any(abs(p.y-145) > 3 for p in route)
    assert all(travel.task.grid.line_clear(a, b) for a, b in zip(route, route[1:]))
    assert w.input.actions.count(("press", "f")) == 1  # no stale-up double dash within a minute
    # Calm HP alone must not resume from inside the hazard's inflated blocked cells.
    travel.task.position = Location(233, 145, None, 0.9)
    w.clock.sleep(4.1)
    see(w)
    w.tick()
    assert w.ctx.travel_guard.retreat_started is not None
    travel.task.position = Location(225, 145, None, 0.9)
    w.clock.sleep(0.2)
    see(w)
    w.tick()
    assert w.ctx.travel_guard.retreat_started is None


def test_critical_retreat_continues_despite_recovery_then_quits_at_eight_seconds(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(health=0.24, low=True)
    w.tick()
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)
    w.clock.sleep(4.1)
    w.status(health=1, low=False)
    see(w)
    w.tick()
    assert not w.brain.quit.is_set() and w.ctx.travel_guard.critical_since is not None
    w.clock.sleep(4)
    w.tick()
    assert w.brain.quit.is_set() and w.input.actions[-1] == ("press", "s")


def test_positive_minimum_hp_does_not_freeze_critical_escape(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(health=0.03, low=True)
    w.tick()
    w.clock.sleep(4)
    see(w)
    w.input.actions.clear()
    w.tick()
    assert not w.ctx.death_guard.dead and not w.brain.quit.is_set()
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)


def test_unseen_attacker_retreat_does_not_require_any_scene_detections(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.ctx.check_safety()
    w.clock.sleep(0.2)
    w.workers.scenes.publish(build_scene(w.clock.now(), (2240, 1400), [], [], w.config))
    w.status(health=0.95)
    w.tick()
    assert w.ctx.scene().is_empty()
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)
    assert ("press", "esc") not in w.input.actions


def test_missing_perception_cancels_retreat_motion_but_deadline_still_quits(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    w.status(health=0.24)
    w.tick()
    w.input.actions.clear()
    w.status(frame=None)
    see(w)
    w.tick()
    assert w.input.actions == [("press", "s")]
    assert not w.brain.quit.is_set()
    w.clock.sleep(8.1)
    w.status(health=None)
    w.tick()
    assert w.brain.quit.is_set()


def test_no_retreat_clicks_without_calibration_or_without_recorded_route(monkeypatch):
    fake_navigation(monkeypatch, [])
    for calibrated in (False, True):
        w = travel_world(calibrated=calibrated)
        w.status(health=0.24)
        w.tick()
        assert not any(a[0] in ("click", "move") for a in w.input.actions)
        assert w.ctx.travel_guard.critical_since is not None and not w.brain.quit.is_set()


def test_forward_travel_records_actual_positions_for_later_retreat(monkeypatch):
    fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK),
                                NavigationStep("move", click=TRAVEL_CLICK)])
    w = travel_world()
    w.tick()
    travel = next(b for b in w.brain.behaviors if b.name == "travel")
    assert travel.trail == [Point(235, 145)]
    travel.task.position = Location(237, 145, None, 0.9)
    see(w)
    w.tick()
    assert travel.trail == [Point(235, 145), Point(237, 145)]


def test_stale_enemy_detections_do_not_block_escape_from_fresh_minimap(monkeypatch):
    fake_navigation(monkeypatch, [])
    w = travel_world()
    seed_route(w)
    see(w, [pred("mob", 1900, 560)])
    w.ctx.check_safety()
    w.clock.sleep(3)
    w.ctx.check_safety()  # healthy baseline after old samples expired
    w.clock.sleep(0.2)
    w.status(health=0.95)
    w.tick()
    assert w.ctx.retreat_required()
    assert any(a[0] == "click" and a[1] < PLAYER[0] for a in w.input.actions)


def test_travel_gives_up_after_no_move_so_the_bot_is_not_held_still(monkeypatch, caplog):
    stuck = NavigationStep("no_route", message="player outside clear mask")
    fake_navigation(monkeypatch, [stuck] * 50)
    w = travel_world()
    w.config.behavior.travel_give_up_seconds = 5.0
    for _ in range(40):
        see(w)
        w.tick()
        if not w.config.behavior.travel_goal:
            break
        w.clock.sleep(0.5)
    assert w.config.behavior.travel_goal == ""
    assert not any(a[0] == "click" for a in w.input.actions)
    assert "travel gave up" in caplog.text


def test_forward_travel_reclicks_about_once_a_second_without_the_stop_key(monkeypatch):
    fake_navigation(monkeypatch, [NavigationStep("move", click=TRAVEL_CLICK)] * 6)
    w = travel_world()
    for _ in range(6):
        see(w)
        w.tick()
        w.clock.sleep(0.3)
    clicks = [a for a in w.input.actions if a[0] == "click"]
    assert 2 <= len(clicks) <= 3
    assert ("press", "s") not in w.input.actions
