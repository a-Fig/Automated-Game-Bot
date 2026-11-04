import pytest

from albot.behaviors import Farm, Heal, Pickup, walk_detour, Plan
from albot.geometry import Point
from albot.state import Detection
from fakes import PLAYER, World, pred


def test_failed_gather_skips_moved_node_but_allows_other_resources_and_expires():
    w = World()
    w.see([pred("fiber", PLAYER[0] + 150, PLAYER[1])])
    w.tick()
    assert len(w.ctx.resources.failed) == 1
    assert not w.workers.want_loading_bar.is_set()
    # Approach moved the camera by 150 pixels: that world node now sits at player.
    scene = w.see([pred("fiber", *PLAYER), pred("hide", PLAYER[0], PLAYER[1] + 100)])
    assert w.ctx.resources.blocked(scene, scene.farmables[0])
    plan = Farm().plan(w.ctx, scene, w.ctx.status())
    assert plan.target.class_name == "hide"
    w.clock.sleep(w.config.behavior.resource_blacklist_seconds + 1)
    scene = w.see([pred("fiber", *PLAYER)])
    assert Farm().plan(w.ctx, scene, w.ctx.status()) is not None


def test_intended_movement_maps_shifted_frames_to_same_world_spot():
    w = World()
    scene = w.see([])
    target = Detection("fiber", 0.9, Point(PLAYER[0] + 400, PLAYER[1] + 100))
    spot = w.ctx.resources.world_point(scene, target.center)
    w.ctx.resources.failure("fiber", spot)
    w.ctx.walk_to(Point(PLAYER[0] + 250, PLAYER[1]))
    w.clock.sleep(0.4)  # 100 pixels at the configured 250 px/s
    shifted = w.see([pred("fiber", target.center.x - 100, target.center.y)])
    assert w.ctx.resources.blocked(shifted, shifted.farmables[0])
    # A frame from before the click must use the old camera origin, despite inference lag.
    assert w.ctx.resources.world_point(scene, target.center) == spot
    w.clock.sleep(10)
    assert w.ctx.resources.position(w.clock.now()) == Point(250, 0)  # capped at destination


def test_new_walk_replaces_previous_intent_and_stopping_freezes_origin():
    w = World()
    scene = w.see([])
    w.ctx.walk_to(Point(PLAYER[0] + 500, PLAYER[1]))
    w.clock.sleep(0.4)
    w.ctx.walk_to(Point(PLAYER[0], PLAYER[1] + 500))
    w.clock.sleep(0.4)
    assert w.ctx.resources.position(w.clock.now()) == pytest.approx((100, 100))
    w.ctx.resources.move(scene, None)
    w.clock.sleep(10)
    assert w.ctx.resources.position(w.clock.now()) == pytest.approx((100, 100))


def test_loading_bar_success_or_danger_does_not_blacklist_a_resource():
    w = World()
    w.see([pred("fiber", PLAYER[0] + 150, PLAYER[1])])
    w.clock.at(0.2, lambda: w.status(loading=True))
    w.clock.at(1, lambda: w.status(loading=False))
    w.tick()
    assert not w.ctx.resources.failed
    assert w.brain.counts["farm"] == 1
    w = World()
    w.see([pred("fiber", PLAYER[0] + 150, PLAYER[1])])
    w.clock.at(0.2, lambda: w.see([pred("red", *PLAYER)]))
    w.tick()
    assert not w.ctx.resources.failed


def test_gather_timeout_blacklists_and_death_cleans_up_loading_request():
    w = World()
    scene = w.see([pred("fiber", PLAYER[0] + 150, PLAYER[1])])
    w.config.behavior.farm_max_seconds = 0.5
    w.clock.at(0.2, lambda: w.status(loading=True))
    assert not Farm().run(w.ctx, Farm().plan(w.ctx, scene, w.ctx.status()))
    assert len(w.ctx.resources.failed) == 1
    assert not w.workers.want_loading_bar.is_set()
    w = World()
    w.see([pred("fiber", PLAYER[0] + 150, PLAYER[1])])
    w.clock.at(0.2, lambda: w.status(health=0))
    w.tick()
    assert w.ctx.death_guard.dead and not w.ctx.resources.failed
    assert not w.workers.want_loading_bar.is_set()


def test_pickup_and_detour_also_update_camera_estimate():
    w = World()
    scene = w.see([pred("silver", PLAYER[0] + 250, PLAYER[1])])
    Pickup().run(w.ctx, Pickup().plan(w.ctx, scene, w.ctx.status()))
    assert w.ctx.resources.position(w.clock.now()) == Point(250, 0)
    scene = w.see([pred("fiber", PLAYER[0], PLAYER[1] + 300)])
    walk_detour(w.ctx, Plan("farm", "detour", target=scene.farmables[0],
                            waypoint=Point(PLAYER[0], PLAYER[1] + 100)))
    assert w.ctx.resources.position(w.clock.now()) == Point(250, 100)


def test_dash_clears_unmeasured_resource_coordinates():
    w = World()
    scene = w.see([])
    w.ctx.resources.failure("fiber", Point(100, 100))
    w.status(low=True, health=0.3)
    w.clock.at(0.2, lambda: w.status(low=False, health=1))
    Heal().run(w.ctx, Heal().plan(w.ctx, scene, w.ctx.status()))
    assert not w.ctx.resources.failed
