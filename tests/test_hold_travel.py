from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from albot.actuator import DryRunInput
from albot.config import Config
from albot.geometry import Point
from albot.hold_travel import (HoldConfig, HoldSteer, map_to_tab, route_ahead, tab_arrow, tab_to_map)
from albot.hold_travel_cli import run_hold

TRANSFORM = [[0.2225, 0.0, -38.48], [0.0, 0.2225, -14.25]]
MATRIX = [[373.34, 274.96], [-59.97, 341.67]]
CITY = (887.6, 784.2)


def tab_frame(*blobs):
    frame = np.full((1400, 2240, 3), (60, 110, 160), np.uint8)  # brown map, BGR
    for x, y in blobs:
        cv2.circle(frame, (round(x), round(y)), 6, (230, 160, 40), -1)  # blue arrow, ~113 px
    return frame


def test_tab_arrow_skips_the_city_icon_and_needs_history_when_ambiguous():
    frame = tab_frame((1312, 681), CITY)
    assert tab_arrow(frame, static_icons=[CITY]).dist(Point(1312, 681)) < 1
    assert tab_arrow(frame) is None  # two blobs, no history
    assert tab_arrow(frame, previous=(1300, 690)).dist(Point(1312, 681)) < 1
    half = cv2.resize(frame, (1120, 700), interpolation=cv2.INTER_AREA)
    assert tab_arrow(half, static_icons=[CITY]).dist(Point(1312, 681)) < 2  # window coordinates


def test_tab_and_map_coordinates_round_trip():
    tab = map_to_tab(TRANSFORM, (253.0, 137.1))
    assert tab_to_map(TRANSFORM, tab).dist(Point(253.0, 137.1)) < 1e-9


def test_route_ahead_follows_corners_and_stops_at_the_end():
    route = [(0, 0), (10, 0), (10, 10)]
    assert route_ahead(route, (2, 1), 3) == pytest.approx((5, 0))
    assert route_ahead(route, (9, 0), 3) == pytest.approx((10, 2))
    assert route_ahead(route, (10, 9), 5) == pytest.approx((10, 10))


def steer_task(score=lambda p, d: 0):
    cfg = HoldConfig()
    return HoldSteer([(0, 0), (20, 0)], (20, 0), [[100, 0], [0, 100]], (1125, 560), cfg, score), cfg


def test_steer_points_the_cursor_along_the_route():
    steer, cfg = steer_task()
    step = steer.update((5, 0), 0)
    assert step.status == "steer"
    assert step.cursor == pytest.approx((1125+cfg.mouse_pixels, 560))
    assert steer.update((19, 0.5), 1).status == "arrived"


def test_stall_detours_to_the_road_side_then_the_other_then_gives_up():
    # Landing on road "below" (+y on screen) scores higher: the first detour turns there.
    steer, cfg = steer_task(lambda p, d: 2 if d[1] > 0 else 0)
    t, angles = 0.0, []
    while True:
        step = steer.update((5, 0), t)
        if step.status == "stalled":
            break
        if step.message.startswith("stall"):
            vector = np.asarray(step.cursor)-(1125, 560)
            angles.append(round(np.degrees(np.arctan2(vector[1], vector[0]))))
        t += 0.1
        assert t < 30
    assert angles == [60, -60, 100, -100]
    assert "after 4 detours" in step.message


def test_detours_reset_after_moving_away_from_the_stall():
    steer, cfg = steer_task()
    t = 0.0
    while not steer.update((5, 0), t).message.startswith("stall"):
        t += 0.1
    assert steer.detour_count == 1
    steer.update((8, 0), t+cfg.detour_seconds+0.1)
    assert steer.detour_count == 0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def hold_task(positions):
    cfg = Config()
    cfg.navigation.minimap_to_screen = MATRIX
    zone = SimpleNamespace(metadata={"tab_to_map": TRANSFORM, "tab_static_icons": [CITY]},
                           road=None)
    nav = SimpleNamespace(zone=zone, goal=Point(*positions[-1]),
                          grid=SimpleNamespace(scale=4, is_safe=lambda p: True))
    frames = iter([tab_frame(map_to_tab(TRANSFORM, p), CITY) for p in positions])
    return cfg, nav, frames


def test_run_hold_steers_to_arrival_and_always_releases():
    positions = [(250+i, 140-0.5*i) for i in range(12)]
    cfg, nav, frames = hold_task(positions)
    clock, inp = FakeClock(), DryRunInput()
    status, _ = run_hold(cfg, nav, positions, lambda: next(frames), inp, 60, clock=clock, sleep=clock.sleep)
    assert status == "arrived"
    kinds = [a[0] for a in inp.actions]
    assert kinds[0] == "hold" and kinds[1] == "press" and "steer" in kinds
    assert kinds[-2:] == ["release_move", "press"]  # button up, then the zone map closed


def test_run_hold_releases_after_an_error_and_on_low_health():
    positions = [(250+i, 140) for i in range(12)]
    cfg, nav, frames = hold_task(positions)
    clock, inp = FakeClock(), DryRunInput()

    def broken():
        raise RuntimeError("capture failed")
    with pytest.raises(RuntimeError):
        run_hold(cfg, nav, positions, broken, inp, 60, clock=clock, sleep=clock.sleep)
    assert ("release_move",) in inp.actions and inp.actions[-1] == ("press", "tab")

    inp = DryRunInput()
    status, message = run_hold(cfg, nav, positions, lambda: next(frames), inp, 60, health=lambda f: 0.4,
                               clock=clock, sleep=clock.sleep)
    assert status == "low_health" and ("release_move",) in inp.actions


def test_run_hold_treats_a_loading_screen_as_a_zone_change_and_leaves_the_map_alone():
    positions = [(250+i, 140) for i in range(12)]
    cfg, nav, frames = hold_task(positions)
    clock, inp = FakeClock(), DryRunInput()
    black = iter([next(frames), np.zeros((1400, 2240, 3), np.uint8)])
    status, _ = run_hold(cfg, nav, positions, lambda: next(black), inp, 60, clock=clock, sleep=clock.sleep)
    assert status == "zone_changed"
    assert inp.actions[-1] == ("release_move",)


def test_bouncing_between_two_spots_does_not_reset_the_detour_budget():
    """Live 2026-11-04: a ledge bounce 2.6 px apart reset detours forever."""
    steer, cfg = steer_task()
    t, spot, status = 0.0, 0, "steer"
    while status != "stalled":
        point = (5, 0) if spot == 0 else (5, 2.6)  # sideways, no closer to the goal
        step = steer.update(point, t)
        status = step.status
        if step.message.startswith("stall"):
            spot ^= 1
        t += 0.1
        assert t < 60
    assert steer.detour_count == cfg.detours


def test_trails_route_either_direction_and_persist(tmp_path):
    from albot.hold_travel import Trails
    path = tmp_path / "hold_trails.json"
    trails = Trails(path)
    walked = [(230+0.1*i, 130+0.05*i) for i in range(200)]  # (230,130) -> (249.9,139.95)
    assert trails.add(walked, when="test")
    assert not trails.add([(1, 1), (1.1, 1.1)])  # too short after thinning
    loaded = Trails(path)
    forward = loaded.route((231, 130.2), (245, 137.4))
    assert forward[0] == Point(231, 130.2) and forward[-1] == Point(245, 137.4)
    assert all(a.x <= b.x+1e-9 for a, b in zip(forward[1:-1], forward[2:-1]))
    backward = loaded.route((248, 139), (233, 131.5))
    assert backward is not None and backward[1].x > backward[-2].x
    assert loaded.route((200, 100), (245, 137.4)) is None  # start not near the trail


def test_route_toward_follows_a_trail_to_its_point_nearest_a_goal_past_its_end(tmp_path):
    from albot.hold_travel import Trails
    trails = Trails(tmp_path / "t.json")
    trails.add([(230+0.1*i, 130) for i in range(100)])  # ends at (239.9,130)
    partial = trails.route_toward((231, 130.5), (250, 125))
    assert partial[0] == Point(231, 130.5) and partial[-1].dist(Point(239.9, 130)) < 0.5
    assert trails.route((231, 130.5), (250, 125)) is None
    assert trails.route_toward((231, 130.5), (229, 140)) is None  # no progress along the trail
