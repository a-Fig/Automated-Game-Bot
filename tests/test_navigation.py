import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from albot.config import Config, NavigationConfig
from albot.geometry import Point
from albot.navigation import (Location, MapLocator, NavigateTo, WalkGrid, Zone,
                              build_walkability, locate, screen_click, solve_calibration)
from albot import navigation_cli


FIXTURES = Path(__file__).parent / "fixtures" / "navigation"


def frame_at(x=235, y=145):
    frame = np.zeros((1400, 2240, 3), np.uint8)
    frame[1040:, 1840:] = (90, 135, 190)
    arrow(frame, x, y)
    return frame


def arrow(frame, x, y):
    crop = frame[1040:, 1840:]
    cv2.fillPoly(crop, [np.array([(x-6, y-5), (x+6, y), (x-6, y+5)])], (10, 10, 10))
    cv2.fillPoly(crop, [np.array([(x-3, y-2), (x+3, y), (x-3, y+2)])], (255, 179, 97))


def recorded_frame(name):
    frame = np.zeros((1400, 2240, 3), np.uint8)
    frame[1040:, 1840:] = cv2.imread(str(FIXTURES / name))
    return frame


def open_zone(cfg):
    metadata = {"terrain_polygon": [[0, 0], [399, 0], [399, 359], [0, 359]],
                "portals": {"NE": {"approach": [290, 145]}}}
    return Zone("fake", Path("."), metadata, WalkGrid(np.ones((360, 400), bool),
                                                      cfg.clearance if cfg.clearance is not None else 2.0))


@pytest.mark.parametrize("name,expected", [
    ("frame_minimap_1.png", (235, 145)), ("s1_00039.png", (235, 145)),
    ("s1_00079.png", (236, 145)), ("s1_00179.png", (225, 125)),
    ("s1_00316.png", (217, 123)),
])
def test_recorded_arrow_not_castle_including_darkened_frames(name, expected):
    frame = recorded_frame(name)
    result = locate(frame)
    assert result is not None and result.point.dist(Point(*expected)) < 2
    assert result.confidence >= 0.7
    assert abs(result.x-157) > 20  # the castle's small blue patch


def test_recorded_tab_overlay_has_no_trustworthy_fix():
    assert locate(recorded_frame("tabmap_open_2.png")) is None


def test_channels_are_explicit_and_heading_is_optional():
    frame = frame_at()
    result = locate(frame)
    assert result is not None and result.point.dist(Point(235, 145)) < 2
    assert result.heading is not None and abs(result.heading) < 0.2
    assert locate(Image.fromarray(frame[:, :, ::-1])) == result
    assert locate(frame[:, :, ::-1]) is None  # numpy RGB violates documented BGR contract


def test_castle_rocks_ui_and_overlays_do_not_become_the_arrow():
    frame = frame_at()
    crop = frame[1040:, 1840:]
    crop[138:153, 225:245] = (90, 135, 190)  # remove arrow
    cv2.rectangle(crop, (145, 150), (175, 180), (255, 179, 97), -1)  # large castle
    cv2.rectangle(crop, (190, 180), (197, 187), (220, 200, 180), -1)  # grey-blue rock
    arrow(frame, 60, 320)  # blue zone banner outside map diamond
    assert locate(frame) is None
    frame[1040:, 1840:] = 0
    assert locate(frame) is None
    assert locate(np.zeros((1080, 1920, 3), np.uint8)) is None


def test_ambiguous_arrows_fail_closed_and_continuity_selects_nearby():
    frame = frame_at()
    arrow(frame, 270, 160)
    assert locate(frame) is None
    previous = Location(234, 145, None, 0.9)
    result = locate(frame, previous=previous)
    assert result is not None and result.point.dist(previous.point) < 3
    assert locate(frame, previous=Location(120, 160, None, 0.9)) is None


def test_tracker_rejects_jump_and_reacquires_after_occlusion():
    tracker = MapLocator(NavigationConfig())
    assert tracker.locate(frame_at(), 0) is not None
    assert tracker.locate(frame_at(237, 145), 0.2) is not None
    assert tracker.locate(frame_at(270, 145), 0.4) is None
    assert tracker.locate(frame_at(270, 145), 3) is not None
    with pytest.raises(ValueError, match="timestamps"):
        tracker.locate(frame_at(), 2)


def test_colour_mask_tan_and_road_open_brown_and_blue_blocked():
    image = np.full((50, 50, 3), (90, 135, 190), np.uint8)
    image[10:20, 10:20] = (65, 90, 125)  # dark brown
    image[30:40, 10:20] = (150, 140, 110)  # blue-grey
    image[10:20, 30:40] = (35, 135, 225)  # orange road
    polygon = [(2, 2), (47, 2), (47, 47), (2, 47)]
    mask = build_walkability(image, polygon, size=(50, 50))
    assert mask[25, 25] and mask[15, 35]
    assert not mask[15, 15] and not mask[35, 15] and not mask[0, 0]
    large = cv2.resize(image, (100, 100), interpolation=cv2.INTER_NEAREST)
    transformed = build_walkability(large, polygon, (50, 50), [[0.5, 0, 0], [0, 0.5, 0]])
    assert np.array_equal(mask, transformed)


def test_mask_png_roundtrip_and_three_state_override(tmp_path):
    mask = np.ones((20, 20), bool)
    mask[10, 10] = False
    grid = WalkGrid(mask, clearance=0)
    path, override = tmp_path / "mask.png", tmp_path / "override.png"
    grid.save(path)
    assert np.array_equal(WalkGrid.load(path, clearance=0).mask, mask)
    paint = np.full(mask.shape, 128, np.uint8)
    paint[10, 10], paint[5, 5] = 255, 0
    cv2.imwrite(str(override), paint)
    loaded = WalkGrid.load(path, override, clearance=0)
    assert loaded.mask[10, 10] and not loaded.mask[5, 5] and loaded.mask[15, 15]
    paint[2, 2] = 127
    cv2.imwrite(str(override), paint)
    with pytest.raises(ValueError, match="override"):
        WalkGrid.load(path, override)


def test_astar_clearance_and_simplification_go_around_walls():
    mask = np.ones((100, 100), bool)
    mask[:75, 45:55] = False
    grid = WalkGrid(mask, clearance=2)
    route = grid.plan((20, 20), (80, 20))
    assert route[0] == Point(20, 20) and route[-1] == Point(80, 20)
    assert 3 <= len(route) <= 6
    assert max(p.y for p in route) >= 77
    assert all(grid.line_clear(a, b) for a, b in zip(route, route[1:]))
    assert not grid.is_safe((43, 30))  # two-pixel clearance from wall


def test_disconnected_routes_and_diagonal_corner_cuts_are_rejected():
    mask = np.ones((30, 30), bool)
    mask[:, 15] = False
    grid = WalkGrid(mask, clearance=0)
    with pytest.raises(ValueError, match="no route"):
        grid.plan((5, 5), (25, 5))
    diagonal = WalkGrid(np.eye(3, dtype=bool), clearance=0, snap_radius=0)
    assert not diagonal.line_clear((0, 0), (2, 2))
    with pytest.raises(ValueError, match="no route"):
        diagonal.plan((0, 0), (2, 2))


def test_snap_is_bounded_and_goal_inputs_validated():
    mask = np.ones((30, 30), bool)
    mask[10:15, 10:15] = False
    grid = WalkGrid(mask, clearance=0, snap_radius=3)
    route = grid.plan((5, 5), (12, 12))
    assert route[-1].dist(Point(12, 12)) <= 3 and grid.is_safe(route[-1])
    with pytest.raises(ValueError, match="outside"):
        grid.plan((-1, 5), (20, 20))
    with pytest.raises(ValueError, match="finite"):
        grid.plan((math.nan, 5), (20, 20))
    zone = open_zone(NavigationConfig())
    assert zone.goal("portal:NE") == Point(290, 145)
    assert zone.goal("250,180") == Point(250, 180)
    with pytest.raises(ValueError, match="unknown portal"):
        zone.goal("portal:INVALID")
    with pytest.raises(ValueError):
        Zone.load("../other")


def test_bundled_zone_routes_recorded_position_to_ne_portal():
    zone = Zone.load("fractured_ground")
    result = locate(recorded_frame("frame_minimap_1.png"))
    route = zone.grid.plan(result.point, zone.goal("portal:NE"))
    assert route[-1].dist(zone.goal("portal:NE")) <= 3
    assert all(zone.grid.line_clear(a, b) for a, b in zip(route, route[1:]))
    assert not zone.metadata["walkability_verified"]
    image = cv2.imread(str(zone.folder / "map.png"))
    with pytest.raises(ValueError, match="full-resolution Tab"):
        zone.build_mask(image, kind="map")  # don't silently restore the obsolete colour mask


@pytest.mark.parametrize("delta", [(100, 0), (-100, 0), (0, 100), (0, -100), (0.01, 0.01)])
def test_click_range_and_hud_bounds_preserve_direction(delta):
    cfg = NavigationConfig()
    cfg.minimap_to_screen = [[40, 0], [0, 40]]
    click = screen_click(delta, cfg)
    player = Point(*cfg.player_screen)
    assert cfg.click_min <= click.dist(player) <= cfg.click_max+1e-8
    x0, y0, x1, y1 = cfg.safe_screen_rect
    assert x0 <= click.x <= x1 and y0 <= click.y <= y1
    assert abs((click.x-player.x)*delta[1]-(click.y-player.y)*delta[0]) < 1e-5


def test_matrix_rotation_reflection_and_invalid_matrix():
    cfg = NavigationConfig()
    cfg.minimap_to_screen = [[0, -30], [40, 0]]
    assert screen_click((10, 0), cfg) == Point(1125, 960)
    assert screen_click((0, 10), cfg) == Point(825, 560)
    assert screen_click((0, 0), cfg) is None
    cfg.minimap_to_screen = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="matrix"):
        screen_click((1, 1), cfg)
    cfg.minimap_to_screen = [[1, 0], [0, 1]]
    cfg.safe_screen_rect = (1100, 540, 1150, 580)
    assert screen_click((1, 0), cfg) is None


def task(cfg=None, goal="portal:NE"):
    cfg = cfg or NavigationConfig()
    cfg.calibrated = True
    cfg.minimap_to_screen = [[40, 0], [0, 40]]
    return NavigateTo(goal, open_zone(cfg), cfg)


def test_follow_moves_advances_and_arrives_without_dispatch():
    nav = task()
    step = nav.tick(frame_at(), 0)
    assert step.status == "move" and step.click is not None
    assert step.click.x > nav.config.player_screen[0]
    step = nav.tick(frame_at(260, 145), 3)
    assert step.status == "move"
    step = nav.tick(frame_at(289, 145), 6)
    assert step.status == "arrived" and step.click is None


def test_uncalibrated_task_and_lost_fix_never_propose_input():
    cfg = NavigationConfig()
    nav = NavigateTo("portal:NE", open_zone(cfg), cfg)
    assert nav.tick(frame_at(), 0).status == "uncalibrated"
    step = nav.tick(np.zeros((1400, 2240, 3), np.uint8), 1)
    assert step.status == "lost" and step.click is None


def test_stuck_sidestep_once_then_block_and_replan():
    nav = task()
    for t in [0, 1, 2, 3]:
        assert nav.tick(frame_at(), t).status == "move"
    step = nav.tick(frame_at(), 4.1)
    assert step.status == "sidestep" and step.click is not None
    old_point = nav.stuck_point
    assert abs(step.click.x-nav.config.player_screen[0]) < 30
    assert nav.tick(frame_at(), 4.5).status == "waiting"
    nav.tick(frame_at(), 5.2)
    assert nav.replans == 1
    assert not nav.grid.mask[round(old_point.y), round(old_point.x)]
    assert nav.zone.grid.mask[round(old_point.y), round(old_point.x)]  # task-local block


def test_stuck_recovery_is_bounded_and_lost_time_does_not_count():
    cfg = NavigationConfig()
    cfg.max_replans = 0
    nav = task(cfg)
    for t in range(4):
        nav.tick(frame_at(), t)
    step = nav.tick(frame_at(), 4.1)
    assert step.status == "stalled" and step.click is None
    assert nav.tick(frame_at(), 5).status == "stalled"
    nav = task()
    nav.tick(frame_at(), 0)
    nav.tick(np.zeros((1400, 2240, 3), np.uint8), 4)
    assert nav.tick(frame_at(), 6).status == "move"


def test_player_in_blocked_mask_does_not_click_through_wall():
    nav = task()
    nav.grid.block((235, 145), 6)
    step = nav.tick(frame_at(), 0)
    assert step.status == "no_route" and step.click is None


def test_calibration_solves_full_matrix_and_speed():
    matrix = np.array([[0, -40], [-50, 0]], float)
    samples = [{"map_delta": d, "screen_delta": (matrix @ d).tolist(), "duration": 2}
               for d in [(7, 0), (0, 8), (5, 5)]]
    result = solve_calibration(samples)
    assert np.allclose(result["minimap_to_screen"], matrix)
    assert result["rms_screen_error"] < 1e-6 and result["calibrated"]
    assert result["map_speed_px_per_s"] > 0


@pytest.mark.parametrize("samples", [[], [{"map_delta": [0, 0], "screen_delta": [350, 0]}]*2,
    [{"map_delta": [1, 0], "screen_delta": [350, 0]}, {"map_delta": [2, 0], "screen_delta": [350, 0]}],
    [{"map_delta": [5, 0], "screen_delta": [350, 0]}, {"map_delta": [0, 5], "screen_delta": [0, 350]},
     {"map_delta": [5, 5], "screen_delta": [-350, -350]}]])
def test_bad_calibration_samples_rejected(samples):
    with pytest.raises(ValueError):
        solve_calibration(samples)


def test_saved_image_cli_never_uses_live_capture_or_input(tmp_path, monkeypatch, capsys):
    from albot import app
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), recorded_frame("frame_minimap_1.png"))
    monkeypatch.setattr(navigation_cli, "live_capture", lambda cfg: pytest.fail("no live capture"))
    monkeypatch.setattr(app, "setup_logging", lambda cfg: None)
    assert app.main(["nav", "--to", "portal:NE", "--dry-run", "--image", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"waypoints"' in output and '"proposed_click"' in output and "uncalibrated" in output
    assert app.main(["nav", "--to", "portal:NE", "--image", str(path)]) == 1


def test_calibration_cli_requires_execute_and_solve_is_offline(tmp_path, monkeypatch):
    from albot import app
    monkeypatch.setattr(app, "setup_logging", lambda cfg: None)
    monkeypatch.setattr(navigation_cli, "_probe", lambda *args: pytest.fail("must not probe"))
    output = tmp_path / "calibration.json"
    assert app.main(["nav-calibrate", "--probe", "350,0", "--out", str(output)]) == 1
    paths = []
    cfg = Config().navigation
    for i, (mapped, screen) in enumerate([([7, 0], [350, 0]), ([0, 7], [0, 350])]):
        sample = {"map_delta": mapped, "screen_delta": screen, "duration": 2,
                  "window_size": list(cfg.window_size), "minimap_rect": list(cfg.minimap_rect), "zone": cfg.zone}
        path = tmp_path / f"probe{i}.json"
        path.write_text(json.dumps(sample))
        paths.append(str(path))
    assert app.main(["nav-calibrate", "--samples", *paths, "--out", str(output)]) == 0
    result = json.loads(output.read_text())
    assert np.allclose(result["minimap_to_screen"], [[50, 0], [0, 50]])
    navigation_cli.load_calibration(cfg, output)
    assert cfg.calibrated
    cfg.window_size = (100, 100)
    with pytest.raises(ValueError, match="does not match"):
        navigation_cli.load_calibration(cfg, output)


def test_live_probe_routine_with_fake_capture_clock_and_input_only(tmp_path, monkeypatch):
    from albot import actuator, vision
    from albot.window import FixedWindow
    now, clicks = [0.0], []
    fake_time = SimpleNamespace(monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setattr(navigation_cli, "time", fake_time)
    monkeypatch.setattr(vision, "read_health", lambda frame, cfg: 1.0)
    monkeypatch.setattr(actuator, "Input", lambda *args: SimpleNamespace(click=clicks.append))
    def grab():
        return frame_at(235+min(6, max(0, int((now[0]-3)*4))), 145)
    monkeypatch.setattr(navigation_cli, "live_capture", lambda cfg: (FixedWindow(2240, 1400), grab))
    path = tmp_path / "probe.json"
    navigation_cli._probe(Config(), (350, 0), path)
    sample = json.loads(path.read_text())
    assert len(clicks) == 1 and clicks[0] == Point(1475, 560)
    assert sample["map_delta"][0] == pytest.approx(6)
    assert sample["screen_delta"] == [350, 0] and sample["duration"] > 0


def test_player_just_off_clear_mask_walks_back_to_nearest_clear_cell():
    nav = task()
    nav.grid.block((235, 145), 1)  # off by a pixel or two, as seen live
    assert not nav.grid.is_safe((235, 145))
    step = nav.tick(frame_at(), 0)
    assert step.status == "move" and step.click is not None
    assert "nearest clear cell" in step.message and not nav.route


def test_stuck_block_never_covers_the_players_own_cell():
    cfg = NavigationConfig()
    cfg.lookahead = 1.0  # the next waypoint is close, as it was live
    nav = task(cfg)
    for t in range(5):
        nav.tick(frame_at(), t)
    assert nav.side_step_at is not None
    step = nav.tick(frame_at(), 6)  # sidestep wait over: block the stuck point, replan
    assert nav.grid.is_safe(step.location.point) and step.status == "move"


def test_stuck_block_that_cuts_the_only_route_is_undone():
    nav = task()
    for t in range(5):
        nav.tick(frame_at(), t)
    assert nav.side_step_at is not None
    real_block = nav.grid.block
    nav.grid.block = lambda point, radius: real_block(point, 60)  # cuts everything nearby
    step = nav.tick(frame_at(), 6)
    assert step.status == "move" and step.click is not None
    assert nav.grid.is_safe(step.location.point)
