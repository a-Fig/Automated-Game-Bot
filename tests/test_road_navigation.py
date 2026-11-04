import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from albot.config import NavigationConfig
from albot.geometry import Point
from albot.hud import hud_reason
from albot.config import Config
from albot.navigation import Location, NavigateTo, ScaledWalkGrid, WalkGrid, Zone, screen_click
from albot.navigation_evidence import LearnedEvidence
from test_navigation_learning import saved_zone


MATRIX = [[373.34, 274.96], [-59.97, 341.67]]
FIXTURE = Path(__file__).parent / "fixtures/navigation/run15_edge19.json"


def test_run15_saved_cut_stays_active_until_explicit_retraction(tmp_path):
    cfg = NavigationConfig(calibrated=True, minimap_to_screen=MATRIX)
    zone = Zone.load("fractured_ground", cfg, learned=False)
    path = tmp_path / "learned.json"
    shutil.copyfile(FIXTURE, path)
    zone.evidence = LearnedEvidence(path, zone.name, (400, 360), 4)
    before = path.read_bytes()
    nav = NavigateTo("portal:NE", zone, cfg)
    assert nav.evidence.edges[0]["source_index"] == 19
    with pytest.raises(ValueError, match=r"indices \[0\].*retained all edges"):
        nav._plan((256.76, 134.29), 0)
    assert path.read_bytes() == before
    zone.evidence.retract([0])  # copied fixture only, not the live runtime file
    nav = NavigateTo("portal:NE", zone, cfg)
    nav._plan((256.76, 134.29), 0)
    assert nav.route


@pytest.mark.parametrize("start", [(255.66, 136.08), (256.76, 134.29)])
def test_short_road_clicks_from_run15_reach_portal_without_clipping_the_road(start):
    cfg = NavigationConfig(calibrated=True, minimap_to_screen=MATRIX, map_speed_px_per_s=0.529)
    zone = Zone.load("fractured_ground", cfg, learned=False)
    nav = NavigateTo("portal:NE", zone, cfg)
    point = Point(*start)
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(*point, None, 0.95))
    road_grid = ScaledWalkGrid(zone.grid.safe & zone.road, 4, clearance=0)
    for now in range(120):
        step = nav.tick(None, now, moving=now > 0)
        if step.status == "arrived":
            assert not nav.evidence.edges
            return
        assert step.status == "move" and step.message == "road click"
        assert 0 < step.click.dist(Point(*cfg.player_screen)) <= cfg.precision_click_max+1e-6
        assert not hud_reason(step.click, cfg.window_size, Config().safety)
        actual_delta = np.linalg.solve(MATRIX, np.asarray(step.click)-cfg.player_screen)
        end = Point(*(np.asarray(point)+actual_delta))
        assert road_grid.line_clear(point, end)
        assert end.dist(nav.goal) < point.dist(nav.goal)
        point = end
        nav.note_walk(now)
    pytest.fail(f"road following did not arrive: {point}")


def test_near_waypoint_click_does_not_extend_past_the_target():
    cfg = NavigationConfig(calibrated=True, minimap_to_screen=[[40, 0], [0, 40]],
                           arrival_tolerance=0.5, waypoint_tolerance=0.1)
    zone = Zone("test", Path("."), {"terrain_polygon": [[0, 0], [99, 0], [99, 99], [0, 99]]},
                WalkGrid(np.ones((100, 100), bool), clearance=0))
    nav = NavigateTo((80, 20), zone, cfg)
    point, target = Point(20, 20), Point(22.5, 20)
    nav.route = [point, target, nav.goal]
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(*point, None, 0.95))
    step = nav.tick(None, 0)
    assert step.message == "near click"
    delta = np.linalg.solve(cfg.minimap_to_screen, np.asarray(step.click)-cfg.player_screen)
    assert np.asarray(point)+delta == pytest.approx(target)
    assert step.click.dist(Point(*cfg.player_screen)) == pytest.approx(100)
    tiny = screen_click((0.1, 0), cfg, precise=True, max_distance=150)
    assert tiny.dist(Point(*cfg.player_screen)) == pytest.approx(4)


def road_task(tmp_path, thin=True):
    cfg = saved_zone(tmp_path)
    cfg.minimap_to_screen = [[100, 0], [0, 100]]
    zone = Zone.load("test", cfg)
    zone.grid.mask[:] = not thin
    zone.grid.mask[20, :] = True
    zone.grid._inflate()
    zone.base_grid = zone.grid.copy_with(0, 8)
    zone.road = np.zeros(zone.grid.mask.shape, bool)
    zone.road[20] = True
    nav = NavigateTo((35, 20), zone, cfg, learn=True)
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(20, 20, None, 0.95))
    return nav


def test_goal_disconnecting_road_stall_is_pending_and_short_retries_are_bounded(tmp_path, caplog):
    nav = road_task(tmp_path)
    waiting = 0
    for now in range(20):
        step = nav.tick(None, now, moving=now > 0)
        if step.click is not None:
            assert step.click.dist(Point(*nav.config.player_screen)) <= 150
            nav.note_walk(now)
        if step.status == "waiting":
            waiting += 1
        if step.status == "stalled":
            break
    assert step.status == "stalled" and waiting == 2
    assert not nav.evidence.edges and len(nav.evidence.road_stalls) == 2
    assert nav._connected(nav.grid, (20, 20), nav.goal)
    assert "would disconnect goal; retained pending" in caplog.text
    saved = json.loads(nav.evidence.path.read_text())
    assert not saved["edges"] and "disconnects current goal" in saved["road_stalls"][-1]["deferred_reason"]
    loaded = LearnedEvidence(nav.evidence.path, "test", (50, 40), 1)
    assert loaded.road_stalls == nav.evidence.road_stalls


def test_second_road_stall_can_still_learn_an_edge_when_an_alternative_exists(tmp_path):
    nav = road_task(tmp_path, thin=False)
    for now in range(8):
        step = nav.tick(None, now, moving=now > 0)
        if step.click is not None:
            nav.note_walk(now)
    assert len(nav.evidence.edges) == 1
    assert nav._connected(nav.grid, (20, 20), nav.goal)


def test_uncalibrated_road_never_proposes_a_precision_click():
    zone = Zone.load("fractured_ground", learned=False)
    nav = NavigateTo("portal:NE", zone)
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(256.76, 134.29, None, 0.95))
    step = nav.tick(None, 0)
    assert step.status == "uncalibrated" and step.click is None


def test_no_road_aim_never_falls_back_to_a_ray_across_a_road_gap(tmp_path):
    nav = road_task(tmp_path, thin=False)
    nav.zone.road[:] = False
    nav.zone.road[20, [20, 35]] = True  # uncertain road extraction; no forward road cell
    step = nav.tick(None, 0)
    assert step.status == "unsafe_click" and step.click is None
    assert not nav.evidence.edges


def test_precise_click_keeps_direction_even_in_a_tight_hud_free_rectangle():
    cfg = NavigationConfig(minimap_to_screen=MATRIX, safe_screen_rect=(1100, 540, 1150, 580))
    delta = np.array([0.1, -0.1])
    click = screen_click(delta, cfg, precise=True, max_distance=150)
    actual = np.linalg.solve(MATRIX, np.asarray(click)-cfg.player_screen)
    assert 0 < np.linalg.norm(actual) <= np.linalg.norm(delta)
    assert actual[0]/delta[0] == pytest.approx(actual[1]/delta[1])
    assert 1100 <= click.x <= 1150 and 540 <= click.y <= 580
