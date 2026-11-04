import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from albot.config import NavigationConfig
from albot.geometry import Point
from albot.navigation import Location, NavigateTo, ScaledWalkGrid, WalkGrid, Zone
from albot.navigation_evidence import LearnedEvidence
from tools.zone.replay_travel import read_positions, replay


def saved_zone(tmp_path, scale=1):
    folder = tmp_path / "test"
    folder.mkdir()
    mask = np.ones((40, 50), np.uint8)*255
    mask[10, 10] = 0
    cv2.imwrite(str(folder / "mask.png"), mask)
    cv2.imwrite(str(folder / "override.png"), np.full(mask.shape, 128, np.uint8))
    cv2.imwrite(str(folder / "mask_hires.png"), cv2.resize(mask, (50*scale, 40*scale),
                                                       interpolation=cv2.INTER_NEAREST))
    (folder / "zone.json").write_text(json.dumps({"clearance": 0, "planning_scale": scale,
        "planning_mask": "mask_hires.png", "terrain_polygon": [[0, 0], [49, 0], [49, 39], [0, 39]]}))
    return NavigationConfig(assets_folder=str(tmp_path), zone="test", minimap_rect=(0, 0, 50, 40),
                            calibrated=True, arrival_tolerance=0.5, waypoint_tolerance=0.5)


def test_learning_roundtrip_loads_discs_and_edges_without_changing_base_mask(tmp_path):
    cfg = saved_zone(tmp_path)
    zone = Zone.load("test", cfg)
    evidence = zone.evidence
    assert evidence.observe((10, 10))
    assert not evidence.observe((10.1, 10.1))  # same native-cell disc
    assert evidence.stall((20, 20), (35, 20), 0, 6, 12.0)
    loaded = Zone.load("test", cfg)
    assert loaded.grid.is_safe((10, 10)) and not loaded.base_grid.is_safe((10, 10))
    assert not loaded.grid.line_clear((20, 20), (35, 20))
    assert loaded.grid.is_safe((20, 20))
    assert loaded.evidence.edges == evidence.edges
    assert not (zone.folder / "learned.json.tmp").exists()
    raw = cv2.imread(str(zone.folder / "mask.png"), 0)
    assert raw[10, 10] == 0 and raw[20, 23] == 255
    data = json.loads((zone.folder / "learned.json").read_text())
    data["coordinate_space"] = "obsolete_asset"
    (zone.folder / "learned.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="coordinates"):
        Zone.load("test", cfg)


@pytest.mark.parametrize("scale,clearance", [(1, 0), (1, 2), (4, 1), (4, 2)])
def test_edges_preserve_player_and_observed_cells_after_reload(tmp_path, scale, clearance):
    path = tmp_path / "learned.json"
    evidence = LearnedEvidence(path, "test", (50, 40), scale)
    for player, target in [((20.2, 20.4), (30, 20)), ((22, 18), (22, 30)), ((24, 20), (14, 20))]:
        evidence.observe(player)
        evidence.stall(player, target, clearance, 6, 0)
    assert not evidence.stall((20, 20), (20.1, 20), clearance, 6, 0)
    loaded = LearnedEvidence(path, "test", (50, 40), scale)
    grid = ScaledWalkGrid(np.ones((40*scale, 50*scale), bool), scale, clearance)
    loaded.apply(grid)
    for point in [*loaded.positions.values(), *(e["player"] for e in loaded.edges)]:
        assert grid.is_safe(point)


def simulate(cfg):
    zone = Zone.load("test", cfg)
    nav = NavigateTo((35, 20), zone, cfg, learn=True)
    real = WalkGrid(np.ones((40, 50), bool), clearance=0)
    real.mask[19:22, 25] = False  # absent from the colour mask
    real._inflate()
    position, stalls, collisions = Point(20, 20), 0, 0
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(*position, None, 0.95))
    moving = False
    for now in range(200):
        step = nav.tick(None, now, moving=moving)
        if step.status == "arrived":
            return stalls, collisions, nav
        if step.status == "waiting":
            stalls += 1
        if step.click is not None:
            vector = np.linalg.solve(cfg.minimap_to_screen, np.asarray(step.click)-cfg.player_screen)
            vector /= np.linalg.norm(vector)
            end = Point(*(np.asarray(position)+vector*min(1, position.dist(nav.route[nav.index]))))
            if real.line_clear(position, end):
                position = end
            else:
                collisions += 1
            moving = True
    pytest.fail(f"simulation failed to arrive: {position}, {step}")


@pytest.mark.parametrize("scale", [1, 4])
def test_two_runs_converge_around_a_hidden_wall(tmp_path, scale):
    cfg = saved_zone(tmp_path, scale)
    first_stalls, first_collisions, first = simulate(cfg)
    assert first_stalls > 0 and first_collisions > 0 and first.evidence.edges
    second_stalls, second_collisions, second = simulate(cfg)
    assert second_stalls == 0 and second_collisions == 0
    assert second.evidence.edges == first.evidence.edges


def test_preview_never_writes_learning_and_curved_motion_is_progress(tmp_path):
    cfg = saved_zone(tmp_path)
    nav = NavigateTo((35, 20), Zone.load("test", cfg), cfg)
    position = Point(20, 20)
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(*position, None, 0.95))
    # Move away from the goal while local pathfinding skirts an obstacle.
    for now in range(10):
        position = Point(20, 20+now)
        assert nav.tick(None, now).status == "move"
        nav.record_position(Location(*position, None, 0.95))
    assert not nav.stalls
    for now in range(10, 16):
        nav.tick(None, now)
    assert nav.stalls
    assert not (tmp_path / "test/learned.json").exists()


def test_retry_budget_is_per_spot_and_standing_preview_does_not_open_ground(tmp_path):
    cfg = saved_zone(tmp_path)
    cfg.max_replans = 1
    nav = NavigateTo((35, 20), Zone.load("test", cfg), cfg)
    position = Point(20, 20)
    nav.locator = SimpleNamespace(locate=lambda frame, now: Location(*position, None, 0.95))
    for now in range(9):
        step = nav.tick(None, now)
    assert step.status == "stalled" and nav.replans == 1
    position = Point(10, 30)
    for now in range(9, 14):
        step = nav.tick(None, now)
    assert step.status == "waiting" and nav.replans == 2
    assert not nav.evidence.positions  # no dispatched movement was reported
    position = Point(10, 10)  # colour mask falsely blocks this known standing fix
    step = nav.tick(None, 14, moving=True)
    assert nav.grid.is_safe(position) and nav.evidence.positions


def test_walkable_evidence_does_not_erase_retreat_hazards(tmp_path):
    cfg = saved_zone(tmp_path)
    nav = NavigateTo((35, 20), Zone.load("test", cfg), cfg)
    nav.block_hazard(Point(20, 20), 3)
    nav.record_position(Location(20, 20, None, 0.95))
    assert not nav.grid.is_safe((20, 20))
    nav.grid = nav._fresh_grid()  # terrain learning/replanning preserves temporary danger
    assert not nav.grid.is_safe((20, 20))
    assert nav.zone.grid.is_safe((20, 20)) and not nav.evidence.edges


def test_replay_identifies_the_unreached_ramp_and_excludes_post_travel(tmp_path):
    fixture = Path(__file__).parent / "fixtures/navigation/travel6.log"
    bot = tmp_path / "run.log"
    bot.write_text("04:27:56 INFO travel move: \n04:28:35 INFO travel stalled: \n"
                   "04:29:05 ERROR travel gave up\n")
    report = replay(fixture, Zone.load("fractured_ground"), bot_log=bot)
    stall = report["stalls"][-1]
    assert stall["from"] == "04:28:31" and stall["seconds"] == 31
    assert Point(*stall["position"]).dist(Point(238.3, 136.2)) < 0.1
    assert Point(*stall["planned_next"]).dist(Point(240, 141)) < 1
    assert report["controller_stalled_at"] == "04:28:35"
    assert all(p["time"] <= "04:29:05" for p in report["plans"])
    assert report["excluded_outside_travel"] > 0 and report["disagreements"]
    assert len(read_positions(fixture)) == 25
