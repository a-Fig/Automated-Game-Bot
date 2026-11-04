import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from albot.config import NavigationConfig, load
from albot.geometry import Point
from albot.navigation import NavigateTo, ScaledWalkGrid, WalkGrid, Zone
from tools.zone.tabmap_mask import (Parameters, build, check_live_registration, load_sample,
                                    project, read_image, segment, write_outputs, apply_observations, write_learned_check)


FIXTURES = Path(__file__).parent / "fixtures/tabmap"


@pytest.fixture(scope="module")
def observations():
    folder = Path(__file__).parent.parent / "assets/zones/fractured_ground"
    return {name: json.loads((folder / name).read_text()) for name in ("walk_ne_fixes.json", "visited.json")}


@pytest.fixture(scope="module")
def generated(observations):
    samples = [load_sample(FIXTURES / f"fg_0{i}.png") for i in range(1, 5)]
    folder = Path(__file__).parent.parent / "assets/zones/fractured_ground"
    metadata = json.loads((folder / "zone.json").read_text())
    return samples, build(samples, read_image(FIXTURES / "reference_old.png"), observations=observations,
                           baseline=read_image(folder / "mask_before_basin.png")[:, :, 0] != 0,
                           baseline_parameters=metadata["observation_baseline"]["parameters"],
                           ordered_walks=["walk_ne_fixes.json"], registration_roi=metadata["registration_check_rect"])


def test_tab_terrain_transform_recovers_independent_live_fixes(generated):
    samples, (_, _, _, report) = generated
    assert report["live_features"]["inliers"] > 200
    assert report["reference_features"]["inliers"] > 700
    for sample, check in zip(samples, report["arrow_checks"]):
        recovered = project([check["tab_arrow"]], report["tab_to_map"])[0]
        assert np.linalg.norm(recovered-sample.fix) < 1.0
    assert report["reference_to_live"][1][2] > 7
    local = report["local_registration"]["checks"][-1]
    assert local["capture"] == "fg_04.png" and local["matches"] >= 6
    assert local["median_error"] < 1.5
    assert report["arrow_checks"][-1]["error"] < 0.5


def test_asset_live_registration_stays_below_one_pixel_and_rejects_drift(generated):
    samples, (image, _, _, _) = generated
    asset = cv2.resize(image, (400, 360), interpolation=cv2.INTER_AREA)
    for sample in samples:
        assert check_live_registration(asset, sample.minimap)["offset"] < 1
    shifted = cv2.warpAffine(asset, np.float32([[1, 0, 0], [0, 1, -5]]), (400, 360))
    with pytest.raises(ValueError, match="registration offset"):
        check_live_registration(shifted, samples[0].minimap)
    with pytest.raises(ValueError, match="registration offset"):
        check_live_registration(read_image(FIXTURES / "reference_old.png"), samples[0].minimap)


def test_corrected_zone_default_and_explicit_config_precedence(tmp_path):
    cfg = NavigationConfig()
    assert cfg.clearance is None
    zone = Zone.load("fractured_ground", cfg)
    assert zone.grid.clearance == 1.0
    assert NavigateTo("portal:NE", zone, cfg).grid.clearance == 1.0
    config = tmp_path / "explicit.toml"
    config.write_text("[navigation]\nclearance = 2.0\n")
    explicit = load(str(config)).navigation
    assert Zone.load("fractured_ground", explicit).grid.clearance == 2
    assert NavigateTo("portal:NE", zone, explicit).grid.clearance == 2
    fake = tmp_path / "zone_without_default"
    fake.mkdir()
    metadata = dict(zone.metadata)
    metadata.pop("clearance")
    metadata.pop("planning_scale")
    metadata.pop("planning_mask")
    (fake / "zone.json").write_text(json.dumps(metadata))
    cv2.imwrite(str(fake / "mask.png"), np.full((360, 400), 255, np.uint8))
    cv2.imwrite(str(fake / "override.png"), np.full((360, 400), 128, np.uint8))
    cfg.assets_folder = str(tmp_path)
    assert Zone.load("zone_without_default", cfg).grid.clearance == 2


def route_samples(route, scale):
    for a, b in zip(route, route[1:]):
        for t in np.linspace(0, 1, max(2, int(a.dist(b)*scale*4))):
            yield round((a.x+(b.x-a.x)*t)*scale), round((a.y+(b.y-a.y)*t)*scale)


def test_route_to_ne_uses_road_at_level_crossings_and_does_not_cut_cliff(generated):
    _, (_, mask, layers, report) = generated
    zone = Zone.load("fractured_ground")
    scale = report["parameters"]["scale"]
    grid = ScaledWalkGrid(layers["walkable"], scale, clearance=zone.grid.clearance)
    route = grid.plan((240, 130), zone.goal("portal:NE"))
    assert route[-1].dist(zone.goal("portal:NE")) < 1
    assert all(grid.line_clear(a, b) for a, b in zip(route, route[1:]))
    live_route = grid.plan((240.87, 130.51), zone.goal("portal:NE"))
    assert max(p.x for p in live_route) < 400
    assert max(p.y for p in live_route) < 360
    assert zone.grid.scale == 4
    assert not layers["inside"][45*scale, 210*scale]  # decorative rim is not playable ground
    crossings = [(x, y) for x, y in route_samples(route, scale) if layers["boundary"][y, x]]
    assert crossings
    assert all(layers["road"][y, x] for x, y in crossings)
    assert any(p.dist(Point(241, 142)) < 4 for p in route)
    assert not grid.line_clear((240, 130), (255, 130))  # real level boundary, away from a ramp


def test_scaled_grid_keeps_map_coordinates_overrides_clearance_and_private_blocks(tmp_path):
    path, override = tmp_path / "hires.png", tmp_path / "override.png"
    mask = np.ones((80, 80), np.uint8)*255
    cv2.imwrite(str(path), mask)
    paint = np.full((20, 20), 128, np.uint8)
    paint[10, 10] = 0
    cv2.imwrite(str(override), paint)
    grid = ScaledWalkGrid.load(path, override, 4, clearance=1, snap_radius=2)
    assert not grid.mask[40:44, 40:44].any()
    assert not grid.is_safe((10.5, 10.5))
    copy = grid.copy_with(1, 2)
    copy.block((5, 5), 1)
    assert grid.is_safe((5, 5)) and not copy.is_safe((5, 5))
    route = grid.plan((3, 10), (17, 10))
    assert route[0] == Point(3, 10) and route[-1] == Point(17, 10)
    assert any(p.y != 10 for p in route)
    assert all(grid.line_clear(a, b) for a, b in zip(route, route[1:]))


def test_both_levels_walkable_but_edges_rocks_and_grey_patches_blocked():
    p = Parameters(scale=2, road_half_width=1, rock_max_area=4)
    image = np.full((200, 200, 3), (120, 165, 225), np.uint8)
    image[50:150, 50:150] = (90, 130, 190)
    image[80:92, 80:92] = (45, 65, 100)
    image[110:125, 80:95] = (150, 145, 125)
    image[115:128, 112:125] = (122, 139, 167)  # real capture's warm grey, HSV H~11/S~69
    image[95:100, 15:185] = (60, 160, 245)
    footprint = np.zeros((200, 200), np.uint8)
    footprint[10:190, 10:190] = 255
    layers = segment(image, footprint, p)
    assert layers["walkable"][30, 30] and layers["walkable"][70, 70]
    assert not layers["walkable"][85, 85] and not layers["walkable"][116, 86]
    assert layers["grey"][120, 118] and not layers["water"][120, 118]
    assert layers["walkable"][120, 118]  # observed neutral/warm-grey is not an obstacle class
    assert not layers["walkable"][50, 80]
    assert layers["road"][97, 50] and layers["walkable"][97, 50]
    assert np.any(layers["boundary"] & layers["road"] & layers["walkable"])
    assert not layers["walkable"][0, 0]
    wider = segment(image, footprint, Parameters(scale=2, boundary_width=1.5, road_half_width=1, rock_max_area=4))
    assert wider["boundary"].sum() > layers["boundary"].sum()


def test_generated_outputs_are_png_masks_and_neutral_override(tmp_path, generated, observations):
    _, (image, mask, layers, report) = generated
    metadata = json.loads((FIXTURES / "reference_old.json").read_text())
    write_outputs(tmp_path, image, mask, layers, dict(report), metadata, reset_override=True,
                   observations=observations)
    assert np.array_equal(cv2.imread(str(tmp_path / "mask.png"), 0) != 0, mask)
    assert cv2.imread(str(tmp_path / "mask_hires.png"), 0).shape == (1440, 1600)
    assert cv2.imread(str(tmp_path / "mask_overlay.png")).shape == (1440, 1600, 3)
    assert np.all(cv2.imread(str(tmp_path / "override.png"), 0) == 128)
    zone = json.loads((tmp_path / "zone.json").read_text())
    assert zone["coordinate_space"] == "live_minimap" and zone["clearance"] == 1
    assert zone["portals"]["NE"]["approach"][1] > 123
    validation = json.loads((tmp_path / "observation_validation.json").read_text())
    assert validation["opened_native_cells"] == 2
    assert validation["disc_radius"] == 1
    old_goal = list(zone["portals"]["NE"]["approach"])
    write_outputs(tmp_path, image, mask, layers, dict(report), zone)
    assert zone["portals"]["NE"]["approach"] == old_goal


def test_inconsistent_fixes_fail_independent_arrow_check(generated):
    samples, _ = generated
    from dataclasses import replace
    wrong = [replace(sample, fix=sample.fix+np.array((0, 5))) for sample in samples]
    with pytest.raises(ValueError, match="independent arrow error"):
        build(wrong, read_image(FIXTURES / "reference_old.png"))


def test_local_ramp_overlay_keeps_run8_edges_and_bends_around_cliff(tmp_path, generated):
    _, (image, _, layers, report) = generated
    folder = Path(__file__).parent.parent / "assets/zones/fractured_ground"
    metadata = json.loads((folder / "zone.json").read_text())
    evidence = FIXTURES.parent / "navigation/learned_after_run8.json"
    before = evidence.read_bytes()
    check = write_learned_check(tmp_path, image, layers, metadata, evidence, report["parameters"]["scale"])
    assert check["retained_edges"] == 8 and max(p[1] for p in check["route"]) >= 146
    assert (tmp_path / "ramp_overlay.png").exists() and evidence.read_bytes() == before


def test_all_observed_fixes_are_directly_safe_and_every_walk_pair_plans(observations):
    zone = Zone.load("fractured_ground")
    assert len(observations["walk_ne_fixes.json"]) == 536
    assert len(observations["visited.json"]) == 723
    for points in observations.values():
        assert all(zone.grid.is_safe(point) for point in points)
    points = observations["walk_ne_fixes.json"]
    for start, goal in zip(points, points[1:]):
        route = zone.grid.plan(start, goal)
        assert route
        assert all(zone.grid.line_clear(a, b) for a, b in zip(route, route[1:]))
    for start in ((221.6, 121.2), (220.2, 124.1)):
        route = zone.grid.plan(start, zone.goal("portal:NE"))
        assert route[-1].dist(zone.goal("portal:NE")) < 0.5


def test_observation_discs_stay_tiny_and_report_prior_false_blocks(generated):
    _, (_, _, layers, report) = generated
    audit = report["observations"]
    assert audit["before"]["walk_ne_fixes.json"]["unsafe"] == 267
    assert audit["before"]["visited.json"]["unsafe"] == 374
    assert audit["classifier"]["walk_ne_fixes.json"]["unsafe"] == 0
    assert audit["classifier"]["visited.json"]["unsafe"] == 2
    assert audit["opened_native_cells"] == int(layers["evidence"].sum()) == 2
    assert audit["opened_map_area"] == 0.125
    assert sum(d["opened_cells"] for d in audit["effective_discs"]) == 2
    centers = np.array([d["center"] for d in audit["effective_discs"]])*4
    for y, x in np.argwhere(layers["evidence"]):
        assert np.min(np.linalg.norm(centers-[x, y], axis=1)) <= 4
    assert all(value["unsafe"] == 0 for value in audit["after"].values())
    assert all(not value["failed"] for value in audit["consecutive_pairs"].values())
    stored = json.loads((Path(__file__).parent.parent / "assets/zones/fractured_ground/observation_validation.json").read_text())
    assert stored == audit


def test_observation_validation_rejects_unproven_large_discs_and_no_evidence(generated, observations):
    _, (_, _, layers, _) = generated
    raw = dict(layers, walkable=layers["walkable"] & ~layers["evidence"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_observations(raw, observations, 4, 1, disc_radius=2)
    with pytest.raises(ValueError, match="ordered walks"):
        apply_observations(raw, observations, 4, 1, ordered_walks=["missing.json"])
    with pytest.raises(ValueError, match="blocked or disconnected"):
        apply_observations(raw, observations, 4, 1, disc_radius=0)
