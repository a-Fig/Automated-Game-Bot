"""Build a cliff-aware zone mask from saved Tab maps, in live minimap pixels.

This tool reads images only. The dimmed live minimap in each capture supplies
terrain correspondences; the independently recorded Tab-closed fixes validate
the result. Fixes are never used to move a fitted transform to pass validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class Parameters:
    scale: int = 4
    rim_margin: float = 15.0
    ground_value: int = 165
    level_value: int = 215
    boundary_width: float = 0.25  # minimap pixels, before runtime wall clearance
    dark_rock_value: int = 120
    rock_max_area: float = 0.0  # size alone is not evidence of an obstacle
    water_saturation: int = 75
    block_neutral_grey: bool = False  # observed basin fixes cross these warm-grey patches
    road_saturation: int = 180
    road_value: int = 230
    road_gap: float = 4.0
    road_half_width: float = 1.5  # narrow ramps, with the zone's 1-pixel clearance


@dataclass
class Sample:
    image: np.ndarray
    origin: tuple[int, int]
    minimap: np.ndarray
    fix: np.ndarray
    name: str


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def load_sample(path, fix_path=None) -> Sample:
    path = Path(path)
    metadata_path = Path(fix_path) if fix_path else path.with_suffix(".json")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    before, after = np.asarray(data["minimap_before"], float), np.asarray(data["minimap_after"], float)
    if before.shape != (2,) or not np.isfinite(before).all() or np.linalg.norm(before-after) > 0.5:
        raise ValueError("capture needs a stationary, finite before/after live fix")
    image = read_image(path)
    if "image_origin" in data:  # cropped test/inspection capture, window coordinates retained
        minimap = read_image(metadata_path.parent / data["live_minimap"])
        origin = tuple(data["image_origin"])
    else:
        if image.shape[1::-1] != tuple(data["window"]) or tuple(data["window"]) != (2240, 1400):
            raise ValueError("expected a 2240x1400 Tab capture or a crop with image_origin")
        minimap = image[1040:1400, 1840:2240].copy()
        origin = (300, 230)
        image = image[230:1400, 300:1950].copy()
    if minimap.shape[:2] != (360, 400):
        raise ValueError("live minimap crop must be 400x360")
    return Sample(image, origin, minimap, (before+after)/2, path.name)


def project(points, matrix):
    points = np.asarray(points, float)
    return np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T


def homogeneous(matrix):
    return np.vstack((matrix, (0, 0, 1)))


def feature_pairs(image, reference, origin=(0, 0), source_scale=1.0, reference_mask=None):
    small = cv2.resize(image, None, fx=source_scale, fy=source_scale) if source_scale != 1 else image
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.01)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), reference_mask)
    if d1 is None or d2 is None or len(d2) < 2:
        raise ValueError("not enough map features for registration")
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    matches = [m for m, n in pairs if m.distance < 0.72*n.distance]
    source = np.asarray([np.asarray(k1[m.queryIdx].pt)/source_scale+origin for m in matches], np.float32)
    target = np.asarray([k2[m.trainIdx].pt for m in matches], np.float32)
    return source, target


def fit_similarity(source, target):
    if len(source) < 20:
        raise ValueError("need at least 20 terrain feature matches")
    matrix, inliers = cv2.estimateAffinePartial2D(np.float32(source), np.float32(target),
                                                method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                                maxIters=5000, refineIters=20)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("map similarity fit failed")
    selected = inliers.ravel().astype(bool)
    if selected.sum() < 30 or np.min(np.ptp(np.asarray(target)[selected], axis=0)) < 50:
        raise ValueError("map fit needs at least 30 inliers spread over the zone")
    residual = np.linalg.norm(project(source, matrix)-target, axis=1)
    return matrix, {"matches": len(source), "inliers": int(selected.sum()),
                    "median_error": float(np.median(residual[selected])),
                    "p95_error": float(np.percentile(residual[selected], 95))}


def check_live_registration(asset, minimap, max_offset=1.0):
    source, target = feature_pairs(asset, minimap)
    matrix, report = fit_similarity(source, target)
    inliers = np.linalg.norm(project(source, matrix)-target, axis=1) < 1
    delta = target[inliers]-source[inliers]
    translation = np.median(delta, axis=0)
    center = np.mean(source[inliers], axis=0)
    report.update(offset=float(np.linalg.norm(translation)), translation=translation.tolist(),
                  center_drift=float(np.linalg.norm(project([center], matrix)[0]-center)),
                  p95_displacement=float(np.percentile(np.linalg.norm(delta, axis=1), 95)),
                  offset_definition="median terrain displacement; similarity translation at off-map origin is not registration drift")
    if report["offset"] >= max_offset or report["center_drift"] >= max_offset or report["p95_displacement"] > 1.5:
        raise ValueError(f'asset/live registration offset {report["offset"]:.2f}px exceeds {max_offset}px')
    return report


def register(samples, reference):
    live_mask = np.zeros((360, 400), np.uint8)
    cv2.fillPoly(live_mask, [np.array([[210, 55], [375, 170], [210, 285], [40, 170]], np.int32)], 255)
    live_pairs, reference_pairs = [], []
    for sample in samples:
        # Half-size Tab features retain the detail around the narrow ramp bend;
        # quarter-size sampling made the four-capture rotation fit unstable.
        live_pairs.append(feature_pairs(sample.image, sample.minimap, sample.origin, 0.5, live_mask))
        reference_pairs.append(feature_pairs(sample.image, reference, sample.origin, 0.5))
    live, diagnostics = fit_similarity(np.concatenate([p[0] for p in live_pairs]),
                                       np.concatenate([p[1] for p in live_pairs]))
    old, old_diagnostics = fit_similarity(np.concatenate([p[0] for p in reference_pairs]),
                                         np.concatenate([p[1] for p in reference_pairs]))
    correction = (homogeneous(live) @ np.linalg.inv(homogeneous(old)))[:2]
    return live, correction, {"live_features": diagnostics, "reference_features": old_diagnostics,
                              "tab_to_reference": old.tolist(), "reference_to_live": correction.tolist()}


def find_arrow(sample, matrix):
    expected = project([sample.fix], np.linalg.inv(homogeneous(matrix))[:2])[0]-sample.origin
    hsv = cv2.cvtColor(sample.image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 60, 120), (115, 255, 255))
    region = np.zeros(mask.shape, np.uint8)
    cv2.circle(region, tuple(np.rint(expected).astype(int)), 60, 255, -1)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask & region)
    candidates = [i for i in range(1, count)
                  if 40 <= stats[i, 4] <= 400 and 10 <= stats[i, 2] <= 40 and 8 <= stats[i, 3] <= 40]
    if not candidates:
        raise ValueError(f"no Tab player arrow near fix in {sample.name}")
    i = min(candidates, key=lambda j: np.linalg.norm(centers[j]-expected))
    x, y, w, h, _ = stats[i]
    # The triangle's filled centroid moves with heading. Its bounding centre is
    # a more stable icon anchor across the three recorded orientations.
    center = np.asarray((x+(w-1)/2, y+(h-1)/2))+sample.origin
    erase = np.zeros(mask.shape, np.uint8)
    cv2.rectangle(erase, (max(0, x-3), max(0, y-3)), (x+w+2, y+h+2), 255, -1)
    return center, erase


def kernel(radius, scale):
    r = max(0, math.ceil(radius*scale))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))


def segment(image, footprint, params: Parameters):
    """Both levels and observed neutral-grey floor are open; retain cliff/blue/dark candidates."""
    p, scale = params, params.scale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ground = cv2.inRange(hsv, (5, 40, p.ground_value), (28, 190, 255)) & footprint
    light = cv2.inRange(hsv, (5, 40, p.level_value), (28, 190, 255)) & footprint
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ground)
    if count <= 1:
        raise ValueError("no light ground found inside the map footprint")
    main = (labels == 1+stats[1:, 4].argmax()).astype(np.uint8)*255
    contours, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside = np.zeros_like(main)
    cv2.drawContours(inside, contours, -1, 255, cv2.FILLED)  # include enclosed dark-level interiors
    edges = cv2.morphologyEx(light, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary = cv2.dilate(edges, kernel(p.boundary_width, scale)) & inside
    rock = cv2.inRange(hsv, (0, 0, 0), (180, 255, p.dark_rock_value))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(((light == 0) & (inside != 0)).astype(np.uint8))
    for i in range(1, count):
        if stats[i, 4] <= p.rock_max_area*scale*scale:
            rock[labels == i] = 255
    # The map's grey rocks can be warm-grey (H~11, S~69), not just blue-grey.
    grey = cv2.inRange(hsv, (0, 0, 40), (180, p.water_saturation, 230))
    water = cv2.inRange(hsv, (40, 0, 40), (140, 255, 230))
    if p.block_neutral_grey:
        water |= grey
    road = cv2.inRange(hsv, (7, p.road_saturation, p.road_value), (23, 255, 255)) & footprint
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road)
    long_roads = np.zeros_like(road)
    components = []
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        if area >= 8*scale*scale and math.hypot(w, h) >= 15*scale:
            long_roads[labels == i] = 255
            contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            components.append(np.concatenate(contours).reshape(-1, 2))
    # The map draws short pale gaps in otherwise continuous orange roads.
    # Join only already-long road components, bounded by road_gap; icons never seed ramps.
    for i, a in enumerate(components):
        for b in components[i+1:]:
            distances = np.sum((a[:, None, :].astype(float)-b[None, :, :])**2, axis=2)
            j, k = np.unravel_index(distances.argmin(), distances.shape)
            if distances[j, k] <= (p.road_gap*scale)**2:
                cv2.line(long_roads, tuple(a[j]), tuple(b[k]), 255, max(1, round(scale*0.4)))
    # The floor contour can exclude a dark gully at a ramp. An orange road is
    # evidence there too; clip it to the playable footprint, not the floor class.
    road = cv2.dilate(long_roads, kernel(p.road_half_width, scale)) & footprint
    inside |= road
    walkable = (inside != 0) & (boundary == 0) & (rock == 0) & (water == 0)
    walkable |= road != 0  # a road crossing a level boundary is a ramp
    return {"inside": inside != 0, "boundary": boundary != 0, "rock": rock != 0, "grey": grey != 0,
            "water": water != 0, "road": road != 0, "walkable": walkable}


def measure_observations(walkable, observations, scale, clearance, layers=None):
    from albot.navigation import ScaledWalkGrid
    grid = ScaledWalkGrid(walkable, scale, clearance)
    result = {}
    for name, points in observations.items():
        native = np.rint(np.asarray(points)*scale).astype(int)
        x, y = native[:, 0], native[:, 1]
        stats = {"samples": len(points), "unsafe": int(np.count_nonzero(~grid.safe[y, x]))}
        if layers is not None:
            stats["near_classes"] = {}
            for kind in ("boundary", "rock", "water", "grey"):
                near = cv2.dilate(layers[kind].astype(np.uint8), kernel(clearance, scale))
                stats["near_classes"][kind] = int(np.count_nonzero(near[y, x]))
        result[name] = stats
    return result


def apply_observations(layers, observations, scale, clearance, disc_radius=1.0, ordered_walks=()):
    """Small observed-position discs are explicit evidence, not colour guesses."""
    from albot.navigation import ScaledWalkGrid
    from albot.navigation_evidence import open_observed_disc
    if not 0 <= disc_radius <= 1.0:
        raise ValueError("observation discs must be between 0 and 1 minimap pixel")
    if any(name not in observations for name in ordered_walks):
        raise ValueError("ordered walks must name an observed-fixes source")
    for name, points in observations.items():
        points = np.asarray(points, float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                or (points < 0).any() or (points[:, 0] >= 400).any() or (points[:, 1] >= 360).any()):
            raise ValueError(f"{name}: expected finite observed positions inside 400x360")
        native = np.rint(points*scale).astype(int)
        if (native[:, 0] >= layers["walkable"].shape[1]).any() or (native[:, 1] >= layers["walkable"].shape[0]).any():
            raise ValueError(f"{name}: observed position rounds outside the mask")
    before = layers["walkable"].copy()
    report = {"classifier": measure_observations(before, observations, scale, clearance, layers),
              "disc_radius": disc_radius, "native_scale": scale, "effective_discs": [],
              "ordered_walks": list(ordered_walks)}
    walkable = before.astype(np.uint8)
    # Apply to every observed fix; report only discs that actually opened cells.
    for points in observations.values():
        if not disc_radius:
            break
        for point in np.rint(np.asarray(points)*scale).astype(int):
            # Observed standing positions supersede the estimated ground edge too.
            # Cropping by `inside` here would preserve the same false boundary.
            opened = open_observed_disc(walkable, point/scale, scale, disc_radius)
            if opened:
                report["effective_discs"].append({"center": (point/scale).tolist(),
                                                   "opened_cells": opened})
    layers = dict(layers)
    layers["evidence"] = (walkable != 0) & ~before
    layers["walkable"] = walkable != 0
    report["opened_native_cells"] = int(layers["evidence"].sum())
    report["opened_map_area"] = report["opened_native_cells"]/(scale*scale)
    report["after"] = measure_observations(layers["walkable"], observations, scale, clearance)
    grid = ScaledWalkGrid(layers["walkable"], scale, clearance)
    # Four-neighbour connectivity is also exactly the connectivity of A* with
    # diagonal corner cutting forbidden. Validate every consecutive fix pair.
    _, labels = cv2.connectedComponents(grid.safe.astype(np.uint8), connectivity=4)
    report["consecutive_pairs"] = {}
    for name, points in observations.items():
        if report["after"][name]["unsafe"]:
            raise ValueError(f"{name}: observed fixes still blocked or disconnected after bounded evidence discs")
        if name not in ordered_walks:
            continue
        native = np.rint(np.asarray(points)*scale).astype(int)
        ids = labels[native[:, 1], native[:, 0]]
        failures = np.flatnonzero((ids[:-1] != ids[1:]) | (ids[:-1] == 0) | (ids[1:] == 0))
        report["consecutive_pairs"][name] = {"checked": max(0, len(points)-1), "failed": failures.tolist()}
        if len(failures):
            raise ValueError(f"{name}: observed fixes still blocked or disconnected after bounded evidence discs")
    return layers, report


def build(samples, reference, params=None, observations=None, disc_radius=1.0,
          baseline=None, baseline_parameters=None, ordered_walks=(), registration_roi=None):
    p = params or Parameters()
    if (p.scale < 1 or p.rim_margin < 0 or p.boundary_width < 0 or p.rock_max_area < 0 or p.road_gap < 0
            or p.road_half_width < 0 or not 0 < p.ground_value <= p.level_value <= 255
            or not 0 <= p.dark_rock_value <= 255 or not 0 <= p.road_value <= 255
            or not 0 <= p.water_saturation <= 255 or not 0 <= p.road_saturation <= 255):
        raise ValueError("invalid mask parameters")
    matrix, correction, report = register(samples, reference)
    if registration_roi is not None:
        x0, y0, x1, y1 = registration_roi
        local = []
        for sample in samples:
            a, b = feature_pairs(sample.image, sample.minimap, sample.origin, 0.5)
            selected = (b[:, 0] >= x0) & (b[:, 0] <= x1) & (b[:, 1] >= y0) & (b[:, 1] <= y1)
            selected &= np.linalg.norm(b-sample.fix, axis=1) > 3  # terrain only, not the player icon
            residual = np.linalg.norm(project(a[selected], matrix)-b[selected], axis=1)
            if len(residual) < 2 or np.median(residual) > 1.5:
                raise ValueError(f"{sample.name}: insufficient local registration agreement in {registration_roi}")
            local.append({"capture": sample.name, "matches": len(residual),
                          "median_error": float(np.median(residual)), "max_error": float(residual.max())})
        report["local_registration"] = {"rect": registration_roi, "checks": local,
                                         "note": "tiny minimap features have about one pixel of sampling uncertainty"}
    size = (400*p.scale, 360*p.scale)
    aligned, checks = [], []
    for sample in samples:
        center, erase = find_arrow(sample, matrix)
        mapped = project([center], matrix)[0]
        error = float(np.linalg.norm(mapped-sample.fix))
        checks.append({"capture": sample.name, "tab_arrow": center.tolist(), "mapped_arrow": mapped.tolist(),
                       "live_fix": sample.fix.tolist(), "error": error})
        if error > 1.2:
            raise ValueError(f"{sample.name}: independent arrow error {error:.2f}px exceeds 1.2px")
        clean = cv2.inpaint(sample.image, erase, 5, cv2.INPAINT_TELEA)
        local_matrix = matrix.copy()
        local_matrix[:, 2] += matrix[:, :2] @ np.asarray(sample.origin)
        aligned.append(cv2.warpAffine(clean, local_matrix*p.scale, size))
    image = np.median(np.stack(aligned), axis=0).astype(np.uint8)
    outer_footprint = cv2.warpAffine((reference.max(axis=2) > 0).astype(np.uint8)*255,
                                     correction*p.scale, size, flags=cv2.INTER_NEAREST)
    # Exclude the tan decorative rim before finding the largest ground component.
    footprint = cv2.erode(outer_footprint, kernel(p.rim_margin, p.scale))
    layers = segment(image, footprint, p)
    if observations:
        layers, validation = apply_observations(layers, observations, p.scale, 1.0, disc_radius, ordered_walks)
        before_stats = None
        if baseline is not None:
            if baseline.shape != layers["walkable"].shape:
                raise ValueError("baseline mask dimensions must match the generated planning mask")
            previous = None
            if baseline_parameters is not None:
                old = Parameters(**baseline_parameters)
                previous = segment(image, cv2.erode(outer_footprint, kernel(old.rim_margin, p.scale)), old)
                validation["baseline_parameters"] = dataclasses.asdict(old)
            before_stats = measure_observations(baseline, observations, p.scale, 1.0, previous)
        if before_stats is not None:
            validation["before"] = before_stats
        report["observations"] = validation
    image[outer_footprint == 0] = 0
    mask = cv2.resize(layers["walkable"].astype(np.uint8)*255, (400, 360), interpolation=cv2.INTER_AREA) >= 128
    report.update(tab_to_map=matrix.tolist(), arrow_checks=checks, parameters=dataclasses.asdict(p),
                  coordinate_space="live_minimap", notes="Fixes validate only; no arrow-based transform adjustment.")
    asset = cv2.resize(image, (400, 360), interpolation=cv2.INTER_AREA)
    report["asset_live_checks"] = [check_live_registration(asset, sample.minimap) for sample in samples]
    return image, mask, layers, report


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def write_learned_check(output, image, layers, metadata, evidence_path, scale):
    """Audit saved collision evidence against this mask; never modify the evidence."""
    from albot.navigation import ScaledWalkGrid
    from albot.navigation_evidence import LearnedEvidence
    evidence = LearnedEvidence(evidence_path, metadata["name"].lower().replace(" ", "_"), (400, 360), scale)
    if not evidence.edges:
        raise ValueError("learned check needs at least one recorded stall")
    grid = evidence.apply(ScaledWalkGrid(layers["walkable"], scale, metadata["clearance"]))
    start, goal = evidence.edges[-1]["player"], metadata["portals"]["NE"]["approach"]
    route = grid.plan(start, goal)
    report = {"evidence": Path(evidence_path).name, "retained_edges": len(evidence.edges),
              "start": start, "goal": goal, "route": [list(p) for p in route]}
    (Path(output) / "ramp_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    overlay = image.copy()
    overlay[layers["road"]] = (overlay[layers["road"]]*0.4+np.array((40, 220, 40))*0.6).astype(np.uint8)
    for i, edge in enumerate(evidence.edges):
        a, b = [tuple(np.rint(np.asarray(edge[k])*scale).astype(int)) for k in ("a", "b")]
        cv2.line(overlay, a, b, (255, 0, 255), 1)
        cv2.putText(overlay, str(i), a, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    cv2.polylines(overlay, [np.rint(np.asarray(route)*scale).astype(int)], False, (255, 255, 255), 1)
    # The local ramp check's map coordinates; retain pixel axes for inspection.
    x0, y0, x1, y1 = metadata["registration_check_rect"]
    x0, y0, x1, y1 = x0-10, y0-5, x1+10, y1+8
    crop = cv2.resize(overlay[y0*scale:y1*scale, x0*scale:x1*scale], None,
                      fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for x in range(math.ceil(x0/5)*5, x1, 5):
        px = (x-x0)*scale*3
        cv2.line(crop, (px, 0), (px, crop.shape[0]-1), (80, 80, 80), 1)
        cv2.putText(crop, str(x), (px+2, 15), 0, 0.4, (255, 255, 255), 1)
    for y in range(math.ceil(y0/5)*5, y1, 5):
        py = (y-y0)*scale*3
        cv2.line(crop, (0, py), (crop.shape[1]-1, py), (80, 80, 80), 1)
        cv2.putText(crop, str(y), (2, py+15), 0, 0.4, (255, 255, 255), 1)
    crop = cv2.copyMakeBorder(crop, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(crop, "green: road; magenta: retained edges; white: route", (8, 20), 0, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    write_png(Path(output) / "ramp_overlay.png", crop)
    return report


def write_outputs(output, image, mask, layers, report, metadata, reset_override=False,
                  observations=None, baseline=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scale = report["parameters"]["scale"]
    correction = np.asarray(report["reference_to_live"])
    if metadata.get("coordinate_space") != "live_minimap":
        for portal in metadata["portals"].values():
            for key in ("marker", "approach"):
                portal[key] = project([portal[key]], correction)[0].tolist()
    contours, _ = cv2.findContours(layers["inside"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    metadata.update(terrain_polygon=(cv2.approxPolyDP(hull, scale, True).reshape(-1, 2)/scale).tolist(),
                    tab_to_map=report["tab_to_map"], coordinate_space="live_minimap",
                    mask_generator="tools/zone/tabmap_mask.py", mask_parameters=report["parameters"],
                    source="registered Tab captures: "+", ".join(c["capture"] for c in report["arrow_checks"]),
                    walkability_verified=False, clearance=1.0,
                    planning_mask="mask_hires.png", planning_scale=scale)
    from albot.navigation import ScaledWalkGrid
    route = []
    if "NE" in metadata["portals"]:
        start = report["arrow_checks"][0]["live_fix"]
        if observations and report.get("observations", {}).get("ordered_walks"):
            start = observations[report["observations"]["ordered_walks"][0]][0]
        route = ScaledWalkGrid(layers["walkable"], scale, clearance=metadata["clearance"]).plan(
            start, metadata["portals"]["NE"]["approach"])
        report["route_check"] = {"start": start, "goal": metadata["portals"]["NE"]["approach"],
                                 "clearance": metadata["clearance"], "waypoints": [list(p) for p in route]}
    write_png(output / "map.png", cv2.resize(image, (400, 360), interpolation=cv2.INTER_AREA))
    write_png(output / "mask.png", mask.astype(np.uint8)*255)
    write_png(output / "mask_hires.png", layers["walkable"].astype(np.uint8)*255)
    write_png(output / "road.png", cv2.resize(layers["road"].astype(np.uint8)*255, (400, 360), interpolation=cv2.INTER_AREA))
    if reset_override:
        write_png(output / "override.png", np.full((360, 400), 128, np.uint8))
    (output / "transform.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if observations:
        metadata["observed_fixes"] = list(observations)
        metadata["observed_walks"] = report["observations"]["ordered_walks"]
        for name, points in observations.items():
            path = output / name
            if not path.exists() or json.loads(path.read_text()) != points:
                path.write_text(json.dumps(points)+"\n", encoding="utf-8")
        (output / "observation_validation.json").write_text(json.dumps(report["observations"], indent=2)+"\n", encoding="utf-8")
        if baseline is not None:
            name = metadata.get("observation_baseline", {}).get("mask", "mask_before_basin.png")
            write_png(output / name, baseline.astype(np.uint8)*255)
    (output / "zone.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    overlay = image.copy()
    blocked = layers["inside"] & ~layers["walkable"]
    overlay[blocked] = (overlay[blocked]*0.4+np.array((30, 30, 230))*0.6).astype(np.uint8)
    road = layers["road"]
    overlay[road] = (overlay[road]*0.5+np.array((40, 220, 40))*0.5).astype(np.uint8)
    overlay[~layers["inside"]] //= 4
    if "observations" in report:
        overlay[layers["evidence"]] = (255, 0, 255)
        for disc in report["observations"]["effective_discs"]:
            point = tuple(np.rint(np.asarray(disc["center"])*scale).astype(int))
            cv2.circle(overlay, point, round(report["observations"]["disc_radius"]*scale), (255, 0, 255), 1)
        count = report["observations"]["opened_native_cells"]
        cv2.putText(overlay, f"magenta: observed-position discs; {count} native cells opened", (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1, cv2.LINE_AA)
    for i, check in enumerate(report["arrow_checks"]):
        point = tuple(np.rint(np.asarray(check["live_fix"])*scale).astype(int))
        cv2.circle(overlay, point, 4, (255, 255, 0), 1)
        cv2.putText(overlay, f'fix{i+1} {check["error"]:.2f}px', (point[0]+8, point[1]-i*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for name, portal in metadata["portals"].items():
        point = tuple(np.rint(np.asarray(portal["approach"])*scale).astype(int))
        cv2.circle(overlay, point, 5, (255, 100, 255), 1)
        cv2.putText(overlay, name, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
    if route:
        cv2.polylines(overlay, [np.rint(np.asarray(route)*scale).astype(np.int32)], False, (255, 255, 255), 2)
    cv2.putText(overlay, f"red: blocked | green: road/ramps | cyan: fixes | white: NE route | map pixels x{scale}", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    write_png(output / "mask_overlay.png", overlay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs="+", required=True)
    parser.add_argument("--fixes", nargs="+", help="defaults to each capture's adjacent JSON")
    parser.add_argument("--reference", default="assets/zones/fractured_ground/map.png")
    parser.add_argument("--metadata", default="assets/zones/fractured_ground/zone.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reset-override", action="store_true", help="replace mixed legacy override with neutral grey")
    parser.add_argument("--observed-fixes", nargs="+", help="live-space fix files; defaults to zone.json observed_fixes")
    parser.add_argument("--ordered-walks", nargs="*", help="observed filenames whose consecutive fixes must connect")
    parser.add_argument("--disc-radius", type=float, default=1.0, help="last-resort observed disc radius, at most 1 map px")
    parser.add_argument("--baseline-mask", help="previous high-resolution mask for the audit report")
    parser.add_argument("--check-learned", help="saved evidence for a read-only local ramp overlay/route check")
    for field in dataclasses.fields(Parameters):
        if isinstance(field.default, bool):
            parser.add_argument("--"+field.name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                                default=field.default)
        else:
            parser.add_argument("--"+field.name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    if args.fixes and len(args.fixes) != len(args.captures):
        parser.error("--fixes must have one entry per capture")
    samples = [load_sample(path, args.fixes[i] if args.fixes else None) for i, path in enumerate(args.captures)]
    parameters = Parameters(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Parameters)})
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    paths = args.observed_fixes if args.observed_fixes is not None else [Path(args.metadata).parent / p
                                                                     for p in metadata.get("observed_fixes", [])]
    observations = {Path(path).name: json.loads(Path(path).read_text()) for path in paths}
    if len(observations) != len(paths):
        raise ValueError("observed fix files must have distinct names")
    baseline_info = metadata.get("observation_baseline", {})
    baseline_path = args.baseline_mask or (Path(args.metadata).parent / baseline_info["mask"]
                                           if "mask" in baseline_info else None)
    baseline = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE) if baseline_path else None
    if baseline_path and baseline is None:
        raise ValueError(f"cannot read baseline mask: {baseline_path}")
    if baseline is not None:
        baseline = baseline != 0
    image, mask, layers, report = build(samples, read_image(args.reference), parameters, observations,
                                       args.disc_radius, baseline, baseline_info.get("parameters"),
                                       args.ordered_walks if args.ordered_walks is not None else metadata.get("observed_walks", []),
                                       metadata.get("registration_check_rect"))
    write_outputs(args.out_dir, image, mask, layers, report, metadata, args.reset_override, observations, baseline)
    if args.check_learned:
        write_learned_check(args.out_dir, image, layers, metadata, args.check_learned, parameters.scale)
    print(json.dumps(report, indent=2))
    print(f"mask/overlay written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
