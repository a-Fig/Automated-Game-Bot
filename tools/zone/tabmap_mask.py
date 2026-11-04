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
    rim_margin: float = 10.0
    ground_value: int = 165
    level_value: int = 205
    boundary_width: float = 0.5  # minimap pixels, before runtime wall clearance
    dark_rock_value: int = 160
    rock_max_area: float = 10.0  # small isolated dark-level blobs, in map pixels squared
    water_saturation: int = 75
    road_saturation: int = 135
    road_value: int = 210
    road_gap: float = 4.0
    road_half_width: float = 2.8  # survives the runtime's 2-pixel clearance


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
        live_pairs.append(feature_pairs(sample.image, sample.minimap, sample.origin, 0.25, live_mask))
        reference_pairs.append(feature_pairs(sample.image, reference, sample.origin, 0.25))
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
    """Both flat levels are open; block their boundaries, rock blobs and grey patches."""
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
    water = cv2.inRange(hsv, (0, 0, 40), (180, p.water_saturation, 230))
    water |= cv2.inRange(hsv, (40, 0, 40), (140, 255, 230))
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
    road = cv2.dilate(long_roads, kernel(p.road_half_width, scale)) & inside
    walkable = (inside != 0) & (boundary == 0) & (rock == 0) & (water == 0)
    walkable |= road != 0  # a road crossing a level boundary is a ramp
    return {"inside": inside != 0, "boundary": boundary != 0, "rock": rock != 0,
            "water": water != 0, "road": road != 0, "walkable": walkable}


def build(samples, reference, params=None):
    p = params or Parameters()
    if (p.scale < 1 or p.rim_margin < 0 or p.boundary_width < 0 or p.rock_max_area < 0 or p.road_gap < 0
            or p.road_half_width < 0 or not 0 < p.ground_value <= p.level_value <= 255
            or not 0 <= p.dark_rock_value <= 255 or not 0 <= p.road_value <= 255
            or not 0 <= p.water_saturation <= 255 or not 0 <= p.road_saturation <= 255):
        raise ValueError("invalid mask parameters")
    matrix, correction, report = register(samples, reference)
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


def write_outputs(output, image, mask, layers, report, metadata, reset_override=False):
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
    (output / "zone.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    (output / "transform.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    overlay = image.copy()
    blocked = layers["inside"] & ~layers["walkable"]
    overlay[blocked] = (overlay[blocked]*0.4+np.array((30, 30, 230))*0.6).astype(np.uint8)
    road = layers["road"]
    overlay[road] = (overlay[road]*0.5+np.array((40, 220, 40))*0.5).astype(np.uint8)
    overlay[~layers["inside"]] //= 4
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
    for field in dataclasses.fields(Parameters):
        parser.add_argument("--"+field.name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    if args.fixes and len(args.fixes) != len(args.captures):
        parser.error("--fixes must have one entry per capture")
    samples = [load_sample(path, args.fixes[i] if args.fixes else None) for i, path in enumerate(args.captures)]
    parameters = Parameters(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Parameters)})
    image, mask, layers, report = build(samples, read_image(args.reference), parameters)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    write_outputs(args.out_dir, image, mask, layers, report, metadata, args.reset_override)
    print(json.dumps(report, indent=2))
    print(f"mask/overlay written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
