"""Map perception and navigation decisions. Nothing in this module sends input.

Map coordinates are pixels relative to the configured minimap crop, including
when a larger Tab map was used to build the zone mask. N/E/S/W name map corners;
the measured 2x2 matrix, not compass labels, determines screen movement.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .config import NavigationConfig
from .geometry import Point
from .navigation_evidence import LearnedEvidence, open_observed_disc

log = logging.getLogger(__name__)


class Location(NamedTuple):
    x: float
    y: float
    heading: float | None  # radians clockwise from map +x, if the tiny triangle is clear
    confidence: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


def as_bgr(frame):
    """Same contract as Capture: PIL is RGB, numpy is OpenCV BGR. Never guess."""
    if hasattr(frame, "convert"):
        return cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a PIL image or a three-channel BGR array")
    return image


def locate(frame, config: NavigationConfig | None = None, previous: Location | None = None,
           polygon=None, max_jump: float = 20.0) -> Location | None:
    """Find the blue arrow; return None on occlusion, ambiguity or wrong UI scale.

    Correct BGR->HSV gives H about 104 on the recorded arrow, not H about 10.
    The castle has a small blue patch too: the arrow's dark triangular border
    distinguishes it. Continuity rejects distant candidates instead of teleporting.
    """
    cfg = config or NavigationConfig()
    image = as_bgr(frame)
    if image.shape[1::-1] != tuple(cfg.window_size):
        return None
    x0, y0, x1, y1 = cfg.minimap_rect
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("minimap_rect is outside the frame")
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, cfg.arrow_min_saturation, cfg.arrow_min_value), (112, 220, 255))
    region = np.zeros(mask.shape, np.uint8)
    boundary = polygon if polygon is not None else [(210, 55), (365, 161), (210, 267), (52, 162)]
    cv2.fillPoly(region, [np.asarray(boundary, np.int32)], 255)
    mask &= region
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # A sideways arrow measured 15 px wide live (2026-11-04), so allow 16.
        if not (12 <= area <= 100 and 4 <= w <= 16 and 4 <= h <= 16 and 0.2 <= area / (w * h) <= 0.85):
            continue
        component = (labels == i).astype(np.uint8)
        value = float(hsv[:, :, 2][component != 0].mean())
        if value < cfg.arrow_min_mean_value:
            continue
        rim = cv2.dilate(component, np.ones((3, 3), np.uint8)) - component
        border = float((crop.max(axis=2)[rim != 0] < 90).mean())
        if border < cfg.arrow_min_dark_border:
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        vertices = cv2.approxPolyDP(hull, 0.08 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(vertices) not in (3, 4):
            continue
        cx, cy = centers[i]
        heading = None
        if len(vertices) == 3:
            lengths = sorted(((math.hypot(px - cx, py - cy), px, py) for px, py in vertices), reverse=True)
            if lengths[0][0] > lengths[1][0] * 1.15:
                heading = math.atan2(lengths[0][2] - cy, lengths[0][1] - cx)
        confidence = min(0.99, 0.6 + 0.25 * border + 0.14 * (value / 255))
        point = Location(float(cx), float(cy), heading, confidence)
        if previous is None or point.point.dist(previous.point) <= max_jump:
            candidates.append(point)
    if not candidates:
        return None
    if previous is not None:
        candidates.sort(key=lambda p: p.point.dist(previous.point))
        if len(candidates) > 1 and abs(candidates[0].point.dist(previous.point)
                                      - candidates[1].point.dist(previous.point)) < 2:
            return None
    else:
        candidates.sort(key=lambda p: p.confidence, reverse=True)
        if len(candidates) > 1 and candidates[0].confidence - candidates[1].confidence < 0.08:
            return None
    return candidates[0]


class MapLocator:
    def __init__(self, config: NavigationConfig, polygon=None) -> None:
        self.config, self.polygon = config, polygon
        self.previous = None
        self.timestamp = None

    def locate(self, frame, now: float) -> Location | None:
        gap = now - self.timestamp if self.timestamp is not None else math.inf
        if gap < 0:
            raise ValueError("navigation timestamps must increase")
        # Long occlusions require a fresh unambiguous fix, not extrapolation.
        previous = self.previous if gap <= 2 else None
        jump = max(4.0, gap * self.config.map_speed_px_per_s * 2) if previous else 20.0
        result = locate(frame, self.config, previous, self.polygon, jump)
        if result is not None:
            self.previous, self.timestamp = result, now
        return result


def build_walkability(frame, polygon, size=(400, 360), transform=None,
                      ground_lower=(5, 45, 155), ground_upper=(28, 180, 255),
                      road_lower=(5, 130, 175), road_upper=(28, 255, 255)) -> np.ndarray:
    """Colour hypothesis, not verified terrain. Transform a Tab map into minimap pixels."""
    image = as_bgr(frame)
    if transform is not None:
        image = cv2.warpAffine(image, np.asarray(transform, float), size)
    if image.shape[1::-1] != tuple(size):
        raise ValueError("map image must match mask size, or supply a Tab-to-map transform")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(ground_lower), tuple(ground_upper))
    mask |= cv2.inRange(hsv, tuple(road_lower), tuple(road_upper))
    # Repair tiny text/icon holes. Larger dark cliffs and grey-blue patches stay blocked.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    boundary = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(boundary, [np.asarray(polygon, np.int32)], 255)
    return (mask & boundary) != 0


class WalkGrid:
    scale = 1

    def __init__(self, mask, clearance: float = 2.0, snap_radius: int = 8) -> None:
        self.mask = np.asarray(mask, dtype=bool).copy()
        if self.mask.ndim != 2 or not self.mask.any():
            raise ValueError("walkability mask must be a nonempty 2D grid")
        if clearance < 0 or snap_radius < 0:
            raise ValueError("clearance and snap radius cannot be negative")
        self.clearance, self.snap_radius = clearance, snap_radius
        self._inflate()

    def _inflate(self):
        padded = np.pad(self.mask.astype(np.uint8), 1)
        distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        self.safe = self.mask & (distances > self.clearance)

    @classmethod
    def load(cls, path, override=None, **kwargs):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot read mask: {path}")
        mask = image >= 128
        if override is not None:
            paint = cv2.imread(str(override), cv2.IMREAD_GRAYSCALE)
            if paint is None or paint.shape != mask.shape or not np.isin(paint, (0, 128, 255)).all():
                raise ValueError("override must match the mask and contain only 0, 128 or 255")
            mask[paint == 0] = False
            mask[paint == 255] = True
        return cls(mask, **kwargs)

    def save(self, path):
        if not cv2.imwrite(str(path), self.mask.astype(np.uint8) * 255):
            raise OSError(f"could not write mask: {path}")

    def copy_with(self, clearance, snap_radius):
        return WalkGrid(self.mask, clearance, snap_radius)

    def is_safe(self, point) -> bool:
        x, y = round(point[0]), round(point[1])
        return 0 <= y < self.safe.shape[0] and 0 <= x < self.safe.shape[1] and bool(self.safe[y, x])

    def snap(self, point) -> tuple[int, int]:
        if not all(math.isfinite(v) for v in point):
            raise ValueError("map coordinates must be finite")
        x, y = round(point[0]), round(point[1])
        if not (0 <= x < self.safe.shape[1] and 0 <= y < self.safe.shape[0]):
            raise ValueError("map point outside the zone mask")
        if self.safe[y, x]:
            return x, y
        r = self.snap_radius
        ys, xs = np.nonzero(self.safe[max(0, y-r):y+r+1, max(0, x-r):x+r+1])
        xs, ys = xs + max(0, x-r), ys + max(0, y-r)
        distances = (xs-x)**2 + (ys-y)**2
        if not len(distances) or distances.min() > r*r:
            raise ValueError("no clear cell near map point")
        i = distances.argmin()
        return int(xs[i]), int(ys[i])

    def line_clear(self, a, b) -> bool:
        steps = max(1, math.ceil(max(abs(a[0]-b[0]), abs(a[1]-b[1])) * 2))
        previous = None
        for t in np.linspace(0, 1, steps + 1):
            p = (round(a[0] + (b[0]-a[0])*t), round(a[1] + (b[1]-a[1])*t))
            if not self.is_safe(p):
                return False
            if previous and p[0] != previous[0] and p[1] != previous[1]:
                if not self.is_safe((p[0], previous[1])) or not self.is_safe((previous[0], p[1])):
                    return False
            previous = p
        return True

    def block(self, point, radius: int = 3):
        cv2.circle(self.mask.view(np.uint8), (round(point[0]), round(point[1])), radius, 0, -1)
        self._inflate()

    def plan(self, start, goal) -> list[Point]:
        """Eight-neighbour A*, no diagonal corner cutting, then visible-waypoint simplification."""
        start, goal = self.snap(start), self.snap(goal)
        if start == goal:
            return [Point(*start)]
        queue = [(math.dist(start, goal), 0.0, start)]
        costs, parents = {start: 0.0}, {}
        while queue:
            _, cost, point = heapq.heappop(queue)
            if cost > costs[point]:
                continue
            if point == goal:
                break
            x, y = point
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                q = (x+dx, y+dy)
                if not self.is_safe(q):
                    continue
                if dx and dy and (not self.is_safe((x+dx, y)) or not self.is_safe((x, y+dy))):
                    continue
                new = cost + math.hypot(dx, dy)
                if new < costs.get(q, math.inf):
                    costs[q], parents[q] = new, point
                    heapq.heappush(queue, (new + math.dist(q, goal), new, q))
        else:
            raise ValueError("no route through the walkability mask")
        route = [goal]
        while route[-1] != start:
            route.append(parents[route[-1]])
        route.reverse()
        simplified, i = [Point(*start)], 0
        while i < len(route)-1:
            j = len(route)-1
            while j > i+1 and not self.line_clear(route[i], route[j]):
                j -= 1
            simplified.append(Point(*route[j]))
            i = j
        return simplified


class ScaledWalkGrid:
    """High-resolution storage, with every public point/distance in minimap pixels."""
    def __init__(self, mask, scale: int, clearance=2.0, snap_radius=8):
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("planning scale must be a positive integer")
        self.scale, self.clearance, self.snap_radius = scale, clearance, snap_radius
        self.native = WalkGrid(mask, clearance*scale, round(snap_radius*scale))

    @property
    def mask(self):
        return self.native.mask

    @property
    def safe(self):
        return self.native.safe

    def save(self, path):
        self.native.save(path)

    def _inflate(self):
        self.native._inflate()

    @classmethod
    def load(cls, path, override, scale, **kwargs):
        mask = WalkGrid.load(path, clearance=0).mask
        paint = cv2.imread(str(override), cv2.IMREAD_GRAYSCALE)
        expected = (mask.shape[0]//scale, mask.shape[1]//scale)
        if paint is None or paint.shape != expected or not np.isin(paint, (0, 128, 255)).all():
            raise ValueError("override must match map dimensions and contain only 0, 128 or 255")
        paint = cv2.resize(paint, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask[paint == 0], mask[paint == 255] = False, True
        return cls(mask, scale, **kwargs)

    def copy_with(self, clearance, snap_radius):
        return ScaledWalkGrid(self.mask, self.scale, clearance, snap_radius)

    def _point(self, point):
        return Point(point[0]*self.scale, point[1]*self.scale)

    def is_safe(self, point):
        return self.native.is_safe(self._point(point))

    def snap(self, point):
        x, y = self.native.snap(self._point(point))
        return x/self.scale, y/self.scale

    def line_clear(self, a, b):
        return self.native.line_clear(self._point(a), self._point(b))

    def block(self, point, radius=3):
        self.native.block(self._point(point), math.ceil(radius*self.scale))

    def plan(self, start, goal):
        return [Point(x/self.scale, y/self.scale)
                for x, y in self.native.plan(self._point(start), self._point(goal))]


@dataclass
class Zone:
    name: str
    folder: Path
    metadata: dict
    grid: WalkGrid
    evidence: LearnedEvidence | None = None
    base_grid: WalkGrid | None = None
    road: np.ndarray | None = None

    @classmethod
    def load(cls, name: str, config: NavigationConfig | None = None):
        cfg = config or NavigationConfig()
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("zone name must contain only lowercase letters, digits and underscores")
        root = Path(cfg.assets_folder)
        if not root.is_absolute() and cfg.assets_folder == "assets/zones":
            root = Path(__file__).resolve().parent.parent / root
        folder = root / name
        with (folder / "zone.json").open(encoding="utf-8") as f:
            metadata = json.load(f)
        clearance = cfg.clearance if cfg.clearance is not None else metadata.get("clearance", 2.0)
        scale = metadata.get("planning_scale", 1)
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("planning scale must be a positive integer")
        if scale == 1:
            grid = WalkGrid.load(folder / "mask.png", folder / "override.png", clearance=clearance,
                                 snap_radius=cfg.snap_radius)
        else:
            grid = ScaledWalkGrid.load(folder / metadata["planning_mask"], folder / "override.png", scale,
                                       clearance=clearance, snap_radius=cfg.snap_radius)
        x0, y0, x1, y1 = cfg.minimap_rect
        if grid.mask.shape != ((y1-y0)*scale, (x1-x0)*scale):
            raise ValueError("zone mask and minimap crop must have the same dimensions")
        evidence = LearnedEvidence(folder / "learned.json", name, (x1-x0, y1-y0), scale)
        base = grid.copy_with(clearance, cfg.snap_radius)
        evidence.apply(grid)
        road = None
        if (folder / "road.png").exists():
            image = cv2.imread(str(folder / "road.png"), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (y1-y0, x1-x0):
                raise ValueError("road mask must match the minimap crop")
            road = cv2.resize(image, (grid.mask.shape[1], grid.mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST) != 0
        return cls(name, folder, metadata, grid, evidence, base, road)

    def goal(self, target) -> Point:
        if isinstance(target, str):
            if target.startswith("portal:"):
                portal = target.split(":", 1)[1].upper()
                try:
                    return Point(*self.metadata["portals"][portal]["approach"])
                except KeyError as e:
                    raise ValueError(f"unknown portal: {portal}") from e
            try:
                target = [float(v) for v in target.split(",")]
            except ValueError as e:
                raise ValueError("goal must be portal:NE or map x,y") from e
        if len(target) != 2 or not all(math.isfinite(v) for v in target):
            raise ValueError("goal must contain two finite map coordinates")
        return Point(*target)

    def build_mask(self, frame, kind="minimap"):
        cfg = self.metadata
        if cfg.get("mask_generator"):
            raise ValueError("use tools/zone/tabmap_mask.py with full-resolution Tab captures and live fixes")
        image = as_bgr(frame)
        if kind == "minimap":
            x0, y0, x1, y1 = cfg["minimap_rect"]
            image = image[y0:y1, x0:x1]
        elif kind not in ("tab", "map"):
            raise ValueError("map kind must be minimap, tab or map")
        return build_walkability(image, cfg["terrain_polygon"], tuple(cfg["size"]),
                                 cfg["tab_to_map"] if kind == "tab" else None,
                                 **cfg.get("colors", {}))


def screen_click(delta, config: NavigationConfig) -> Point | None:
    """Project a map delta into the HUD-free screen rectangle, preserving direction."""
    matrix = np.asarray(config.minimap_to_screen, float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("minimap_to_screen must be a finite, nonsingular 2x2 matrix")
    vector = matrix @ np.asarray(delta, float)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-9:
        return None
    direction = vector / length
    player = np.asarray(config.player_screen, float)
    x0, y0, x1, y1 = config.safe_screen_rect
    if not (0 <= x0 <= player[0] <= x1 < config.window_size[0]
            and 0 <= y0 <= player[1] <= y1 < config.window_size[1]
            and 0 < config.click_min <= config.click_max):
        raise ValueError("invalid click limits or safe_screen_rect")
    limit = config.click_max
    for p, d, lo, hi in zip(player, direction, (x0, y0), (x1, y1)):
        if d > 0:
            limit = min(limit, (hi-p)/d)
        elif d < 0:
            limit = min(limit, (lo-p)/d)
    if limit < config.click_min:
        return None
    distance = min(limit, max(config.click_min, length))
    return Point(*(player + direction * distance))


@dataclass(frozen=True)
class NavigationStep:
    status: str
    location: Location | None = None
    click: Point | None = None
    waypoints: tuple[Point, ...] = ()
    message: str = ""


class NavigateTo:
    """Task API: call tick(frame, monotonic_time); dispatch its proposals elsewhere.

    It never controls the game, presses Tab, switches zones, sells, or banks.
    A caller must apply focus/health checks before dispatch and call tick again
    on fresh captures. Uncalibrated transforms produce no clicks unless explicitly
    allowed for a dry-run preview.
    """
    def __init__(self, goal, zone: Zone | None = None, config: NavigationConfig | None = None,
                 allow_uncalibrated: bool = False, learn: bool = False):
        self.config = config or NavigationConfig()
        if (self.config.lookahead <= 0 or self.config.arrival_tolerance <= 0
                or self.config.waypoint_tolerance <= 0 or self.config.stuck_seconds <= 0
                or self.config.side_step_seconds < 0 or self.config.progress_pixels <= 0
                or self.config.map_speed_px_per_s <= 0 or self.config.stuck_block_radius < 1
                or self.config.max_replans < 0 or not math.isfinite(self.config.learned_edge_length)
                or self.config.learned_edge_length <= 0):
            raise ValueError("invalid navigation timing, distances or recovery limits")
        self.zone = zone or Zone.load(self.config.zone, self.config)
        self.goal = self.zone.goal(goal)
        size = (self.zone.grid.mask.shape[1]//self.zone.grid.scale,
                self.zone.grid.mask.shape[0]//self.zone.grid.scale)
        self.evidence = (self.zone.evidence or LearnedEvidence(None, self.zone.name, size,
                                                              self.zone.grid.scale)).copy(persist=learn)
        self.hazards = []
        self.grid = self._fresh_grid()
        self.locator = MapLocator(self.config, self.zone.metadata["terrain_polygon"])
        self.allow_uncalibrated = allow_uncalibrated
        self.route: list[Point] = []
        self.index = 1
        self.progress_at = None
        self.replans = 0
        self.stalls = []
        self.motion_anchor = None
        self.failed = False
        self.failed_point = None
        self.last_tick = None
        self.no_route_reports = {}

    def _fresh_grid(self, edges=None):
        base = self.zone.base_grid or self.zone.grid
        grid = base.copy_with(self.config.clearance if self.config.clearance is not None
                             else self.zone.grid.clearance, self.config.snap_radius)
        self.evidence.apply(grid, edges=edges)
        for point, radius in self.hazards:
            grid.block(point, radius)
        return grid

    def block_hazard(self, point, radius):
        # Standing is evidence of walkability, not evidence that an attacker
        # left. Keep retreat's temporary danger areas separate from terrain.
        self.hazards.append((point, radius))
        self.grid.block(point, radius)

    def record_position(self, location):
        """Dispatcher calls only for fixes obtained while real travel was moving."""
        if location is not None and self.evidence.observe(location.point):
            self.no_route_reports.clear()
            was_safe = self.grid.is_safe(location.point)
            self.grid = self._fresh_grid()
            if not was_safe:
                self.route = []

    def _plan(self, start, now):
        try:
            self.route = self.grid.plan(start, self.goal)
        except ValueError as error:
            message = self._learned_no_route(start)
            if message is None:
                raise
            if message:
                raise ValueError(message) from error
            self.route = self.grid.plan(start, self.goal)
        self.index = 1
        self.progress_at = now
        self.motion_anchor = Point(*start)

    @staticmethod
    def _connected(grid, start, goal):
        # This is the same connectivity as A* without diagonal corner cutting,
        # but avoids a full failed search for each diagnostic edge subset.
        try:
            a, b = grid.snap(start), grid.snap(goal)
        except ValueError:
            return False
        _, labels = cv2.connectedComponents(grid.safe.astype(np.uint8), connectivity=4)
        x, y = [round(v*grid.scale) for v in a]
        gx, gy = [round(v*grid.scale) for v in b]
        return bool(labels[y, x] and labels[y, x] == labels[gy, gx])

    def _learned_no_route(self, start):
        if not self.evidence.edges:
            return None
        key = (tuple(round(v*self.grid.scale) for v in start), len(self.evidence.edges))
        if key in self.no_route_reports:
            return self.no_route_reports[key]
        if not self._connected(self._fresh_grid(edges=[]), start, self.goal):
            self.no_route_reports[key] = None
            return None  # the base mask or a temporary combat hazard already prevents this route
        opened = 0
        if self.zone.road is not None:
            near = cv2.dilate(self.zone.road.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (2*self.grid.scale+1, 2*self.grid.scale+1)))
            for point in self.evidence.positions.values():
                x, y = np.rint(np.asarray(point)*self.grid.scale).astype(int)
                if near[y, x]:
                    opened += open_observed_disc(self.grid.mask, point, self.grid.scale)
        if opened:
            self.grid._inflate()
            for point, radius in self.hazards:
                self.grid.block(point, radius)
        # These discs are normally already open; explicitly try them before
        # diagnosing the cut. Never invent a corridor between sampled positions.
        if self._connected(self.grid, start, self.goal):
            log.warning("navigation: recovered learned-edge route with %d observed road-disc cells", opened)
            return ""
        removed = []
        for i in range(len(self.evidence.edges)):
            removed.append(i)
            keep = [e for j, e in enumerate(self.evidence.edges) if j not in removed]
            if self._connected(self._fresh_grid(edges=keep), start, self.goal):
                break
        # Minimise the explanatory cut set. These are diagnostic copies only;
        # no edge is deleted from the task or its persisted evidence.
        for i in removed[:]:
            trial = [j for j in removed if j != i]
            keep = [e for j, e in enumerate(self.evidence.edges) if j not in trial]
            if self._connected(self._fresh_grid(edges=keep), start, self.goal):
                removed = trial
        message = (f"no route: learned edge indices {removed} disconnect goal; retained all edges; "
                   f"observed road discs opened {opened} cells")
        log.warning("navigation: %s", message)
        self.no_route_reports[key] = message
        return message

    def _clear_from(self, point, target) -> bool:
        """Line check from the player's exact spot or from its grid cell.

        Routes are planned from the rounded cell, so a sub-pixel start like
        (230.05, 121.49) could clip a blocked corner that the planned segment
        from (230, 121) avoids. Seen live on 2026-11-04: every tick then
        reported "no clear segment" and the walk never started.
        """
        scale = self.grid.scale
        cell = Point(round(point.x*scale)/scale, round(point.y*scale)/scale)
        return self.grid.line_clear(point, target) or self.grid.line_clear(cell, target)

    def tick(self, frame, now: float, moving: bool = False) -> NavigationStep:
        if self.last_tick is not None and now < self.last_tick:
            raise ValueError("navigation timestamps must increase")
        gap = now-self.last_tick if self.last_tick is not None else 0
        self.last_tick = now
        location = self.locator.locate(frame, now)
        if location is None:
            # Do not count time behind an overlay as walking without progress.
            self.progress_at = None
            return NavigationStep("lost", message="minimap arrow missing or ambiguous; no click")
        point = location.point
        if moving:
            self.record_position(location)
        if self.failed_point is not None and point.dist(self.failed_point) > self.config.stuck_block_radius:
            self.failed, self.failed_point = False, None
        if point.dist(self.goal) <= self.config.arrival_tolerance:
            return NavigationStep("arrived", location, waypoints=tuple(self.route))
        if not self.grid.is_safe(point) and self._fresh_grid().is_safe(point):
            # The player is standing on ground that only this task's stuck blocks
            # closed (live 2026-11-04, run 5): those blocks were wrong. Drop them.
            log.info("navigation: player inside a private stuck block at %.1f,%.1f; clearing blocks",
                     point.x, point.y)
            self.grid, self.route = self._fresh_grid(), []
        if not self.grid.is_safe(point):
            # Seen live on 2026-11-04: the character drifted a pixel off a thin
            # corridor and every later tick returned no_route. Walk back to the
            # nearest clear cell instead, then plan a fresh route from there.
            # Only for small drifts: a far "nearest" cell could be across a wall.
            try:
                cell = self.grid.snap(point)
            except ValueError:
                cell = None
            if cell is None or point.dist(Point(*cell)) > self.grid.clearance + 2:
                return NavigationStep("no_route", location, message="player outside clear mask; calibrate mask/clearance")
            self.route = []
            if not self.config.calibrated and not self.allow_uncalibrated:
                return NavigationStep("uncalibrated", location)
            click = screen_click(Point(cell[0]-point.x, cell[1]-point.y), self.config)
            return NavigationStep("move" if click is not None else "unsafe_click", location, click,
                                  message="outside clear mask; returning to nearest clear cell")
        if self.failed:
            return NavigationStep("stalled", location, message="replan limit reached; no click")
        if self.progress_at is None or gap > 2:
            self.progress_at = now
            self.motion_anchor = point
        try:
            if not self.route:
                self._plan(point, now)
            while self.index < len(self.route) and point.dist(self.route[self.index]) <= self.config.waypoint_tolerance:
                self.index += 1
                self.progress_at = now
            if self.index >= len(self.route):
                return NavigationStep("no_route", location, waypoints=tuple(self.route),
                                      message="snapped route ends outside arrival tolerance; adjust goal/mask")
            if not self.config.calibrated and not self.allow_uncalibrated:
                return NavigationStep("uncalibrated", location, waypoints=tuple(self.route))
            if not self._clear_from(point, self.route[self.index]):
                self._plan(point, now)
            if (self.index >= len(self.route)
                    or not self._clear_from(point, self.route[self.index])):
                return NavigationStep("no_route", location, message="no clear segment from current position")
            target = self.route[self.index]
            distance = point.dist(target)
            # Click-to-move bends around rocks. Actual displacement counts as
            # progress even when it temporarily moves away from the waypoint.
            if self.motion_anchor is None or point.dist(self.motion_anchor) >= self.config.progress_pixels:
                self.motion_anchor = point
                self.progress_at = now
            dx, dy = target.x-point.x, target.y-point.y
            scale = min(1.0, self.config.lookahead / max(distance, 1e-9))
            delta = Point(dx*scale, dy*scale)
            if now-self.progress_at >= self.config.stuck_seconds:
                attempts = sum(point.dist(old) <= self.config.stuck_block_radius for old in self.stalls)
                if attempts >= self.config.max_replans:
                    self.failed = True
                    self.failed_point = point
                    return NavigationStep("stalled", location, waypoints=tuple(self.route))
                self.stalls.append(point)
                self.replans += 1
                blocked = self.evidence.stall(point, target, self.grid.clearance, self.config.learned_edge_length,
                                              now, road=self.zone.road)
                self.grid = self._fresh_grid()
                log.warning("navigation: stall at %.2f,%.2f toward %.2f,%.2f; edge=%s; replan %d at spot",
                            point.x, point.y, target.x, target.y, blocked, attempts+1)
                self._plan(point, now)
                # Return without a side-step click; next fresh tick follows the
                # alternative route through the normal guarded dispatch path.
                return NavigationStep("waiting", location, waypoints=tuple(self.route),
                                      message=self.evidence.stall_message)
            click = screen_click(delta, self.config)
            return NavigationStep("move" if click is not None else "unsafe_click", location,
                                  click, tuple(self.route))
        except ValueError as e:
            return NavigationStep("no_route", location, waypoints=tuple(self.route), message=str(e))


def solve_calibration(samples) -> dict:
    """Least-squares screen_delta = matrix @ map_delta, from independent probes."""
    if len(samples) < 2:
        raise ValueError("need at least two independent movement probes")
    mapped = np.asarray([s["map_delta"] for s in samples], float)
    screen = np.asarray([s["screen_delta"] for s in samples], float)
    if (mapped.shape != (len(samples), 2) or screen.shape != mapped.shape
            or not np.isfinite(mapped).all() or not np.isfinite(screen).all()
            or (np.linalg.norm(mapped, axis=1) < 1).any()):
        raise ValueError("invalid or too-small movement probe")
    if np.linalg.matrix_rank(mapped) < 2 or np.linalg.cond(mapped) > 20:
        raise ValueError("probe directions are collinear or poorly conditioned; use perpendicular moves")
    matrix = np.linalg.lstsq(mapped, screen, rcond=None)[0].T
    error = float(np.sqrt(np.mean((mapped @ matrix.T-screen)**2)))
    if abs(np.linalg.det(matrix)) < 1e-8 or error > 40:
        raise ValueError("inconsistent movement probes; repeat in clear ground")
    speeds = [np.linalg.norm(s["map_delta"]) / s["duration"] for s in samples if s.get("duration", 0) > 0]
    return {"minimap_to_screen": matrix.tolist(), "rms_screen_error": error,
            "map_speed_px_per_s": float(np.median(speeds)) if speeds else None,
            "calibrated": True}
