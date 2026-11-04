"""
Hold-and-steer travel: hold the move button with the Tab zone map open.

Tyler's tip (2026-11-04): while the move button is held, the character keeps
walking toward the cursor, and the Tab map shows the player arrow at about 4.5x
the minimap's resolution. Steering a held button replaces click-to-move, whose
stop-start walking and 1 px (~370 screen px) minimap fixes produced false
stalls and learned walls across the portal road (runs 15 to 18).

Two things learned live the same day:
- The game ignores SetCursorPos during a hold. Only relative mouse input steers.
- Props smaller than a map pixel (the broken wagon, sandstone rocks) still block.
  A stall turns 60 degrees to the side that lands on road, then the other side,
  then 100 degrees each way, and resumes the route after each detour.

Nothing here learns evidence: a hold run never writes learned.json.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import Point

log = logging.getLogger(__name__)

TAB_REGION = (300, 250, 1950, 1300)  # Tab map area at 2240x1400
ARROW_HSV = ((95, 120, 150), (125, 255, 255))
ARROW_AREA = (80, 260)  # pixels at 2240x1400


def tab_arrow(frame, previous=None, static_icons=(), size=(2240, 1400)):
    """Window position of the blue player arrow on the open Tab map, or None.

    Fixed blue icons (Fractured Ground's city marker) are excluded by position.
    With several candidates and no history the fix is ambiguous and refused.
    """
    height, width = frame.shape[:2]
    sx, sy = width/size[0], height/size[1]
    x0, y0, x1, y1 = (round(TAB_REGION[0]*sx), round(TAB_REGION[1]*sy),
                      round(TAB_REGION[2]*sx), round(TAB_REGION[3]*sy))
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1, :3]), cv2.COLOR_BGR2HSV)
    count, _, stats, centers = cv2.connectedComponentsWithStats(cv2.inRange(hsv, *ARROW_HSV))
    area = sx*sy
    blobs = []
    for i in range(1, count):
        if not ARROW_AREA[0]*area <= stats[i, cv2.CC_STAT_AREA] <= ARROW_AREA[1]*area:
            continue
        center = np.array(((centers[i][0]+x0)/sx, (centers[i][1]+y0)/sy))
        if all(np.linalg.norm(center-icon) > 6 for icon in static_icons):
            blobs.append(center)
    if not blobs:
        return None
    if previous is not None:
        return Point(*min(blobs, key=lambda c: np.linalg.norm(c-np.asarray(previous))))
    return Point(*blobs[0]) if len(blobs) == 1 else None


def tab_to_map(transform, point):
    matrix = np.asarray(transform, float)
    return Point(*(matrix[:, :2]@np.asarray(point, float)+matrix[:, 2]))


def map_to_tab(transform, point):
    matrix = np.asarray(transform, float)
    return Point(*np.linalg.solve(matrix[:, :2], np.asarray(point, float)-matrix[:, 2]))


def route_progress(route, point):
    """Arc length along the route to the point closest to `point`."""
    best, travelled, best_progress = math.inf, 0.0, 0.0
    for a, b in zip(route, route[1:]):
        ab = np.subtract(b, a)
        length = float(np.linalg.norm(ab))
        t = float(np.clip(np.subtract(point, a)@ab/max(length*length, 1e-9), 0, 1))
        d = math.dist(point, np.add(a, t*ab))
        if d < best:
            best, best_progress = d, travelled+t*length
        travelled += length
    return best_progress


def route_ahead(route, point, distance):
    """The point `distance` map px along the route from the closest point to `point`."""
    if len(route) < 2:
        return Point(*route[-1])
    best = (math.inf, 0, 0.0)
    for i, (a, b) in enumerate(zip(route, route[1:])):
        ab = np.subtract(b, a)
        t = float(np.clip(np.subtract(point, a)@ab/max(float(ab@ab), 1e-9), 0, 1))
        d = math.dist(point, np.add(a, t*ab))
        if d < best[0]:
            best = (d, i, t)
    _, i, t = best
    position = np.add(route[i], t*np.subtract(route[i+1], route[i])).astype(float)
    left = distance
    for j in range(i, len(route)-1):
        end = np.asarray(route[j+1], float)
        segment = float(np.linalg.norm(end-position))
        if segment >= left:
            return Point(*(position+(end-position)*left/max(segment, 1e-9)))
        left -= segment
        position = end
    return Point(*route[-1])


def rotate(vector, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array((c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]))


@dataclass
class HoldConfig:
    lookahead: float = 2.5        # map px along the route to steer at
    mouse_pixels: float = 350.0   # cursor distance from the player
    arrival: float = 1.5          # map px
    stall_seconds: float = 1.2
    stall_pixels: float = 0.5
    detour_seconds: float = 1.0
    detours: int = 4              # per spot, then give up
    detour_reset: float = 2.0     # map px further along the route than the first stall of a spot


@dataclass
class SteerStep:
    status: str                   # steer, detour, arrived, stalled
    cursor: Point | None = None   # window coordinates
    message: str = ""


class HoldSteer:
    """Pure steering state: a map fix in, a cursor position out."""

    def __init__(self, route, goal, matrix, player_screen, config=None, landing_score=None):
        self.route = [Point(*p) for p in route]
        self.goal = Point(*goal)
        self.matrix = np.asarray(matrix, float)
        self.player_screen = np.asarray(player_screen, float)
        self.c = config or HoldConfig()
        self.landing_score = landing_score or (lambda point, direction: 0)
        self.anchor = self.anchor_at = None
        self.detour_spot, self.detour_count, self.side = None, 0, 1
        self.detour_progress = 0.0
        self.detour_until, self.detour_direction = -math.inf, None

    def cursor(self, direction):
        return Point(*(self.player_screen+self.c.mouse_pixels*direction))

    def update(self, point, now) -> SteerStep:
        point = Point(*point)
        if point.dist(self.goal) <= self.c.arrival:
            return SteerStep("arrived", message=f"arrived at {point.x:.1f},{point.y:.1f}")
        # Reset only on real progress along the route. Live 2026-11-04: bouncing
        # between two ledge spots reset a distance-from-stall count forever, and
        # straight-line distance to a portal grew shorter on a detour that led
        # nowhere, since the route first heads away from the portal.
        if (self.detour_spot is not None
                and route_progress(self.route, point) > self.detour_progress+self.c.detour_reset):
            self.detour_spot, self.detour_count = None, 0
        target = route_ahead(self.route, point, self.c.lookahead)
        direction = self.matrix@np.subtract(target, point)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            target, direction = self.goal, self.matrix@np.subtract(self.goal, point)
            norm = float(np.linalg.norm(direction))
        direction = direction/max(norm, 1e-9)
        if self.anchor is None or point.dist(self.anchor) >= self.c.stall_pixels:
            self.anchor, self.anchor_at = point, now
        elif now-self.anchor_at >= self.c.stall_seconds and now >= self.detour_until:
            if self.detour_count >= self.c.detours:
                return SteerStep("stalled", message=f"stalled at {point.x:.1f},{point.y:.1f} "
                                                    f"after {self.detour_count} detours")
            if self.detour_count == 0:
                self.detour_spot, self.detour_progress = point, route_progress(self.route, point)
                self.side = 1 if (self.landing_score(point, rotate(direction, 60))
                                  >= self.landing_score(point, rotate(direction, -60))) else -1
            angle = (60 if self.detour_count < 2 else 100)*(self.side if self.detour_count % 2 == 0 else -self.side)
            self.detour_direction = rotate(direction, angle)
            self.detour_until = now+self.c.detour_seconds
            self.detour_count += 1
            self.anchor, self.anchor_at = point, now+self.c.detour_seconds
            return SteerStep("detour", self.cursor(self.detour_direction),
                             f"stall at {point.x:.2f},{point.y:.2f}: detour {self.detour_count} at {angle:+d} deg")
        if now < self.detour_until:
            return SteerStep("detour", self.cursor(self.detour_direction), "detour")
        return SteerStep("steer", self.cursor(direction), f"toward {target.x:.1f},{target.y:.1f}")


def plan_route(nav, start):
    """Plan from the fix, or from the nearest clear cell that reaches the goal.

    Live 2026-11-04: beside the portal road the nearest clear cell sat in a
    disconnected scrap of mask, so "nearest" alone found no route.
    """
    start = Point(*start)
    try:
        nav._plan(start, 0)
        return [start, *nav.route]
    except ValueError:
        pass
    cells = sorted(((start.x+dx/4, start.y+dy/4) for dx in range(-16, 17) for dy in range(-16, 17)),
                   key=lambda c: math.dist(c, start))
    for cell in cells:
        if not nav.grid.is_safe(cell):
            continue
        try:
            nav._plan(Point(*cell), 0)
        except ValueError:
            continue
        return [start, *nav.route]
    raise ValueError(f"no connected clear cell within 4 map px of {start.x:.1f},{start.y:.1f}")


def road_landing_score(nav, matrix):
    """Score a detour direction by where one map px of travel would land."""
    def score(point, direction):
        landing = np.asarray(point)+np.linalg.solve(np.asarray(matrix, float), direction*370)
        x, y = [round(v*nav.grid.scale) for v in landing]
        road = nav.zone.road
        on_road = (road is not None and 0 <= y < road.shape[0] and 0 <= x < road.shape[1]
                   and bool(road[y, x]))
        return 2*on_road+nav.grid.is_safe(landing)
    return score


class Trails:
    """Paths hold travel actually walked, in live map pixels (local, gitignored).

    The planner's mask is a visual guess: on 2026-11-04 it routed a basin-to-road
    trip over a ledge, while the real passage was 3 px west. A trail that worked
    once is ground truth, so a later trip near both its ends follows it.
    """

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self.trails = []
        if self.path is not None and self.path.exists():
            self.trails = json.loads(self.path.read_text(encoding="utf-8"))["trails"]

    def add(self, points, spacing=0.4, when=""):
        thinned = []
        for p in points:
            if not thinned or math.dist(p, thinned[-1]) >= spacing:
                thinned.append([round(float(p[0]), 2), round(float(p[1]), 2)])
        if len(thinned) < 2:
            return False
        self.trails.append({"at": when, "points": thinned})
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".json.tmp")
            temp.write_text(json.dumps({"version": 1, "trails": self.trails}, indent=1)+"\n", encoding="utf-8")
            temp.replace(self.path)
        return True

    def _graph(self, link):
        nodes = [tuple(p) for trail in self.trails for p in trail["points"]]
        edges = {i: [] for i in range(len(nodes))}
        k = 0
        for trail in self.trails:
            for a in range(k, k+len(trail["points"])-1):
                edges[a].append(a+1)
                edges[a+1].append(a)
            k += len(trail["points"])
        array = np.asarray(nodes, float).reshape(-1, 2)
        for a in range(len(nodes)):
            close = np.flatnonzero(np.linalg.norm(array-array[a], axis=1) <= link)
            edges[a].extend(int(b) for b in close if b != a)
        return nodes, edges

    def route_toward(self, start, goal, near=2.5, gain=3.0, link=1.0):
        """Shortest walk over all saved trails, from near start to the trail point closest to goal.

        Trails link wherever they pass within `link` map px, so a road trail and a
        basin trail chain into one route. Needs `gain` map px of progress; the
        caller plans whatever remains past the trails.
        """
        import heapq
        nodes, edges = self._graph(link)
        distance, parent, queue = {}, {}, []
        for i, node in enumerate(nodes):
            d = math.dist(node, start)
            if d <= near:
                distance[i] = d
                heapq.heappush(queue, (d, i))
        while queue:
            d, i = heapq.heappop(queue)
            if d > distance[i]:
                continue
            for j in edges[i]:
                nd = d+math.dist(nodes[i], nodes[j])
                if nd < distance.get(j, math.inf):
                    distance[j], parent[j] = nd, i
                    heapq.heappush(queue, (nd, j))
        if not distance:
            return None
        end = min(distance, key=lambda i: (round(math.dist(nodes[i], goal), 1), distance[i]))
        if math.dist(nodes[end], goal) > math.dist(start, goal)-gain:
            return None
        path = [end]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        return [Point(*start), *(Point(*nodes[i]) for i in reversed(path))]

    def route(self, start, goal, near=2.5):
        """A route over saved trails that ends within `near` of goal, or None."""
        partial = self.route_toward(start, goal, near, gain=0.0)
        if partial is None or partial[-1].dist(Point(*goal)) > near:
            return None
        return [*partial, Point(*goal)]
