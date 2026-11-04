"""
2D geometry for line-of-path checks, in window pixel coordinates.

Coordinates are window-relative everywhere in the bot. They become screen
coordinates only when the input layer clicks, so a moved window can't make
stored positions wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, NamedTuple, Sequence


class Point(NamedTuple):
    x: float
    y: float

    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def _orientation(p: Point, q: Point, r: Point) -> int:
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.y, r.y) <= q.y <= max(p.y, r.y)


def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


@dataclass(frozen=True)
class Polygon:
    points: tuple[Point, ...]
    confidence: float = 1.0

    @cached_property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    @cached_property
    def centroid(self) -> Point:
        n = len(self.points)
        return Point(sum(p.x for p in self.points) / n, sum(p.y for p in self.points) / n)

    def edges(self) -> Iterable[tuple[Point, Point]]:
        n = len(self.points)
        if n < 2:
            return
        # includes the closing edge from the last point back to the first
        for i in range(n):
            yield self.points[i], self.points[(i + 1) % n]

    def contains(self, p: Point) -> bool:
        # ray casting
        inside = False
        for a, b in self.edges():
            if (a.y > p.y) != (b.y > p.y):
                x_cross = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
                if p.x < x_cross:
                    inside = not inside
        return inside

    def blocks(self, a: Point, b: Point) -> bool:
        # cheap reject first: segment's box doesn't overlap the polygon's box
        x0, y0, x1, y1 = self.bbox
        if max(a.x, b.x) < x0 or min(a.x, b.x) > x1 or max(a.y, b.y) < y0 or min(a.y, b.y) > y1:
            return False
        return any(segments_intersect(a, b, p, q) for p, q in self.edges())


def path_clear(a: Point, b: Point, obstacles: Sequence[Polygon]) -> bool:
    return not any(o.blocks(a, b) for o in obstacles)


def convex_hull(points: Sequence[Point]) -> list[Point]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def corner_candidates(obstacle: Polygon, margin: float, max_corners: int = 12) -> list[Point]:
    """
    Points just outside an obstacle's corners, for paths to round.

    Segmentation outlines can have dozens of points. Taking the convex hull and
    keeping its sharpest corners keeps the search small, and pushing each one out
    by `margin` keeps paths off the edges.
    """
    hull = convex_hull(obstacle.points)
    n = len(hull)
    if n == 0:
        return []
    if n > max_corners:
        def sharpness(i: int) -> float:
            a, b, c = hull[i - 1], hull[i], hull[(i + 1) % n]
            v1 = (a.x - b.x, a.y - b.y)
            v2 = (c.x - b.x, c.y - b.y)
            l1, l2 = math.hypot(*v1), math.hypot(*v2)
            if l1 == 0 or l2 == 0:
                return 0.0
            # smaller interior angle = sharper corner
            return -((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2))
        keep = sorted(sorted(range(n), key=sharpness)[:max_corners])
        hull = [hull[i] for i in keep]
    c = obstacle.centroid
    out = []
    for v in hull:
        d = v.dist(c)
        if d > 0:
            out.append(Point(v.x + (v.x - c.x) / d * margin, v.y + (v.y - c.y) / d * margin))
    return out


def find_path(start: Point, goal: Point, obstacles: Sequence[Polygon], margin: float,
              bounds: tuple[float, float] | None = None, max_ms: float = 60.0) -> list[Point] | None:
    """
    Shortest path from start to goal through obstacle corners: A* over a
    visibility graph. Returns the waypoints between start and goal (empty if the
    straight line is clear), or None if there is no path or the search runs past
    max_ms.

    Edges between two corners are checked against each obstacle's convex hull,
    which has far fewer edges than a segmentation outline. The hull contains the
    outline, so clearing the hull guarantees clearing the outline. Edges touching
    the start or goal use the real outline, since either may sit in a concave
    pocket of a hull.
    """
    if path_clear(start, goal, obstacles):
        return []
    import heapq
    import time

    deadline = time.perf_counter() + max_ms / 1000
    hulls = [Polygon(tuple(convex_hull(o.points)), o.confidence) for o in obstacles]
    nodes = [start, goal]
    for o in obstacles:
        for w in corner_candidates(o, margin):
            if bounds is not None and not (0 <= w.x <= bounds[0] and 0 <= w.y <= bounds[1]):
                continue
            if any(h.contains(w) for h in hulls):
                continue
            nodes.append(w)

    def clear(i: int, j: int) -> bool:
        if i <= 1 or j <= 1:
            return path_clear(nodes[i], nodes[j], obstacles)
        return path_clear(nodes[i], nodes[j], hulls)

    g = {0: 0.0}
    prev: dict[int, int] = {}
    done: set[int] = set()
    heap = [(start.dist(goal), 0)]
    while heap:
        if time.perf_counter() > deadline:
            return None
        _, i = heapq.heappop(heap)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            path = []
            while i != 0:
                path.append(nodes[i])
                i = prev[i]
            path.reverse()
            return path[:-1]   # drop the goal itself
        for j in range(len(nodes)):
            if j in done:
                continue
            ng = g[i] + nodes[i].dist(nodes[j])
            if ng >= g.get(j, math.inf):
                continue
            if clear(i, j):
                g[j] = ng
                prev[j] = i
                heapq.heappush(heap, (ng + nodes[j].dist(goal), j))
    return None


def find_detour(start: Point, goal: Point, obstacles: Sequence[Polygon], margin: float,
                bounds: tuple[float, float] | None = None) -> Point | None:
    """
    The first waypoint of the shortest path around obstacles, or None if the
    straight line is clear or no path exists. The bot walks to it and re-plans
    from a fresh screenshot, since the camera follows the player and every
    screen position shifts as it walks.
    """
    path = find_path(start, goal, obstacles, margin, bounds)
    return path[0] if path else None
