"""Short-lived resource memory in an approximate, camera-independent pixel space."""
from __future__ import annotations

from .geometry import Point


class ResourceMemory:
    def __init__(self, config, clock) -> None:
        self.b = config.behavior
        self.clock = clock
        self.clear()

    def clear(self) -> None:
        self.moves = [(self.clock.now(), Point(0, 0), Point(0, 0), 0.0)]
        self.failed: list[tuple[str, Point, float]] = []

    def position(self, timestamp: float) -> Point:
        # Use the camera position at capture time, not at the end of inference.
        start, origin, delta, duration = self.moves[0]
        for move in reversed(self.moves):
            if move[0] <= timestamp:
                start, origin, delta, duration = move
                break
        fraction = min(1.0, max(0.0, timestamp - start) / duration) if duration else 0.0
        return Point(origin.x + delta.x * fraction, origin.y + delta.y * fraction)

    def move(self, scene, target: Point | None) -> None:
        now = self.clock.now()
        origin = self.position(now)
        delta = Point(target.x - scene.player.x, target.y - scene.player.y) if target else Point(0, 0)
        duration = delta.dist(Point(0, 0)) / self.b.walk_speed_px_per_s
        # Intent is not measured motion: collisions, latency, auto-pathing, attack
        # range and speed buffs all cause drift. We cap travel at the clicked spot,
        # expire entries after three minutes, and clear on death or an unmeasured dash.
        self.moves.append((now, origin, delta, duration))
        cutoff = now - self.b.resource_blacklist_seconds
        while len(self.moves) > 1 and self.moves[1][0] < cutoff:
            self.moves.pop(0)

    def world_point(self, scene, point: Point) -> Point:
        origin = self.position(scene.timestamp)
        return Point(origin.x + point.x - scene.player.x, origin.y + point.y - scene.player.y)

    def failure(self, class_name: str, point: Point) -> None:
        self.failed.append((class_name, point, self.clock.now() + self.b.resource_blacklist_seconds))

    def blocked(self, scene, target) -> bool:
        now = self.clock.now()
        self.failed = [(name, p, until) for name, p, until in self.failed if until > now]
        point = self.world_point(scene, target.center)
        return any(name == target.class_name and p.dist(point) <= self.b.resource_blacklist_radius
                   for name, p, _ in self.failed)
