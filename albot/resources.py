"""Sourced zone resource types and failure history in approximate world pixels."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from .geometry import Point

log = logging.getLogger(__name__)


RESOURCE_TYPES = frozenset(("hide", "fiber", "ore", "wood", "stone"))
RESOURCE_ALIASES = {"cotton": "fiber", "log": "wood", "tree": "wood"}


def zone_resource_exclusions(navigation):
    """A trusted T2+ list permits filtering; any T2+ positive vetoes exclusion."""
    from .navigation import Zone

    path = (Zone.asset_folder(navigation.zone, navigation).parent / "resources.json").resolve()
    return _zone_resource_exclusions(path, navigation.zone)


@lru_cache(maxsize=16)
def _zone_resource_exclusions(path, zone):
    # Catalogue data is fixed for a run. Never load navigation masks/learning or
    # fetch the network in a scene worker. Restart after a catalogue refresh.
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    if data.get("version") != 1:
        raise ValueError("unsupported zone resource catalogue version")
    observations = data.get("zones", {}).get(zone, {}).get("observations", [])
    positive, excluded, complete = set(), set(), False
    for observation in observations:
        resources = observation.get("resources")
        if resources is None:
            continue  # unknown is not an empty resource list
        if any(resource["type"] not in RESOURCE_TYPES for resource in resources):
            raise ValueError("unknown zone resource type")
        types = {resource["type"] for resource in resources
                 if not observation.get("t1_ambient") and not resource.get("t1_ambient")
                 and (resource.get("tiers") is None or resource["tiers"][1] >= 2)}
        positive.update(types)  # advisory evidence may protect a type, never create a negative
        source = observation.get("source", "")
        trusted = source in ("ingame_tab", "user") or source.startswith("user (")
        if not trusted or observation.get("t1_ambient"):
            continue
        complete |= observation.get("complete_t2plus", False)
        absent = set(observation.get("absent_t2plus", []))
        if not absent <= RESOURCE_TYPES:
            raise ValueError("unknown absent zone resource type")
        excluded.update(absent)
    if complete:
        excluded.update(RESOURCE_TYPES-positive)
    return frozenset(excluded-positive)


def resource_type(class_name):
    return RESOURCE_ALIASES.get(class_name, class_name)


@dataclass
class FailedResource:
    class_name: str
    point: Point
    until: float
    failures: int


class ResourceMemory:
    def __init__(self, config, clock) -> None:
        self.b = config.behavior
        self.clock = clock
        self.durations = (self.b.resource_blacklist_durations if self.b.resource_blacklist_durations is not None
                          else [self.b.resource_blacklist_seconds, 600.0, 1800.0])
        if not self.durations or any(not 0 < duration < float("inf") for duration in self.durations):
            raise ValueError("resource_blacklist_durations must contain positive finite seconds")
        self.clear()

    def clear(self) -> None:
        self.moves = [(self.clock.now(), Point(0, 0), Point(0, 0), 0.0)]
        self.failed: list[FailedResource] = []

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
        # range and speed buffs all cause drift. We cap travel at the clicked spot
        # and clear on death or an unmeasured dash. Longer failure histories also
        # retain this drift; radius matches can merge two nearby same-class nodes.
        self.moves.append((now, origin, delta, duration))
        cutoff = now - self.b.resource_blacklist_seconds
        while len(self.moves) > 1 and self.moves[1][0] < cutoff:
            self.moves.pop(0)

    def world_point(self, scene, point: Point) -> Point:
        origin = self.position(scene.timestamp)
        return Point(origin.x + point.x - scene.player.x, origin.y + point.y - scene.player.y)

    def failure(self, class_name: str, point: Point) -> None:
        nearby = [entry for entry in self.failed if entry.class_name == class_name
                  and entry.point.dist(point) <= self.b.resource_blacklist_radius]
        entry = min(nearby, key=lambda entry: entry.point.dist(point)) if nearby else None
        if entry is None:
            entry = FailedResource(class_name, point, 0, 0)
            self.failed.append(entry)
        entry.failures += 1
        duration = self.durations[min(entry.failures-1, len(self.durations)-1)]
        entry.until = self.clock.now() + duration
        log.info("resource %s failure %d at (%.0f,%.0f): blacklisted for %.0fs",
                 class_name, entry.failures, point.x, point.y, duration)

    def success(self, class_name: str, point: Point) -> None:
        self.failed = [entry for entry in self.failed if entry.class_name != class_name
                       or entry.point.dist(point) > self.b.resource_blacklist_radius]

    def blocked(self, scene, target) -> bool:
        now = self.clock.now()
        point = self.world_point(scene, target.center)
        # Expiry permits another attempt, but keeps its count until success or
        # coordinate reset. Otherwise each three-minute retry would be failure #1.
        return any(entry.until > now and entry.class_name == target.class_name
                   and entry.point.dist(point) <= self.b.resource_blacklist_radius for entry in self.failed)
