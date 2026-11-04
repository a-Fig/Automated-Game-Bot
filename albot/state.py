"""
What the bot knows about the game at one moment, and a thread-safe holder for
the newest copy.

A Scene is one screenshot run through the scene and obstacle models. A Status is
one screenshot read for HUD health, spells and sometimes the loading bar.
Both are immutable once built, so a behavior holding one can't see it change
underneath it. Behaviors that want the newest data ask for it again.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence, TypeVar

from .geometry import Point, Polygon, path_clear


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    center: Point          # window coordinates
    width: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class Scene:
    timestamp: float
    window_size: tuple[int, int]
    player: Point
    player_detected: bool
    detections: tuple[Detection, ...]   # everything the scene model returned
    mobs: tuple[Detection, ...]
    threats: tuple[Detection, ...]
    farmables: tuple[Detection, ...]
    pickups: tuple[Detection, ...]
    wisps: tuple[Detection, ...]
    obstacles: tuple[Polygon, ...]
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    # the raw frame, kept only when the debug view needs it
    frame: Optional[object] = field(default=None, compare=False, repr=False)

    def reachable(self, target: Point) -> bool:
        return path_clear(self.player, target, self.obstacles)

    def is_close(self, target: Point, fraction: float) -> bool:
        return self.player.dist(target) < self.window_size[1] * fraction

    def by_distance(self, items: Sequence[Detection]) -> list[Detection]:
        return sorted(items, key=lambda d: self.player.dist(d.center))

    def is_empty(self) -> bool:
        return len(self.detections) == 0


@dataclass(frozen=True)
class Status:
    timestamp: float
    spells: tuple[str, ...]          # class per slot, "up" or "cooldown"
    spells_calibrated: bool
    low_health: bool
    loading_bar: Optional[bool] = None   # None when the loading bar model wasn't run
    health_fraction: Optional[float] = None  # None when the HUD could not be read
    frame: Optional[object] = field(default=None, compare=False, repr=False)

    def ready(self, slot_index: int) -> bool:
        return 0 <= slot_index < len(self.spells) and self.spells[slot_index] == "up"


T = TypeVar("T")


class Latest(Generic[T]):
    """
    Holds the newest item by timestamp.

    Several workers can publish at once. An item older than the current one is
    dropped, so a slow worker can't replace a newer result with an older one.
    wait_newer lets a caller block until something newer arrives, without polling.
    """

    def __init__(self) -> None:
        self._item: Optional[T] = None
        self._cond = threading.Condition()
        self.published = 0
        self.dropped = 0
        self._times: list[float] = []

    def publish(self, item: T) -> bool:
        with self._cond:
            if self._item is not None and item.timestamp < self._item.timestamp:  # type: ignore[attr-defined]
                self.dropped += 1
                return False
            self._item = item
            self.published += 1
            now = time.monotonic()
            self._times.append(now)
            if len(self._times) > 20:
                self._times.pop(0)
            self._cond.notify_all()
            return True

    def get(self) -> Optional[T]:
        with self._cond:
            return self._item

    def wait_newer(self, than: float, timeout: float) -> Optional[T]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._item is None or self._item.timestamp <= than:  # type: ignore[attr-defined]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._item

    def rate(self) -> float:
        """Published items per second over the last 20."""
        with self._cond:
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0
