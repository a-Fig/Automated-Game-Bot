"""
Map spell detections to fixed action bar slots.

Sorting detections by x and padding the end breaks when the model misses a slot
in the middle: everything to its right slides down one index. Instead, a frame
with exactly one detection per slot records each slot's x position, and later
frames snap each detection to its nearest recorded slot. Empty slots read as
cooldown. Until the first full frame, every slot reads as cooldown, so the bot
holds spells instead of casting from a guessed layout.
"""
from __future__ import annotations

import threading
from typing import Sequence


class SpellSlotMapper:
    def __init__(self, slot_count: int = 6) -> None:
        self.slot_count = slot_count
        self.slot_x: list[float] | None = None
        self._lock = threading.Lock()

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self.slot_x is not None

    def map(self, predictions: Sequence) -> tuple[str, ...]:
        """predictions need .x, .class_name and .confidence. Returns a class per slot."""
        preds = sorted(predictions, key=lambda p: p.x)
        if len(preds) == self.slot_count:
            with self._lock:
                self.slot_x = [p.x for p in preds]
            return tuple(p.class_name for p in preds)

        with self._lock:
            slot_x = self.slot_x
        if slot_x is None:
            return ("cooldown",) * self.slot_count

        result = ["cooldown"] * self.slot_count
        best = [-1.0] * self.slot_count
        for p in preds:
            i = min(range(self.slot_count), key=lambda s: abs(slot_x[s] - p.x))
            if p.confidence > best[i]:
                result[i] = p.class_name
                best[i] = p.confidence
        return tuple(result)
