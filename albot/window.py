"""The game window: where it is, and whether it has focus."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int


class GameWindow:
    def __init__(self, title: str) -> None:
        import pygetwindow  # Windows only; imported here so tests run anywhere
        self._gw = pygetwindow
        self.title = title
        self._window = None

    def wait_for(self) -> None:
        announced = False
        while self._find() is None:
            if not announced:
                log.info("waiting for '%s' to open", self.title)
                announced = True
            time.sleep(2)
        log.info("found '%s'", self.title)

    def _find(self):
        windows = self._gw.getWindowsWithTitle(self.title)
        self._window = windows[0] if windows else None
        return self._window

    def _get(self):
        # the handle goes stale if the game restarts; look it up again then
        if self._window is None:
            self.wait_for()
        return self._window

    def rect(self) -> Rect:
        w = self._get()
        try:
            return Rect(w.left, w.top, w.width, w.height)
        except Exception:
            self._window = None
            w = self._get()
            return Rect(w.left, w.top, w.width, w.height)

    def is_active(self) -> bool:
        try:
            return bool(self._get().isActive)
        except Exception:
            self._window = None
            return False


class FixedWindow:
    """Stands in for the game window in replay mode and tests."""

    def __init__(self, width: int, height: int, active: bool = True) -> None:
        self._rect = Rect(0, 0, width, height)
        self.active = active

    def rect(self) -> Rect:
        return self._rect

    def is_active(self) -> bool:
        return self.active

    def wait_for(self) -> None:
        pass
