"""
Keyboard and mouse output.

Everything goes through Input, which checks the game window has focus before
each key or click. If you alt-tab mid-action, the bot can't type into your other
window. Positions arrive in window coordinates and are converted to screen
coordinates here, at the moment of the click.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .geometry import Point

log = logging.getLogger(__name__)


class FocusLost(Exception):
    """Raised when the game loses focus mid-action. The brain pauses on it."""


class MouseFailsafe(Exception):
    """pyautogui's corner failsafe fired. The brain pauses on it."""


class Input:
    def __init__(self, window, config) -> None:
        self.window = window
        self.keys = config.keys

    def _check_focus(self) -> None:
        if not self.window.is_active():
            raise FocusLost()

    def _to_screen(self, p: Point) -> tuple[int, int]:
        r = self.window.rect()
        return int(r.left + p.x), int(r.top + p.y)

    def press(self, key: str) -> None:
        self._check_focus()
        import keyboard
        keyboard.press_and_release(key)

    def click(self, p: Point) -> None:
        """Stop, move to p, left click: what helper.clickOn did."""
        self._check_focus()
        import keyboard
        import pyautogui
        try:
            keyboard.press_and_release(self.keys.stop)
            pyautogui.moveTo(*self._to_screen(p), duration=0.2)
            pyautogui.click()
        except pyautogui.FailSafeException as e:
            raise MouseFailsafe() from e

    def walk_to(self, p: Point) -> None:
        self._check_focus()
        import pyautogui
        try:
            pyautogui.moveTo(*self._to_screen(p), duration=0.2)
            pyautogui.click(button=self.keys.move_button)
        except pyautogui.FailSafeException as e:
            raise MouseFailsafe() from e


@dataclass
class DryRunInput:
    """Logs what would be pressed and clicked, and sends nothing. For testing."""
    actions: list[tuple] = field(default_factory=list)

    def press(self, key: str) -> None:
        self.actions.append(("press", key))
        log.info("[dry run] press %s", key)

    def click(self, p: Point) -> None:
        self.actions.append(("click", round(p.x), round(p.y)))
        log.info("[dry run] click %d,%d", p.x, p.y)

    def walk_to(self, p: Point) -> None:
        self.actions.append(("walk", round(p.x), round(p.y)))
        log.info("[dry run] walk to %d,%d", p.x, p.y)
