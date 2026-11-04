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
from .hud import hud_reason
from .safety import SafetyStop

log = logging.getLogger(__name__)


class FocusLost(Exception):
    """Raised when the game loses focus mid-action. The brain pauses on it."""


class MouseFailsafe(Exception):
    """pyautogui's corner failsafe fired. The brain pauses on it."""


class Input:
    def __init__(self, window, config) -> None:
        self.window = window
        self.keys = config.keys
        self.safety = config.safety

    def _check_focus(self) -> None:
        if not self.window.is_active():
            raise FocusLost()

    def release(self) -> None:
        """Release only controls the bot uses; key-up must also work after focus loss."""
        import keyboard
        import pyautogui
        for key in dict.fromkeys([*self.keys.slots, self.keys.attack_after_click, self.keys.stop,
                                  self.keys.food, self.keys.close_menus]):
            keyboard.release(key)
        for button in dict.fromkeys(("left", self.keys.move_button)):
            pyautogui.mouseUp(button=button)

    def _to_screen(self, p: Point) -> tuple[int, int]:
        r = self.window.rect()
        return int(r.left + p.x), int(r.top + p.y)

    def press(self, key: str) -> None:
        self._check_focus()
        import keyboard
        keyboard.press_and_release(key)

    def click(self, p: Point, stop: bool = True) -> None:
        """Stop, move to p, left click: what helper.clickOn did. Travel skips the stop."""
        self._check_focus()
        self._check_world(p)
        self._left_click(p, stop)

    def _check_world(self, p):
        rect = self.window.rect()
        reason = hud_reason(p, (rect.width, rect.height), self.safety)
        if reason:
            log.info("world click/aim refused at %.0f,%.0f: %s", p.x, p.y, reason)
            raise SafetyStop()

    def toggle_minimap(self):
        """The only UI click: fixed configured button, never an arbitrary world target."""
        self._check_focus()
        rect = self.window.rect()
        x, y = self.safety.minimap_toggle
        w, h = self.safety.hud_size
        self._left_click(Point(x*rect.width/w, y*rect.height/h), stop=False)

    def _left_click(self, p, stop):
        import keyboard
        import pyautogui
        try:
            if stop:
                keyboard.press_and_release(self.keys.stop)
            pyautogui.moveTo(*self._to_screen(p), duration=0.2)
            pyautogui.click()
        except pyautogui.FailSafeException as e:
            raise MouseFailsafe() from e

    def walk_to(self, p: Point) -> None:
        self._check_focus()
        self._check_world(p)
        import pyautogui
        try:
            pyautogui.moveTo(*self._to_screen(p), duration=0.2)
            pyautogui.click(button=self.keys.move_button)
        except pyautogui.FailSafeException as e:
            raise MouseFailsafe() from e

    def move_to(self, p: Point) -> None:
        """Aim a directional spell without clicking or stopping movement."""
        self._check_focus()
        self._check_world(p)
        import pyautogui
        try:
            pyautogui.moveTo(*self._to_screen(p), duration=0.2)
        except pyautogui.FailSafeException as e:
            raise MouseFailsafe() from e


@dataclass
class DryRunInput:
    """Logs what would be pressed and clicked, and sends nothing. For testing."""
    actions: list[tuple] = field(default_factory=list)

    def toggle_minimap(self) -> None:
        self.actions.append(("minimap_toggle",))
        log.info("[dry run] toggle minimap")

    def release(self) -> None:
        self.actions.append(("release",))

    def press(self, key: str) -> None:
        self.actions.append(("press", key))
        log.info("[dry run] press %s", key)

    def click(self, p: Point, stop: bool = True) -> None:
        self.actions.append(("click", round(p.x), round(p.y)))
        log.info("[dry run] click %d,%d", p.x, p.y)

    def walk_to(self, p: Point) -> None:
        self.actions.append(("walk", round(p.x), round(p.y)))
        log.info("[dry run] walk to %d,%d", p.x, p.y)

    def move_to(self, p: Point) -> None:
        self.actions.append(("move", round(p.x), round(p.y)))
        log.info("[dry run] aim at %d,%d", p.x, p.y)
