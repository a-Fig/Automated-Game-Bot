"""
The decision loop.

Each tick: take the newest scene and status, ask each behavior in priority order
for a plan, and run the first one. If it fails, the next behavior gets a turn in
the same tick, as in the original if-chain. Pausing, focus loss, the
nothing-detected check and errors are all handled here, so behaviors only
describe the game.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Callable, Optional

from .actuator import FocusLost, MouseFailsafe
from .behaviors import Behavior, Blacklist, Plan, SystemRandom, danger, default_behaviors
from .state import Detection, Scene, Status

log = logging.getLogger(__name__)


class Clock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class Context:
    """Everything a behavior can see and do."""

    def __init__(self, config, input_, workers, window, clock=None, rng=None, saver=None) -> None:
        self.config = config
        self.input = input_
        self.workers = workers
        self.window = window
        self.clock = clock or Clock()
        self.random = rng or SystemRandom()
        self.saver = saver
        self.blacklist = Blacklist(config, self.clock)
        self.bored = 0
        self.last_ate = self.clock.now()
        self.current_plan: Optional[Plan] = None

    # what the bot sees -----------------------------------------------------
    def scene(self) -> Scene:
        return self.workers.scenes.get()

    def status(self) -> Status:
        return self.workers.statuses.get()

    def danger(self, count_low_health: bool = True) -> bool:
        return danger(self.scene(), self.status(), self.config, count_low_health)

    def slot(self, key: str) -> int:
        return self.config.keys.slots.index(key)

    def now(self) -> float:
        return self.clock.now()

    # waiting -----------------------------------------------------------------
    def wait(self, seconds: float, interrupt: Optional[Callable[[], bool]] = None) -> bool:
        """Wait, checking focus throughout. False if `interrupt` fired first."""
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            if not self.window.is_active():
                raise FocusLost()
            if interrupt is not None and interrupt():
                return False
            self.clock.sleep(0.02)
        return True

    def wait_until(self, condition: Callable[[], bool], timeout: float,
                   interrupt: Optional[Callable[[], bool]] = None) -> bool:
        """True once `condition` holds. False on timeout or interrupt."""
        end = self.clock.now() + timeout
        while self.clock.now() < end:
            if not self.window.is_active():
                raise FocusLost()
            if condition():
                return True
            if interrupt is not None and interrupt():
                return False
            self.clock.sleep(0.02)
        return condition()

    def wait_for_new_scene(self, timeout: float = 2.0) -> None:
        current = self.scene()
        self.workers.scenes.wait_newer(current.timestamp if current else 0, timeout)

    def want_loading_bar(self, on: bool) -> None:
        (self.workers.want_loading_bar.set if on else self.workers.want_loading_bar.clear)()

    # acting --------------------------------------------------------------------
    def cast(self, step) -> None:
        for _ in range(step.repeat):
            self.input.press(step.slot)
            self.wait(step.delay)

    def cast_if_ready(self, step) -> bool:
        if self.status().ready(self.slot(step.slot)):
            self.cast(step)
            return True
        return False

    def check_progress(self, target: Detection) -> None:
        """After acting on a target: is the same thing still at the same screen spot?"""
        scene = self.scene()
        radius = self.config.behavior.stuck_radius
        still_there = any(
            d.class_name == target.class_name and d.center.dist(target.center) <= radius
            for d in scene.detections
        )
        if still_there:
            self.blacklist.failure(target.center)
        else:
            self.blacklist.success(target.center)


class Brain:
    def __init__(self, ctx: Context, behaviors: Optional[dict[str, Behavior]] = None) -> None:
        self.ctx = ctx
        self.config = ctx.config
        all_behaviors = behaviors or default_behaviors()
        b = self.config.behavior
        unknown = [n for n in b.priority if n not in all_behaviors]
        if unknown:
            raise ValueError(f"unknown behaviors in priority list: {unknown}")
        self.behaviors = [all_behaviors[n] for n in b.priority if n not in b.disabled]
        self.counts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.errors = 0
        self.paused = threading.Event()
        self.quit = threading.Event()
        self.pause_reason = ""
        self._empty_since: Optional[float] = None
        self._pressed_esc = False

    # control -------------------------------------------------------------------
    def pause(self, reason: str) -> None:
        if not self.paused.is_set():
            self.pause_reason = reason
            self.paused.set()
            self.ctx.workers.pause()
            log.warning("PAUSED: %s. Press %s to resume, %s to quit.",
                        reason, self.config.keys.pause.upper(), self.config.keys.quit.upper())

    def resume(self) -> None:
        if self.paused.is_set():
            self.paused.clear()
            self.ctx.workers.resume()
            self._empty_since = None
            log.warning("RESUMED")

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.resume()
        else:
            self.pause("paused by hotkey")

    # deciding ------------------------------------------------------------------
    def choose(self, scene: Scene, status: Status, skip: tuple[str, ...] = ()) -> Optional[tuple[Behavior, Plan]]:
        for behavior in self.behaviors:
            if behavior.name in skip:
                continue
            plan = behavior.plan(self.ctx, scene, status)
            if plan is not None:
                return behavior, plan
        return None

    def tick(self) -> None:
        ctx = self.ctx
        scene, status = ctx.scene(), ctx.status()
        if scene is None or status is None or ctx.now() - scene.timestamp > self.config.perception.stale_after:
            ctx.wait_for_new_scene(timeout=0.5)
            return

        if ctx.saver is not None:
            ctx.saver.maybe("main", self.config.data.sample_main)

        if scene.is_empty():
            self._handle_nothing_detected()
            return
        self._empty_since = None
        self._pressed_esc = False

        tried: list[str] = []
        while True:
            choice = self.choose(ctx.scene(), ctx.status(), tuple(tried))
            if choice is None:
                break
            behavior, plan = choice
            ctx.current_plan = plan
            log.info("%-11s %s", behavior.name.upper(), plan.description)
            ok = behavior.run(ctx, plan)
            if ok:
                self.counts[behavior.name] += 1
                if getattr(behavior, "resets_boredom", True):
                    ctx.bored = 0
                return
            self.failures[behavior.name] += 1
            tried.append(behavior.name)

        ctx.current_plan = None
        ctx.bored += 1
        if ctx.saver is not None:
            ctx.saver.maybe("bored", self.config.data.sample_bored)
        ctx.clock.sleep(self.config.behavior.idle_sleep)

    def _handle_nothing_detected(self) -> None:
        """
        Nothing on screen usually means a menu is open, a loading screen, or death.
        After a few seconds, press esc once (closes most menus). If there's still
        nothing, pause, so the bot doesn't click blindly.
        """
        ctx = self.ctx
        now = ctx.now()
        if self._empty_since is None:
            self._empty_since = now
        if ctx.saver is not None:
            ctx.saver.maybe("nothing_detected", self.config.data.sample_nothing_detected)
        waited = now - self._empty_since
        limit = self.config.behavior.lost_seconds
        if waited >= limit and not self._pressed_esc:
            log.info("nothing detected for %.0fs, pressing %s", waited, self.config.keys.close_menus)
            ctx.input.press(self.config.keys.close_menus)
            self._pressed_esc = True
        elif waited >= 2 * limit:
            if ctx.saver is not None:
                ctx.saver.save("nothing_detected")
            self.pause(f"nothing detected for {waited:.0f}s")
        ctx.clock.sleep(0.2)

    # running -------------------------------------------------------------------
    def run(self) -> None:
        ctx = self.ctx
        while not self.quit.is_set():
            if self.paused.is_set():
                self.quit.wait(0.2)
                continue
            if not ctx.window.is_active():
                self._wait_for_focus()
                continue
            try:
                self.tick()
            except FocusLost:
                log.info("game lost focus mid-action")
            except MouseFailsafe:
                self.pause("mouse moved to a screen corner (failsafe)")
            except Exception:
                self.errors += 1
                log.exception("error during a tick; carrying on")
                ctx.clock.sleep(0.5)

    def _wait_for_focus(self) -> None:
        ctx = self.ctx
        log.warning("game window not focused, waiting (no input is sent meanwhile)")
        ctx.workers.pause()
        while not ctx.window.is_active() and not self.quit.is_set() and not self.paused.is_set():
            self.quit.wait(0.25)
        if self.quit.is_set() or self.paused.is_set():
            return
        log.warning("focus back, resuming in 2s")
        self.quit.wait(2)
        ctx.workers.resume()
        ctx.wait_for_new_scene(timeout=3)

    def summary(self) -> str:
        parts = [f"{name} {n}" for name, n in self.counts.most_common()]
        fails = sum(self.failures.values())
        return f"actions: {', '.join(parts) or 'none'} | failed attempts {fails} | errors {self.errors}"
