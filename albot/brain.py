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
from .geometry import Point
from .behaviors import Behavior, Blacklist, Plan, SystemRandom, danger, default_behaviors
from .state import Detection, Scene, Status
from .resources import ResourceMemory
from .safety import DeathGuard, SafetyStop, TravelGuard

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
        self.resources = ResourceMemory(config, self.clock)
        self.death_guard = DeathGuard(config)
        self.travel_guard = TravelGuard(config)
        self.travel_moving = False
        self.travel_retreating = False
        self.active_behavior = ""
        self.behaviors = []
        self.quit = threading.Event()
        self.bored = 0
        self.last_ate = self.clock.now()
        self.current_plan: Optional[Plan] = None
        self._safety_reason = ""

    # what the bot sees -----------------------------------------------------
    def scene(self) -> Scene:
        return self.workers.scenes.get()

    def status(self) -> Status:
        return self.workers.statuses.get()

    def danger(self, count_low_health: bool = True) -> bool:
        return self.under_attack() or danger(self.scene(), self.status(), self.config, count_low_health)

    def under_attack(self) -> bool:
        return self.travel_guard.under_attack(self.now())

    def travel_enabled(self) -> bool:
        return bool(self.config.behavior.travel_goal) and any(b.name == "travel" for b in self.behaviors)

    def higher_priority_applies(self, name: str) -> bool:
        scene, status = self.scene(), self.status()
        if name == "travel" and self.retreat_required(scene):
            return False
        for behavior in self.behaviors:
            if behavior.name == name:
                break
            if behavior.plan(self, scene, status) is not None:
                return True
        return False

    def retreat_required(self, scene=None) -> bool:
        guard = self.travel_guard
        scene = scene if scene is not None else self.scene()
        unseen = (scene is None or self.now()-scene.timestamp > self.config.perception.stale_after
                  or not scene.mobs)
        return (guard.critical_since is not None
                or (guard.retreat_started is not None and unseen))

    def frame(self):
        """Use the health worker's capture, so navigation does not grab its own frames."""
        status = self.status()
        return status.frame if status is not None else None

    def stop_travel(self) -> None:
        if self.travel_moving:
            # Called after safety checks (also from check_safety itself): do not
            # recurse through ctx.press. Input still enforces window focus.
            self.input.press(self.config.keys.stop)
            self.travel_moving = False
            self.travel_retreating = False
            if self.scene() is not None:
                self.resources.move(self.scene(), None)

    def slot(self, key: str) -> int:
        return self.config.keys.slots.index(key)

    def now(self) -> float:
        return self.clock.now()

    # waiting -----------------------------------------------------------------
    def check_safety(self) -> None:
        if self.quit.is_set():
            raise SafetyStop()
        status = self.status()
        if (self.travel_guard.critical_since is not None and status is not None
                and status.health_fraction is not None
                and status.health_fraction > self.config.safety.zero_health):
            # Very low positive HP can still be alive during the bounded escape.
            # Keep immediate zero-health detection, without latching the minimum hold.
            self.death_guard.minimum_since = None
        if self.death_guard.observe(status, self.now()):
            count = len(self.death_guard.deaths)
            if count >= self.config.safety.death_limit:
                self.quit.set()
            self.resources.clear()
            log.error("death detected at %s (health %.3f): %d in %.0fs%s",
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status.health_fraction,
                      count, self.config.safety.death_window_seconds,
                      "; quitting" if self.quit.is_set() else "; waiting for recovery")
            if self.saver is not None:
                self.saver.death(status.health_fraction, status.frame)
        travelling = self.travel_enabled()
        guard = self.travel_guard
        if travelling and guard.abort(status, self.now()):
            if guard.critical_since is None:
                log.error("health %.3f below %.3f: retreating for up to %.1fs before quitting",
                          status.health_fraction, self.config.safety.travel_abort_health,
                          self.config.safety.critical_retreat_seconds)
            guard.start_retreat(self.now(), critical=True)
        if travelling and self.travel_guard.retreat_expired(self.now()):
            self.quit.set()
            log.error("critical travel retreat finished after %.1fs; quitting",
                      self.config.safety.critical_retreat_seconds)
            if not self.death_guard.dead:
                self.stop_travel()
            raise SafetyStop()
        reason = ""
        if self.death_guard.unsafe:
            reason = "death detected; waiting for sustained health recovery"
        elif status is None or self.now() - status.timestamp > self.config.perception.stale_after:
            reason = "health status missing or stale"
        elif travelling and not self.travel_guard.valid(status, self.now()):
            reason = "travel requires readable numeric HUD health; model fallback is insufficient"
        elif status.health_fraction is None and not self.config.health.model_fallback:
            reason = "HUD health unreadable; recalibrate [health] or explicitly enable model_fallback"
        if reason != self._safety_reason:
            if reason:
                log.warning("input suspended: %s", reason)
            else:
                log.warning("health readable; input resumed")
            self._safety_reason = reason
        if reason:
            if travelling and not self.death_guard.dead:
                self.stop_travel()
            raise SafetyStop()
        if self.travel_guard.observe(status, self.now(), travelling,
                                     self.active_behavior in ("fight_close", "fight_far")):
            log.warning("under attack while travelling: HP fell more than %.0f%% within %.1fs",
                        self.config.safety.attack_health_drop*100, self.config.safety.attack_window_seconds)
        scene = self.scene()
        if travelling and self.under_attack() and (scene is None or not scene.mobs
                or self.now()-scene.timestamp > self.config.perception.stale_after):
            guard.start_retreat(self.now())
        if (self.under_attack() or self.retreat_required()) and not self.travel_retreating:
            self.stop_travel()
        if self.retreat_required() and self.active_behavior not in ("", "travel"):
            raise SafetyStop()  # unwind a cast/heal wait; the next tick dispatches retreat

    def wait(self, seconds: float, interrupt: Optional[Callable[[], bool]] = None) -> bool:
        """Wait, checking focus throughout. False if `interrupt` fired first."""
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            self.check_safety()
            if not self.window.is_active():
                raise FocusLost()
            if interrupt is not None and interrupt():
                return False
            self.clock.sleep(0.02)
        self.check_safety()
        return True

    def wait_until(self, condition: Callable[[], bool], timeout: float,
                   interrupt: Optional[Callable[[], bool]] = None) -> bool:
        """True once `condition` holds. False on timeout or interrupt."""
        end = self.clock.now() + timeout
        while self.clock.now() < end:
            self.check_safety()
            if not self.window.is_active():
                raise FocusLost()
            if condition():
                return True
            if interrupt is not None and interrupt():
                return False
            self.clock.sleep(0.02)
        self.check_safety()
        return condition()

    def wait_for_new_scene(self, timeout: float = 2.0) -> None:
        current = self.scene()
        self.workers.scenes.wait_newer(current.timestamp if current else 0, timeout)

    def want_loading_bar(self, on: bool) -> None:
        (self.workers.want_loading_bar.set if on else self.workers.want_loading_bar.clear)()

    # acting --------------------------------------------------------------------
    def press(self, key) -> None:
        self.check_safety()
        self.input.press(key)

    def move_to(self, point) -> None:
        self.check_safety()
        self.input.move_to(point)

    def click(self, point, stop: bool = True) -> None:
        self.check_safety()
        self.input.click(point, stop=stop)
        if self.scene() is not None:
            self.resources.move(self.scene(), point)

    def walk_to(self, point) -> None:
        self.check_safety()
        self.input.walk_to(point)
        if self.scene() is not None:
            self.resources.move(self.scene(), point)

    def cast(self, step) -> None:
        for _ in range(step.repeat):
            self.press(step.slot)
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
        priority = list(b.priority)
        if b.travel_goal and "travel" not in priority and "travel" not in b.disabled:
            # Existing albot.toml files predate travel. Insert it without requiring
            # users to copy the new example priority list.
            priority.insert(priority.index("fight_close")+1 if "fight_close" in priority else 0, "travel")
        self.behaviors = [all_behaviors[n] for n in priority if n not in b.disabled]
        if b.travel_goal and "travel" not in b.disabled:
            names = [behavior.name for behavior in self.behaviors]
            if any(n not in names or names.index(n) > names.index("travel")
                   for n in ("shield", "heal", "fight_close")):
                raise ValueError("travel requires shield, heal and fight_close ahead of it in priority")
        ctx.behaviors = self.behaviors
        self.counts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.errors = 0
        self.paused = threading.Event()
        self.quit = ctx.quit
        self.pause_reason = ""
        self._empty_since: Optional[float] = None
        self._pressed_esc = False
        self._lost_steps = 0

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
            self._lost_steps = 0
            log.warning("RESUMED")

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.resume()
        else:
            self.pause("paused by hotkey")

    # deciding ------------------------------------------------------------------
    def choose(self, scene: Scene, status: Status, skip: tuple[str, ...] = ()) -> Optional[tuple[Behavior, Plan]]:
        if self.ctx.retreat_required(scene) and "travel" not in skip:
            for behavior in self.behaviors:
                if behavior.name == "travel":
                    plan = behavior.plan(self.ctx, scene, status)
                    if plan is not None:
                        return behavior, plan
        for behavior in self.behaviors:
            if behavior.name in skip:
                continue
            plan = behavior.plan(self.ctx, scene, status)
            if plan is not None:
                return behavior, plan
        return None

    def tick(self) -> None:
        try:
            self._tick()
        except SafetyStop:
            self.ctx.current_plan = None
            self.ctx.clock.sleep(0.1)

    def _tick(self) -> None:
        ctx = self.ctx
        ctx.check_safety()
        scene, status = ctx.scene(), ctx.status()
        retreat = ctx.retreat_required(scene)
        if not retreat and (scene is None or status is None or ctx.now() - scene.timestamp > self.config.perception.stale_after):
            ctx.wait_for_new_scene(timeout=0.5)
            return

        if ctx.saver is not None:
            ctx.saver.maybe("main", self.config.data.sample_main)

        if not retreat and scene.is_empty():
            self._handle_nothing_detected()
            return
        self._empty_since = None
        self._pressed_esc = False
        self._lost_steps = 0

        tried: list[str] = []
        while True:
            ctx.check_safety()
            choice = self.choose(ctx.scene(), ctx.status(), tuple(tried))
            if choice is None:
                break
            behavior, plan = choice
            ctx.current_plan = plan
            log.info("%-11s %s", behavior.name.upper(), plan.description)
            if behavior.name != "travel":
                ctx.stop_travel()
            ctx.active_behavior = behavior.name
            try:
                ok = behavior.run(ctx, plan)
            finally:
                ctx.active_behavior = ""
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
        status = ctx.status()
        if status is not None and status.health_fraction is not None:
            # The HUD is readable, so we're in the world with no menu open: the
            # model just can't see anything (2026-11-04: the player stood behind
            # a palm). Esc would open the game menu (Logout, Close Application),
            # so step out of cover instead, then pause if that doesn't help.
            steps = int(waited // limit)
            if steps > self._lost_steps and steps <= 4:
                self._lost_steps = steps
                w, h = scene.window_size if (scene := ctx.scene()) is not None else (0, 0)
                if w and h:
                    dx, dy = ((300, 0), (0, 250), (-300, 0), (0, -250))[(steps - 1) % 4]
                    p = Point(w / 2 + self.config.detection.player_offset[0] + dx,
                              h / 2 + self.config.detection.player_offset[1] + dy)
                    log.info("nothing detected for %.0fs with the HUD visible; stepping out of cover", waited)
                    ctx.click(p, stop=False)
            elif steps > 4:
                if ctx.saver is not None:
                    ctx.saver.save("nothing_detected")
                self.pause(f"nothing detected for {waited:.0f}s")
            ctx.clock.sleep(0.2)
            return
        if waited >= limit and not self._pressed_esc:
            log.info("nothing detected for %.0fs, pressing %s", waited, self.config.keys.close_menus)
            ctx.press(self.config.keys.close_menus)
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
