"""Fakes for running the brain and behaviors without the game."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from albot.actuator import DryRunInput
from albot.behaviors import SystemRandom
from albot.brain import Brain, Context
from albot.config import Config
from albot.state import Latest, Status
from albot.vision import build_scene
from albot.window import FixedWindow

W, H = 1000, 800
# where the player stands by default: centre plus the (2, -95) offset
PLAYER = (W / 2 + 2, H / 2 - 95)


class FakeClock:
    """Time only moves when something sleeps. Callbacks fire at set times."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.events: list[tuple[float, callable]] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        due = [e for e in self.events if e[0] <= self.t]
        self.events = [e for e in self.events if e[0] > self.t]
        for _, fn in sorted(due, key=lambda e: e[0]):
            fn()

    def at(self, offset: float, fn) -> None:
        self.events.append((self.t + offset, fn))


class FakeWorkers:
    def __init__(self) -> None:
        self.scenes = Latest()
        self.statuses = Latest()
        self.want_loading_bar = threading.Event()
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


def pred(class_name, x, y, confidence=0.9, w=30, h=30):
    return SimpleNamespace(class_name=class_name, x=x, y=y, confidence=confidence, width=w, height=h)


def poly(points, confidence=0.9):
    return SimpleNamespace(confidence=confidence, points=[SimpleNamespace(x=x, y=y) for x, y in points])


class World:
    """A brain wired to fakes, with helpers to set what the bot sees."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.clock = FakeClock()
        self.workers = FakeWorkers()
        self.window = FixedWindow(W, H)
        self.input = DryRunInput()
        self.ctx = Context(self.config, self.input, self.workers, self.window,
                           clock=self.clock, rng=SystemRandom(1))
        self.brain = Brain(self.ctx)
        self.spells = ["up"] * 6
        self.low = False
        self.loading = None

    def see(self, predictions=(), obstacles=(), player=True):
        preds = list(predictions)
        if player:
            preds.append(pred("player", *PLAYER))
        scene = build_scene(self.clock.now(), (W, H), preds, list(obstacles), self.config)
        self.workers.scenes.publish(scene)
        self.status()
        return scene

    def status(self, spells=None, low=None, loading="keep"):
        if spells is not None:
            self.spells = list(spells)
        if low is not None:
            self.low = low
        if loading != "keep":
            self.loading = loading
        self.workers.statuses.publish(Status(self.clock.now(), tuple(self.spells), True, self.low, self.loading))

    def tick(self):
        self.brain.tick()
        return self.input.actions
