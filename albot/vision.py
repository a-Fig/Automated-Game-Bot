"""
Screen capture, the Roboflow models, and turning model output into Scene and Status.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from .config import Config
from .geometry import Point, Polygon
from .spells import SpellSlotMapper
from .state import Detection, Scene, Status
from .window import Rect

log = logging.getLogger(__name__)


class Capture:
    """
    Grabs the game window.

    "mss" returns a numpy array in BGR order, which the Roboflow inference package
    treats as an OpenCV image. "pyautogui" returns a PIL image in RGB, which is
    what the original bot passed in. Both should give the models the same picture.
    If detections look worse with mss, switch the backend in the config and compare:
    that would mean the colour order assumption is wrong.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        if backend == "mss":
            import mss  # noqa: F401  checked here so a missing package fails at startup
            import numpy  # noqa: F401
        elif backend != "pyautogui":
            raise ValueError(f"unknown capture backend '{backend}'")
        # mss handles are per thread
        import threading
        self._local = threading.local()

    def grab(self, rect: Rect) -> Any:
        if self.backend == "mss":
            import mss
            import numpy as np
            sct = getattr(self._local, "sct", None)
            if sct is None:
                sct = self._local.sct = mss.mss()
            shot = sct.grab({"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height})
            return np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        import pyautogui
        return pyautogui.screenshot(region=(rect.left, rect.top, rect.width, rect.height))


class Models:
    """Loads the five Roboflow models. Weights download once, then run locally."""

    def __init__(self, config: Config) -> None:
        import inference  # heavy; imported here so tests and --help stay fast
        m = config.models
        self.scene = inference.get_model(m.scene, config.key_for_model("scene"))
        self.obstacles = inference.get_model(m.obstacles, config.key_for_model("obstacles"))
        self.spells = inference.get_model(m.spells, config.key_for_model("spells"))
        self.loading_bar = inference.get_model(m.loading_bar, config.key_for_model("loading_bar"))
        self.health = inference.get_model(m.health, config.key_for_model("health"))


def predict(model: Any, image: Any) -> list:
    return model.infer(image)[0].predictions


def choose_player(predictions: Sequence, config: Config, size: tuple[int, int]) -> tuple[Point, bool]:
    """
    The player is the "player" detection nearest where the player usually stands.
    The original took the first "player" in the model's output, which can be
    another player standing nearby.
    """
    expected = Point(size[0] / 2 + config.detection.player_offset[0],
                     size[1] / 2 + config.detection.player_offset[1])
    players = [p for p in predictions
               if p.class_name == config.detection.player_class
               and p.confidence >= config.confidence_for(p.class_name)]
    if not players:
        return expected, False
    best = min(players, key=lambda p: Point(p.x, p.y).dist(expected))
    return Point(best.x, best.y), True


def build_scene(timestamp: float, size: tuple[int, int], predictions: Sequence,
                obstacle_predictions: Sequence, config: Config,
                capture_ms: float = 0.0, inference_ms: float = 0.0, frame: Any = None) -> Scene:
    d = config.detection
    player, detected = choose_player(predictions, config, size)
    groups: dict[str, list[Detection]] = {"mob": [], "threat": [], "farm": [], "pickup": [], "wisp": []}
    by_class = {}
    for name in d.mob_classes:
        by_class[name] = "mob"
    for name in d.threat_classes:
        by_class[name] = "threat"
    for name in d.farm_classes:
        by_class[name] = "farm"
    for name in d.pickup_classes:
        by_class[name] = "pickup"
    for name in d.wisp_classes:
        by_class[name] = "wisp"

    detections = []
    for p in predictions:
        det = Detection(p.class_name, p.confidence, Point(p.x, p.y),
                        getattr(p, "width", 0.0), getattr(p, "height", 0.0))
        detections.append(det)
        group = by_class.get(p.class_name)
        if group and p.confidence >= config.confidence_for(p.class_name):
            groups[group].append(det)

    obstacles = []
    for o in obstacle_predictions:
        pts = getattr(o, "points", None) or []
        if o.confidence >= d.min_obstacle_confidence and len(pts) >= 3:
            obstacles.append(Polygon(tuple(Point(pt.x, pt.y) for pt in pts), o.confidence))

    def near_first(items: list[Detection]) -> tuple[Detection, ...]:
        return tuple(sorted(items, key=lambda x: player.dist(x.center)))

    return Scene(
        timestamp=timestamp, window_size=size, player=player, player_detected=detected,
        detections=tuple(detections),
        mobs=near_first(groups["mob"]), threats=near_first(groups["threat"]),
        farmables=near_first(groups["farm"]), pickups=near_first(groups["pickup"]),
        wisps=near_first(groups["wisp"]), obstacles=tuple(obstacles),
        capture_ms=capture_ms, inference_ms=inference_ms, frame=frame,
    )


class Perceiver:
    """Runs the models on a frame. One instance is shared by all worker threads."""

    def __init__(self, config: Config, models: Models, capture: Capture | None, window) -> None:
        self.config = config
        self.models = models
        self.capture = capture
        self.window = window
        self.spell_slots = SpellSlotMapper(len(config.keys.slots))

    def scene_from(self, frame: Any, size: tuple[int, int], timestamp: float,
                   capture_ms: float = 0.0) -> Scene:
        t0 = time.perf_counter()
        preds = predict(self.models.scene, frame)
        obstacles = predict(self.models.obstacles, frame)
        inference_ms = (time.perf_counter() - t0) * 1000
        keep = frame if self.config.debug.view else None
        return build_scene(timestamp, size, preds, obstacles, self.config, capture_ms, inference_ms, keep)

    def status_from(self, frame: Any, timestamp: float, want_loading_bar: bool) -> Status:
        spells = self.spell_slots.map(predict(self.models.spells, frame))
        health = predict(self.models.health, frame)
        low = bool(health) and health[0].class_name == "low"
        loading = None
        if want_loading_bar:
            loading = len(predict(self.models.loading_bar, frame)) > 0
        return Status(timestamp, spells, self.spell_slots.calibrated, low, loading)

    def grab(self) -> tuple[Any, tuple[int, int], float, float]:
        rect = self.window.rect()
        t0 = time.perf_counter()
        frame = self.capture.grab(rect)
        timestamp = time.monotonic()
        return frame, (rect.width, rect.height), timestamp, (time.perf_counter() - t0) * 1000

    def scan_scene(self) -> Scene:
        frame, size, ts, capture_ms = self.grab()
        return self.scene_from(frame, size, ts, capture_ms)

    def scan_status(self, want_loading_bar: bool) -> Status:
        frame, _, ts, _ = self.grab()
        return self.status_from(frame, ts, want_loading_bar)
