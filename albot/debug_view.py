"""
Drawing what the bot sees: detections, obstacles, the player, and the current plan.

Used live (an OpenCV window, if debug.view is on) and by replay mode (saved images).
"""
from __future__ import annotations

import logging
import threading

from .behaviors import Plan
from .state import Scene

log = logging.getLogger(__name__)

# BGR colours
COLOURS = {
    "mob": (0, 140, 255), "threat": (0, 0, 255), "farm": (0, 200, 0),
    "pickup": (255, 200, 0), "wisp": (255, 0, 255), "other": (180, 180, 180),
    "obstacle": (60, 60, 200), "player": (255, 255, 255), "plan": (0, 255, 255),
}


def annotate(frame, scene: Scene, plan: Plan | None, header: str = ""):
    """Returns a copy of the frame (numpy BGR) with everything drawn on it."""
    import cv2
    import numpy as np
    img = np.ascontiguousarray(frame).copy()

    groups = {}
    for name, items in (("mob", scene.mobs), ("threat", scene.threats), ("farm", scene.farmables),
                        ("pickup", scene.pickups), ("wisp", scene.wisps)):
        for d in items:
            groups[id(d)] = name

    for poly in scene.obstacles:
        pts = np.array([[int(p.x), int(p.y)] for p in poly.points], dtype=np.int32)
        cv2.polylines(img, [pts], True, COLOURS["obstacle"], 2)

    for d in scene.detections:
        colour = COLOURS[groups.get(id(d), "other")]
        x0, y0 = int(d.center.x - d.width / 2), int(d.center.y - d.height / 2)
        x1, y1 = int(d.center.x + d.width / 2), int(d.center.y + d.height / 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(img, f"{d.class_name} {d.confidence:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    px, py = int(scene.player.x), int(scene.player.y)
    cv2.circle(img, (px, py), 6, COLOURS["player"], -1 if scene.player_detected else 2)
    radius = int(scene.window_size[1] * 0.3)
    cv2.circle(img, (px, py), radius, COLOURS["player"], 1)

    if plan is not None:
        goal = plan.target.center if plan.target else plan.point
        if plan.waypoint is not None:
            w = (int(plan.waypoint.x), int(plan.waypoint.y))
            cv2.line(img, (px, py), w, COLOURS["plan"], 2)
            if goal is not None:
                cv2.line(img, w, (int(goal.x), int(goal.y)), COLOURS["plan"], 1)
        elif goal is not None:
            cv2.line(img, (px, py), (int(goal.x), int(goal.y)), COLOURS["plan"], 2)

    text = header or (plan.description if plan else "idle")
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class LiveView:
    """Shows the newest annotated scene in a window. Needs debug.view so frames are kept."""

    def __init__(self, workers, ctx) -> None:
        self.workers = workers
        self.ctx = ctx
        self._thread = threading.Thread(target=self._loop, name="debug-view", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        import cv2
        last = 0.0
        while not self.workers.stopping.is_set():
            scene = self.workers.scenes.wait_newer(last, 1.0)
            if scene is None or scene.frame is None:
                continue
            last = scene.timestamp
            try:
                import numpy as np
                frame = scene.frame if not hasattr(scene.frame, "save") else np.asarray(scene.frame)[:, :, ::-1]
                cv2.imshow("albot debug", annotate(frame, scene, self.ctx.current_plan))
                cv2.waitKey(1)
            except Exception:
                log.exception("debug view failed; turning it off")
                return
