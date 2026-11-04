"""Read-only navigation previews and an explicitly invoked live calibration probe."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2

from .geometry import Point
from .navigation import MapLocator, NavigateTo, Zone, locate, solve_calibration


def live_capture(cfg):
    from .vision import Capture
    from .window import GameWindow
    window = GameWindow(cfg.window_title)
    rect = window.rect()
    if (rect.width, rect.height) != tuple(cfg.navigation.window_size):
        raise ValueError("window size does not match navigation calibration")
    capture = Capture(cfg.perception.capture_backend)
    return window, lambda: capture.grab(window.rect())


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def load_calibration(cfg, path):
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("zone") != cfg.zone or data.get("window_size") != list(cfg.window_size)
                or data.get("minimap_rect") != list(cfg.minimap_rect)):
            raise ValueError("calibration zone/window/crop does not match current configuration")
        cfg.minimap_to_screen = data["minimap_to_screen"]
        cfg.calibrated = bool(data.get("calibrated", False))
        if data.get("map_speed_px_per_s") is not None:
            cfg.map_speed_px_per_s = data["map_speed_px_per_s"]


def cmd_nav(cfg, args) -> int:
    if not args.dry_run:
        raise ValueError("nav is a preview command: supply --dry-run; use NavigateTo for task integration")
    nav = cfg.navigation
    if args.zone:
        nav.zone = args.zone
    load_calibration(nav, args.calibration)
    zone = Zone.load(nav.zone, nav)
    frame = load_image(args.image) if args.image else live_capture(cfg)[1]()
    task = NavigateTo(args.to, zone, nav, allow_uncalibrated=True)
    step = task.tick(frame, time.monotonic())
    if not nav.calibrated:
        print("WARNING: uncalibrated direction-only transform; proposed clicks are provisional.")
    print(json.dumps({"zone": zone.name, "goal": list(task.goal), "status": step.status,
                      "location": list(step.location) if step.location else None,
                      "waypoints": [list(p) for p in step.waypoints],
                      "proposed_click": list(step.click) if step.click else None,
                      "message": step.message}, indent=2))
    # One frame can only propose the next click. Never invent future positions.
    return 0 if step.status in ("move", "arrived") else 1


def _probe(cfg, vector, out):
    """One user-invoked click, then watch until movement settles. Never called by nav."""
    from .actuator import Input
    from .vision import read_health

    nav = cfg.navigation
    if not 250 <= math.hypot(*vector) <= 400:
        raise ValueError("probe screen distance must be 250 to 400 pixels")
    target = Point(nav.player_screen[0]+vector[0], nav.player_screen[1]+vector[1])
    x0, y0, x1, y1 = nav.safe_screen_rect
    if not (x0 <= target.x <= x1 and y0 <= target.y <= y1):
        raise ValueError("probe target lies outside safe_screen_rect")
    zone = Zone.load(nav.zone, nav)
    print("One calibration click in 3 seconds: focus the game now. Other drivers must be stopped.", flush=True)
    time.sleep(3)
    window, grab = live_capture(cfg)
    frame = grab()
    before = locate(frame, nav, polygon=zone.metadata["terrain_polygon"])
    health = read_health(frame, cfg)
    if before is None or health is None or health < cfg.health.low_threshold:
        raise ValueError("probe requires a visible arrow and readable healthy HUD")
    tracker = MapLocator(nav, zone.metadata["terrain_polygon"])
    started = time.monotonic()
    tracker.previous, tracker.timestamp = before, started
    previous, after, last_moved = before, before, started
    Input(window, cfg).click(target)  # explicit --execute is required by cmd_calibrate
    while time.monotonic()-started < 8:
        time.sleep(0.1)
        if not window.is_active():
            raise ValueError("focus lost during probe; sample discarded")
        frame = grab()
        current = tracker.locate(frame, time.monotonic())
        health = read_health(frame, cfg)
        if current is None or health is None or health < cfg.health.low_threshold:
            raise ValueError("lost arrow/healthy HUD during probe; sample discarded")
        after = current
        if current.point.dist(previous.point) > 0.4:
            previous, last_moved = current, time.monotonic()
        if after.point.dist(before.point) >= 1 and time.monotonic()-last_moved >= 0.8:
            break
    else:
        raise ValueError("probe did not move and settle within 8s; do not use it for calibration")
    sample = {"map_delta": [after.x-before.x, after.y-before.y], "screen_delta": list(vector),
              "duration": max(0.1, last_moved-started), "before": list(before), "after": list(after),
              "window_size": list(nav.window_size), "minimap_rect": list(nav.minimap_rect), "zone": nav.zone}
    Path(out).write_text(json.dumps(sample, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(sample, indent=2))


def cmd_calibrate(cfg, args) -> int:
    if args.samples:
        if args.probe or args.execute:
            raise ValueError("--samples cannot be combined with --probe/--execute")
        samples = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.samples]
        if any(s.get("window_size") != list(cfg.navigation.window_size)
               or s.get("minimap_rect") != list(cfg.navigation.minimap_rect)
               or s.get("zone") != cfg.navigation.zone for s in samples):
            raise ValueError("probe zone/window/crop does not match current configuration")
        result = solve_calibration(samples)
        result.update(zone=cfg.navigation.zone, window_size=list(cfg.navigation.window_size),
                      minimap_rect=list(cfg.navigation.minimap_rect))
        Path(args.out).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        print("[navigation]\ncalibrated = true\nminimap_to_screen = " + json.dumps(result["minimap_to_screen"]))
        if result["map_speed_px_per_s"] is not None:
            print(f'map_speed_px_per_s = {result["map_speed_px_per_s"]:.4f}')
        return 0
    if not args.probe or not args.execute:
        raise ValueError("live calibration needs --probe dx,dy --execute; it sends ONE world click")
    try:
        vector = tuple(float(v) for v in args.probe.split(","))
    except ValueError as e:
        raise ValueError("probe must be screen dx,dy") from e
    if len(vector) != 2 or not all(math.isfinite(v) for v in vector):
        raise ValueError("probe must be two finite screen coordinates")
    _probe(cfg, vector, args.out)
    return 0
