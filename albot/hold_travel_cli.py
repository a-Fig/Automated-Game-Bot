"""`albot travel-hold`: walk to a goal by holding the move button with the Tab map open."""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np

from .geometry import Point
from .hold_travel import HoldConfig, HoldSteer, Trails, map_to_tab, plan_route, road_landing_score, tab_arrow, tab_to_map
from .safety import ZoneGuard

log = logging.getLogger(__name__)


def run_hold(cfg, nav, route, grab, inp, max_seconds, *, health=None, clock=time.monotonic,
             sleep=time.sleep, hold_config=None, min_health=0.6, trail=None, goal=None):
    """Hold, open the zone map, steer until arrival or a stop. Always releases.

    Returns (status, message). status: arrived, stalled, zone_changed,
    low_health, arrow_lost or time_limit. Every map fix is appended to `trail`.
    """
    zone = nav.zone
    transform = zone.metadata["tab_to_map"]
    icons = [np.asarray(p, float) for p in zone.metadata.get("tab_static_icons", [])]
    matrix = cfg.navigation.minimap_to_screen
    steer = HoldSteer(route, goal or nav.goal, matrix, cfg.navigation.player_screen, hold_config,
                      road_landing_score(nav, matrix))
    first = steer.update(route[0], clock())
    if first.cursor is None:
        return first.status, first.message
    guard = ZoneGuard(cfg)
    held = tab_open = False
    status, message = "time_limit", f"no arrival within {max_seconds:.0f}s"
    try:
        inp.hold_move(first.cursor)
        held = True
        sleep(0.25)
        inp.press(cfg.keys.zone_map)
        tab_open = True
        sleep(0.35)
        # Seed the tracker with the minimap fix so a second blue icon is never chosen.
        start, previous, misses, last = clock(), map_to_tab(transform, route[0]), 0, ""
        while clock()-start < max_seconds:
            frame, now = grab(), clock()
            reason = guard.observe(SimpleNamespace(timestamp=now, frame=frame), now)
            if reason:
                status, message, tab_open = "zone_changed", reason, False  # the map closes with the zone
                break
            hp = health(frame) if health is not None else None
            if hp is not None and hp < min_health:
                status, message = "low_health", f"health {hp:.2f}"
                break
            arrow = tab_arrow(frame, previous, icons)
            if arrow is None:
                misses += 1
                if misses >= 8:
                    status, message = "arrow_lost", "no player arrow on the zone map for 8 frames"
                    break
                sleep(0.15)
                continue
            misses, previous = 0, arrow
            point = tab_to_map(transform, arrow)
            if trail is not None:
                trail.append(point)
            step = steer.update(point, now)
            if step.status in ("arrived", "stalled"):
                status, message = step.status, step.message
                break
            if step.status != last or step.message.startswith("stall"):
                log.info("travel-hold %s at %.2f,%.2f: %s", step.status, point.x, point.y, step.message)
            last = step.status
            inp.steer(step.cursor)
            sleep(0.15)
    finally:
        if held:
            inp.release_move()
        if tab_open:
            sleep(0.2)
            try:
                inp.press(cfg.keys.zone_map)
            except Exception:  # focus lost: the map stays open, the button is already up
                log.warning("travel-hold: could not close the zone map")
    log.info("travel-hold %s: %s", status, message)
    return status, message


def cmd_travel_hold(cfg, args):
    from .actuator import Input
    from .navigation import NavigateTo
    from .navigation_cli import live_capture
    from .vision import read_health
    nav = NavigateTo(args.to, config=cfg.navigation)
    window, grab = live_capture(cfg)
    fix = nav.locator.locate(grab(), time.monotonic())
    if fix is None:
        raise ValueError("no minimap fix: is the minimap visible and the zone map closed?")
    trails = Trails(nav.zone.folder / "hold_trails.json")
    route, source = trails.route(fix.point, nav.goal), "saved trail"
    if route is None:
        partial = trails.route_toward(fix.point, nav.goal)
        if partial is not None:
            route, source = partial+plan_route(nav, partial[-1])[1:], "saved trail, then planner"
        else:
            route, source = plan_route(nav, fix.point), "planner"
    log.info("travel-hold (%s) from %.1f,%.1f to %.1f,%.1f via %s", source, *fix.point, *nav.goal,
             " ".join(f"{p.x:.1f},{p.y:.1f}" for p in route[1:]))
    if not args.execute:
        log.info("plan only; add --execute to walk")
        return 0
    walked = []
    # A saved trail is ground truth: follow it closely instead of cutting corners.
    hold_config = HoldConfig(lookahead=1.2) if source.startswith("saved trail") else None
    status, _ = run_hold(cfg, nav, route, grab, Input(window, cfg), args.seconds,
                         health=lambda frame: read_health(frame, cfg), trail=walked, hold_config=hold_config)
    if status in ("arrived", "zone_changed") and trails.add(walked, when=time.strftime("%Y-%m-%d %H:%M")):
        log.info("travel-hold saved a %d-fix trail to %s", len(walked), trails.path)
    if status == "arrived" and args.to.startswith("portal:"):
        # The goal is the approach. The exit trigger is further along the gate
        # passage: the measured "exit" point (live 2026-11-04: the NE trigger is
        # east of the approach, and walking at the signpost hits the gate wall).
        portal = nav.zone.metadata["portals"][args.to.split(":", 1)[1]]
        exit_point = Point(*portal.get("exit", portal["marker"]))
        here = walked[-1] if walked else nav.goal
        log.info("travel-hold: at the portal approach; walking into the exit toward %.1f,%.1f", *exit_point)
        status, _ = run_hold(cfg, nav, [here, exit_point], grab, Input(window, cfg), 15,
                             health=lambda frame: read_health(frame, cfg), goal=exit_point,
                             hold_config=HoldConfig(arrival=0.3))
    return 0 if status in ("arrived", "zone_changed") else 2
