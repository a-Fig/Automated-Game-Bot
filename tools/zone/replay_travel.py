"""Read-only route audit of HH:MM:SS (x, y, 'safe') position logs. No capture/input."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from albot.geometry import Point
from albot.navigation import Zone


def seconds(clock):
    h, m, s = map(int, clock.split(":"))
    return h*3600+m*60+s


def read_positions(path):
    result, day = [], 0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"(\d\d:\d\d:\d\d) \(([-\d.]+), ([-\d.]+), '[^']+'\)", line.strip())
        if not match:
            if not line.strip() or line.strip().endswith("None"):
                continue
            raise ValueError(f"unrecognised position log line: {line}")
        clock, x, y = match.groups()
        at = seconds(clock)+day
        if result and at < result[-1][0]:
            day += 86400
            at += 86400
        point = Point(float(x), float(y))
        if not all(math.isfinite(v) for v in point):
            raise ValueError("non-finite position")
        result.append((at, clock, point))
    if not result:
        raise ValueError("position log has no fixes")
    return result


def replay(path, zone, goal="portal:NE", bot_log=None, stall_seconds=8.0):
    fixes = read_positions(path)
    cutoff, started, controller_stalled = None, None, None
    if bot_log:
        for line in Path(bot_log).read_text(encoding="utf-8-sig").splitlines():
            if started is None and "travel move:" in line:
                started = seconds(line[:8])
                while started < fixes[0][0]:
                    started += 86400
            if controller_stalled is None and "travel stalled:" in line:
                controller_stalled = line[:8]
            if "travel gave up" in line or "travel arrived" in line:
                cutoff = seconds(line[:8])
                while cutoff < fixes[0][0]:
                    cutoff += 86400
                break
    active = [fix for fix in fixes if (cutoff is None or fix[0] <= cutoff)
              and (started is None or fix[0] >= started)]
    target, plans, disagreements, stalls = zone.goal(goal), [], [], []
    for i, (at, clock, point) in enumerate(active):
        try:
            route = zone.grid.plan(point, target)
            next_point = next((p for p in route[1:] if p.dist(point) > 2), route[-1])
            plans.append({"time": clock, "position": list(point), "next_waypoint": list(next_point),
                          "safe": zone.grid.is_safe(point), "route": [list(p) for p in route]})
        except ValueError as e:
            plans.append({"time": clock, "position": list(point), "error": str(e)})
            continue
        if i+1 == len(active):
            continue
        end = active[i+1][2]
        actual, intended = Point(end.x-point.x, end.y-point.y), Point(next_point.x-point.x, next_point.y-point.y)
        length = math.hypot(*intended)
        sideways = abs(actual.x*intended.y-actual.y*intended.x)/max(length, 1e-9)
        away = actual.x*intended.x+actual.y*intended.y <= 0
        clear = zone.grid.line_clear(point, end)
        if point.dist(end) >= 0.8 and (sideways >= 2 or away or not clear):
            disagreements.append({"time": clock, "from": list(point), "to": list(end),
                                  "planned_next": list(next_point), "lateral_delta": round(sideways, 3),
                                  "away_from_waypoint": away, "sample_chord_clear": clear})
    # Stationary windows use actual displacement, not distance-to-goal. Samples
    # are sparse: neither the chord nor a stall establishes the exact cliff/ramp.
    i = 0
    while i < len(active):
        j = i+1
        while j < len(active) and active[j][2].dist(active[i][2]) < 0.8:
            j += 1
        if j-i > 1 and active[j-1][0]-active[i][0] >= stall_seconds:
            stalls.append({"from": active[i][1], "until": active[j-1][1],
                           "seconds": active[j-1][0]-active[i][0], "position": list(active[i][2]),
                           "planned_next": plans[i].get("next_waypoint")})
        i = j
    return {"zone": zone.name, "goal": list(target), "samples": len(fixes), "active_samples": len(active),
            "excluded_outside_travel": len(fixes)-len(active), "controller_stalled_at": controller_stalled,
            "plans": plans,
            "disagreements": disagreements, "stalls": stalls,
            "limitations": "Sparse samples are not commanded clicks or a continuous path. A blocked chord may be a local detour. Stalls constrain the failed approach, not the exact ramp location."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--bot-log", help="exclude samples after travel gave up or arrived")
    parser.add_argument("--zone", default="fractured_ground")
    parser.add_argument("--to", default="portal:NE")
    parser.add_argument("--out", help="optional JSON report; learned evidence is never written")
    args = parser.parse_args(argv)
    report = replay(args.log, Zone.load(args.zone), args.to, args.bot_log)
    text = json.dumps(report, indent=2)+"\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
