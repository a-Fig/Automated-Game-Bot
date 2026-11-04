# Offline map navigation

Navigation uses no Roboflow models or game input. `NavigateTo.tick(frame, now)`
returns a proposal, not an action. No navigation behavior is enabled in the farming
priority list. Its caller must check focus, health and freshness before dispatching
any proposal. NumPy frames are **BGR**, PIL images are **RGB**. Positions and all
path tolerances use minimap-crop pixels, not full-frame or physical world units.

## Preview a route

From the repo, using the pinned venv (substitute its full Python path if needed):

```powershell
python -m albot nav --to portal:NE --dry-run --image "C:\Users\smash\PycharmProjects\datasets\albion_v2\nav_samples\frame_minimap_1.png"
python -m albot nav --to 250,180 --dry-run --image "saved_frame.png"
python -m albot nav --to portal:NE --dry-run
```

The last command takes one live screenshot; it sends no input. Output includes
the current location, simplified route and the next proposed click. One screenshot
cannot predict subsequent player positions, so it does not invent a sequence of
future clicks. Until calibrated, previews warn that the identity matrix is only a
directional guess. A normal task refuses to propose clicks with an uncalibrated
matrix. A blocked/unreachable goal, missing arrow, ambiguous castle match, wrong
window size or Tab overlay produces no click.

## Live calibration (run by the game owner)

The world-click origin, minimap crop/zoom, UI scale and walkability are still
calibration assumptions. Use clear, safe ground and stop other bot drivers.
`nav-calibrate --execute` sends **one** left world click per probe. It never opens
maps or clicks the minimap. It rejects unreadable/low health, loss of focus or an
ambiguous arrow. The requested screen displacement is assumed to be completed;
a collision or partial move invalidates the scale estimate even if the math fits.

Use two independent directions, preferably repeat both, and visually verify that
each click reaches its intended destination. The routine watches the arrow until
it settles; map motion below one pixel is rejected. Collect probes in the same
zone, zoom and window configuration:

```powershell
python -m albot nav-calibrate --probe 350,0 --execute --out nav_east.json
python -m albot nav-calibrate --probe 0,350 --execute --out nav_south.json
python -m albot nav-calibrate --samples nav_east.json nav_south.json --out nav_calibration.json
python -m albot nav --to portal:NE --dry-run --calibration nav_calibration.json
```

The probe command waits three seconds for you to focus the game. The solve command
is offline and sends no input. It solves `screen_delta = matrix @ minimap_delta`
without assuming orientation, scale, reflection or axis alignment; collinear,
tiny and inconsistent samples are rejected. Two samples exactly determine four
coefficients, so **they cannot detect a consistently obstructed probe**. Three or
more samples provide a residual check. The output JSON includes the matrix and an
estimated minimap speed; the printed `[navigation]` entries can be copied into
`albot.toml`. `--calibration` also loads the JSON directly for previews.

## Task API

```python
import time
from albot.config import load
from albot.navigation import NavigateTo

cfg = load("albot.toml")
task = NavigateTo("portal:NE", config=cfg.navigation)  # or (250, 180)
step = task.tick(frame, time.monotonic())
# step.status: move, sidestep, waiting, arrived, lost, uncalibrated,
#              no_route, unsafe_click or stalled
# step.click: a window-relative Point, or None. It has not been dispatched.
```

An A* grid inflated by `clearance` avoids wall corners; line-of-sight simplification
keeps only safe turns. The controller follows the next segment up to `lookahead`,
clamps radial click length to 250-400 pixels and shortens the ray before entering
the configured HUD-free rectangle's edge. It never clamps x and y independently.
The minimum click radius can aim past a nearby waypoint; the dispatcher must
recheck frames frequently and replace movement before overshooting a turn. This
API does not claim that a single click's entire travel is safe to run open-loop.

If distance to the next waypoint does not decrease for `stuck_seconds`, it proposes
one perpendicular sidestep. After `side_step_seconds` it blocks the intended stuck
point (not the current player cell) in a private task mask and replans. After
`max_replans` it stops proposing clicks. Lost fixes do not count as stuck time.
Arrival means near the portal's **approach point**, not a completed zone transfer.

## Still needs live validation

- Crop, map zoom, arrow thresholds, world-click origin and safe screen rectangle at
  2240x1400; other UI scales require recalibration. A darkened or covered arrow is
  deliberately treated as missing. No claim of location behind overlays.
- Matrix and walking speed via the probes above; remeasure after changing zoom,
  movement mode or UI scaling. The movement speed is used for tracking gates, not
  for open-loop travel. Follow ticks need frequent fresh frames.
- Tan/road vs brown/blue walkability, wall clearance, lookahead and arrival radius.
  Paint the per-zone override mask after observing collisions. Colour alone cannot
  establish traversability or determine whether a grey patch is water vs rock.
- Portal approach points. NE is intended for Bridgewatch, but transition, banking
  and selling remain outside this task API. The configured zone is trusted; a
  caller must stop navigation on zone change or unexpected overlays.
