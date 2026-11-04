# Offline map navigation

Navigation uses no Roboflow models or game input. `NavigateTo.tick(frame, now)`
returns a proposal, not an action. The bot's optional `travel` behavior uses these
proposals through the normal guarded input path. Other callers must check focus, health and freshness before dispatching
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

For live bot travel, use the existing bot loop rather than a standalone click loop:

```toml
[behavior]
travel_goal = "portal:NE" # or "250,180"; empty (the default) disables travel
```

Copy the measured navigation matrix/speed into `[navigation]` and set
`calibrated = true` first. Travel follows shield, heal and fight_close, before
pickup/farm/wander. Old priority lists receive travel after fight_close when a
goal is enabled; configurations missing/disabled combat priorities are rejected.
Each tick handles one fresh status frame and rechecks combat before dispatch.
Arrival logs and clears the goal in memory, sending no further input that tick.
It does not rewrite the TOML file; clear the file's goal to prevent reuse on restart.

An HP drop greater than 3% within 2 seconds outside an active fight signals an
attacker. If a mob is detected, shield/threat handling and fight_close engage the
nearest mob even without a red label or the usual close-range restriction.
If no mob is detected, retreat takes priority: the bot backtracks actual visited
journey positions until HP has stopped falling for 4 seconds. Ready boots are
aimed along the retreat direction and cast once, with a local cooldown guard.
A later attack within 3 minimap pixels of an earlier attack blocks a radius-3
area in this journey's private grid; travel replans from the retreat endpoint.
It keeps backtracking until outside the hazard's inflated clearance as well as
four seconds of stable HP, so it can replan from an unblocked cell.

Fresh HP below 25% starts a critical retreat, including during combat waits. It
backtracks and dashes for up to 8 seconds, then quits, even if HP recovers. Very low
positive HP does not latch the minimum-health death heuristic during this escape;
zero-health detection remains active. No scene detections are required to retreat
from fresh status/minimap frames. A detected mob can preempt an ordinary retreat;
critical retreat takes priority over combat. These guards are off when travel is
disabled. All thresholds are in `[safety]` and breadcrumb/hazard distances are in
`[navigation]`.

Only actually visited positions are retreat targets. At the oldest breadcrumb it
stops further movement and waits for calm/the critical deadline. If travel has not
yet recorded a route, it logs that no escape direction is known; it cannot invent
a safe path. Missing/ambiguous minimap fixes or uncalibrated navigation also prevent
escape clicks. Fresh numeric HUD health is required even with model fallback on;
missing/stale numeric health cancels movement. The critical deadline still expires
after 8 seconds if perception is lost, so retreat never runs indefinitely.

The proposal-only API remains available for offline use:

```python
import time
from albot.config import load
from albot.navigation import NavigateTo

cfg = load("albot.toml")
task = NavigateTo("portal:NE", config=cfg.navigation)  # or (250, 180); read-only by default
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

Clearance inherits `zone.json` when omitted from `[navigation]`. Fractured Ground's
Tab-derived mask uses 1.0 minimap pixel; zones without a value still use 2.0. An
explicit `navigation.clearance` always wins, including an old explicit 2.0. Remove
that key or set it to 1.0 to use the detailed mask. See the zone asset README for
the saved-image generator, registered overlays and the independent drift checks.
Fractured Ground plans on the 4x Tab grid; every returned waypoint and controller
distance still uses the original 400x360 minimap coordinate space. Painted overrides
remain 400x360 and are expanded with nearest-neighbour sampling before clearance.

If actual displacement stays below `progress_pixels` for `stuck_seconds`, it records
a suspected collision edge and replans without a short sidestep. A curved local
click-to-move trajectory counts as progress even when it moves away from the next
waypoint. `max_replans` now limits retries **per spot** (`stuck_block_radius` groups
nearby stalls), rather than sharing a small budget across the journey. The default
is 8; an old explicit 3 still means 3 per spot. Lost fixes and pauses over two
seconds do not count as stuck time. `side_step_seconds` is accepted for old TOMLs
but unused. A learned edge that disconnects the goal is retained and reported as
no_route. The controller first reapplies radius-1 observed standing discs near the
road; if still disconnected, the log identifies a minimal set of edge indices
whose removal would restore the route. Those removals happen only in diagnostic
copies, never in the task or JSON. Existing discs are normally already open, so
the reported additional cell count is usually zero. If the base mask cannot route
even without edges, it reports the normal mask failure instead. The existing travel
give-up timer then clears the goal safely. A real cliff outranks a guessed road.
Arrival means near the portal's **approach point**, not a completed zone transfer.

## Learned travel evidence

Real travel saves `assets/zones/<zone>/learned.json` next to its zone assets (or in
the custom assets folder). This file and its atomic-write temporary are gitignored;
merges and mask regeneration leave it alone. Zone.load applies it after the base
mask/painted override, before planning. Rename the file to reset learning, or remove
individual `edges` entries to retract a mistaken collision. Keep a copy when
changing map coordinates, zoom or UI calibration; the loader checks zone, size,
scale and coordinate-space tags, but cannot detect an unreported coordinate shift.

Every trusted arrow fix obtained while travel/retreat is moving contributes a
radius-1-map-px observed-position disc, using the mask builder's same function.
Discs are deduplicated by native cell (0.25 map px on Fractured Ground); repeated
stationary readings do not grow the file. It saves when a new disc or stall is
recorded, by replacing the JSON atomically. No straight-line corridor is inferred
between fixes: Albion can walk around objects between captures.

Each stall stores the player position, attempted waypoint, navigation timestamp
and endpoints of a short transverse segment between them. `learned_edge_length`
defaults to 6 map px; the segment is one native cell thick, not a blocked disc.
It is placed ahead by at least clearance plus two native cells. Every stalled
player position and recorded standing fix has a clearance halo cut out of inferred
edge blocks, both immediately and on reload, so a guessed edge cannot imprison the
player. A waypoint too close for safe placement is not blocked. A later standing
fix contradicting an edge opens its local disc and protects that cell; it does not
prove that the whole surrounding cliff is passable. Combat danger blocks stay
separate, temporary, and survive terrain replans.

An edge crossing road.png needs two stalls within 1.5 map pixels before becoming
active. The first is saved as an unapplied `road_stalls` entry so a repeat on the
next run can confirm it. Existing active edges remain authoritative on load;
legacy evidence is not discarded for lacking a confirmation field. Both pending
and active nearby stall origins count toward confirmation. Confirmed road edges
are kept even when they disconnect a route: the road mask may be wrong.

Normal `nav --dry-run`, dry-run bot input and replay never write learned evidence.
Standalone guarded dispatchers can opt into `NavigateTo(..., learn=True)` and call
`tick(frame, now, moving=True)` only while previously dispatched movement is active.
`record_position(location)` is available for their retreat path. Merely proposing
a click is not evidence that movement occurred. The integrated travel behavior
sets these flags through its existing input/safety path.

List or explicitly retract evidence without opening the game:

```powershell
python -m albot nav-edges list --zone fractured_ground
python -m albot nav-edges retract --zone fractured_ground --index 7
python -m albot nav-edges retract --zone fractured_ground --near 238.57,143.05 --radius 1.5
python -m albot nav-edges list --file "copied_learned.json"
```

Indices are zero-based and renumber after retraction. `--near` matches either the
stall origin or the segment. Retraction also clears associated pending road stalls;
standing positions are kept. `--file` targets a copied JSON instead of the zone's
runtime file. Stop travel before retracting: a running task keeps its own snapshot
and can overwrite a concurrent edit. List and preview are read-only. Retraction
uses the same atomic writer as travel and requires an explicit index/near selector.

## Offline position-log audit

```powershell
python tools/zone/replay_travel.py "C:\Users\smash\PycharmProjects\datasets\albion_v2\travel_runs\travel6.log" --bot-log "C:\Users\smash\PycharmProjects\datasets\albion_v2\travel_runs\run6.log" --to portal:NE --out replay_out/travel6.json
```

This reads logs and the zone only; it never captures or sends input or writes
learned evidence. Omit `--out` to print JSON without saving a report.
For each fix it prints a fresh mask plan and flags sideways/away movement and
sample chords rejected by the mask. `--bot-log` excludes samples before travel's
first move and after arrival/give-up, and reports the controller's stalled time.
Sparse sample chords are not measured paths or the actual commanded clicks.

The saved `docs/travel6_replay.json` audit, using the pre-fg_04 mask, shows the terminal stationary span from
04:28:31 to 04:29:02 near (238.3,136.2), where fresh plans still aim at
(239.75,140.75). The controller had already stalled at 04:28:35 and gave up at
04:29:05, so the whole 31-second span was not active walking. Earlier fixes bend
east/west while the planner keeps aiming south toward the ramp. These data refute
that direct approach as reliable; they do not locate the true ramp or establish
that every alternative road approach is blocked. Live learning will preserve
failed approaches instead of repeating the same guessed corridor on restart.

## Still needs live validation

- Crop, map zoom, arrow thresholds, world-click origin and safe screen rectangle at
  2240x1400; other UI scales require recalibration. A darkened or covered arrow is
  deliberately treated as missing. No claim of location behind overlays.
- Matrix and walking speed via the probes above; remeasure after changing zoom,
  movement mode or UI scaling. The movement speed is used for tracking gates, not
  for open-loop travel. Follow ticks need frequent fresh frames.
- Flat tan and brown ground levels, cliff boundaries, road ramps and grey patches;
  wall clearance, lookahead and arrival radius. Paint verified override corrections
  after observing collisions. Colour alone cannot establish actual traversability.
- Portal approach points. NE is intended for Bridgewatch, but transition, banking
  and selling remain outside this task API. The configured zone is trusted; a
  caller must stop navigation on zone change or unexpected overlays.
