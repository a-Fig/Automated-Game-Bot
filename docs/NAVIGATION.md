# Offline map navigation

Navigation uses no Roboflow models or game input. `NavigateTo.tick(frame, now)`
returns a proposal, not an action. The bot's optional `travel` behavior uses these
proposals through the normal guarded input path. Other callers must check focus, health and freshness before dispatching
any proposal. NumPy frames are **BGR**, PIL images are **RGB**. Positions and all
path tolerances use minimap-crop pixels, not full-frame or physical world units.

## Zone changes and exit buffers

The normal bot loop protects farming, chasing and travel with `[safety] zone_guard
= true`. At startup it keeps a copy of the first non-black zone-name strip in
memory. `zone_label_rect = [2008,1352,2146,1376]` excludes the changing player count
and clock and is calibrated at
`zone_label_size = [2240,1400]` and scales with capture dimensions. A mean absolute
RGB difference over `zone_label_threshold = 15` for `zone_label_seconds = 2` on
fresh captures latches a stop. A whole-frame mean below `zone_loading_mean = 15`
stops immediately, including at startup. This is visual change detection, not OCR;
start in the intended zone with its HUD visible. It cannot identify a wrong zone
that was already present when the process started.
The current label difference is logged at DEBUG about every 30 seconds.

The status capture checks loading before running models and retains the latch
even if Latest has since published another frame. Context logs CRITICAL
`zone changed`, cancels click-to-move, releases the bot's keys/mouse buttons, and
quits. Restoring the HUD, changing focus or pressing resume does not clear it;
restart explicitly after verifying the zone. Numeric health is not required for
zone detection, and the label guard stays active when exit projection is skipped.

`zone.json` has radius-8 no-go discs at each NW/NE/SE/SW marker and inner approach.
Targets are projected from the latest minimap fix using the inverse calibrated
matrix. Both the target and the intended straight segment are checked, including
farming, loot, combat chase, detours, wandering and spell aiming. Wander rejects
unsafe candidates and picks another direction; the shared input path checks again.
Only an active travel behavior explicitly targeting `portal:<name>` exempts that
portal's discs. A numeric goal near an exit does not grant an exemption.
The exemption allows the approach; crossing that portal still trips the zone guard.

When already inside a buffer, movement must strictly increase distance from both
the gate marker and its approach, monotonically along the intended segment.
Permitted outward escapes log INFO. This allows leaving a buffer without allowing
a click through the gate. Refused travel proposals are not learned as cliffs.

For farming configs without `[navigation]`, Fractured Ground supplies the measured
long-walk matrix `[[373.34,274.96],[-59.97,341.67]]` for exit projection only. It is
scoped to 2240x1400 and minimap crop `[1840,1040,2240,1400]`; it does not enable
uncalibrated travel. A different window/crop skips projection with one WARNING.
No trusted minimap fix also skips this check, as requested. Exit buffers remain
approximate: Albion's local pathfinding can bend away from the intended segment.

World targets also exclude `[safety] hud_rects`, scaled from `hud_size = [2240,1400]`.
The minimap/buttons, label/clock, portrait/buffs, menus and ability bar are excluded
before grouping model detections and again before any world click or spell aim.
Refusals log INFO; wandering samples another direction.

`minimap_recovery = true` checks the small panel crop without models. With readable
numeric health, five seconds of fresh frames without an arrow fix **and** without
the tan diamond frame propose one click on `minimap_toggle = [2197,1290]`, scaled
from `hud_size`. Visible parchment, weak frame evidence or a flat/black crop vetoes the click.
The input path stops movement, sends only this named UI action, logs WARNING
`minimap hidden; toggled`, and unwinds the interrupted behavior. Attempts are at
least 60 seconds apart, including after a recovered fix. If still hidden five
seconds after the third attempt, it cancels movement, releases input and latches
a run stop until restart. A restored arrow resets the failure count. Timing,
attempt count and panel/button coordinates are configurable in the example.
The frame geometry is calibrated on the two run-13 crops; other UI layouts need
new fixtures/calibration, even though the rectangles scale with window size.

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
keeps only safe turns. Away from roads/corners the controller follows the next
segment up to `lookahead`, using the legacy 250-400px radial range. On a road, or
within `near_waypoint_distance = 3` map pixels of a waypoint, it uses at most
`precision_click_max = 150` screen pixels with **no minimum extension**. A tiny
projected delta stays tiny instead of becoming a 250px click past the waypoint.
On roads it prefers a forward interior road cell within that short click, checking
the entire short segment against both road and learned-safe masks. Road waypoints
use at most 0.35px reach tolerance; no corner is skipped unless the next segment
is clear. Every projection shortens the ray before the HUD-free rectangle's edge
and preserves direction. Fresh-frame safety/combat checks still precede dispatch.
Albion may bend its actual trajectory; this is not an open-loop path guarantee.

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
but unused. Stopping travel for combat/healing also suspends both terrain timers,
including short interruptions. A saved edge that disconnects the goal is retained
and reported as no_route. The controller first reapplies radius-1 observed standing discs near the
road; if still disconnected, the log identifies a minimal set of edge indices
whose removal would restore the route. Those removals happen only in diagnostic
copies, never in the task or JSON. Existing discs are normally already open, so
the reported additional cell count is usually zero. If the base mask cannot route
even without edges, it reports the normal mask failure instead. The existing travel
give-up timer then clears the goal safely. A real cliff outranks a guessed road.
Arrival means near the portal's **approach point**, not a completed zone transfer.

After an actual dispatched walk, `zero_motion_pixels = 0.3` held for
`zero_motion_seconds = 3.0` triggers unreachable-target learning before the normal
stuck timer. Proposals and throttled re-clicks do not start/reset this clock;
movement in any direction, missing fixes, stopped travel and pauses reset it.
The failed direction receives a transverse edge with short wings at +/-45 degrees
(one logical fan record with three segments). Short wings avoid blocking behind
the player; standing-position clearance protection still applies. Targets too close
to place a safe barrier are recorded without blocking the player's own cell.
The two-stall confirmation rule still applies if any part of a fan crosses a road.
Before activating a **new** road-crossing edge, the controller checks whether it
would disconnect the current player from the goal. If so it logs the conflict and
keeps that candidate in `road_stalls` with `deferred_reason`/`deferred_goal`; the
candidate is not applied as a wall. Road movement already uses short probes. After
`road_retry_attempts = 2` failed probes at that spot, travel hands off to bounded
breadcrumb backtracking rather than repeating indefinitely. A road edge with a
valid alternative may still be activated after two stalls. Non-road edges retain
their previous policy. Existing saved edges are never silently removed or crossed.

Run 15's saved barrier is an example: at the time of inspection, index 19 was
from `(255.66,136.08)` toward `(271.75,124.25)` with endpoints
`(256.30,131.88)`–`(259.85,136.72)`. The copied local fixture reproduces its cut;
removing it from that **copy** restores the road route. Short-click simulation
from both the stall and best position stays on the road and reaches the approach.
The 11:01:07/11:01:11 stalls occurred after the last logged fight at 11:00:42 and
before the next HP-drop retreat at 11:01:27; the log alone cannot establish their
physical cause. This change does not infer elevations or open the plateau north
of the road. Road appearance and projection still need live validation.

At the per-spot retry limit, or when learned cliffs leave no route, travel attempts
to backtrack along its actual journey breadcrumbs. It sends guarded right-click
walks, without boots, and yields to combat/health retreat. On reaching at least
`backtrack_distance = 3.0` map pixels from the stalled spot it trims the abandoned
trail portion, clears that spot's retry counter, and replans with all learned edges
retained. At most `backtrack_attempts = 2` escapes are allowed per spot per journey.
No recorded point at least 3px away means no invented escape; the journey gives up.
An escape also stops after `backtrack_seconds = 20.0` of active walking time. Combat
or focus interruptions do not consume that timeout. This assumes that the walked
approach is reversible; live validation is still needed on plateau entrances.

Each journey emits one `travel summary` line on arrival, cancellation, goal change,
give-up or run/safety stop: goal, start, best observed position, progress, remaining
distance, new active edge count and outcome. Progress is the reduction in straight
map distance to the goal, not total path length. Pending road stalls are not counted
as active edges; a three-segment fan counts as one learned record.

## Learned travel evidence

Real travel saves `assets/zones/<zone>/learned.json` next to its zone assets (or in
the custom assets folder). This file and its atomic-write temporary are gitignored;
merges and mask regeneration leave it alone. Zone.load applies it after the base
mask/painted override, before planning. Rename the file to reset learning, or remove
individual `edges` entries to retract a mistaken collision. Keep a copy when
changing map coordinates, zoom or UI calibration; the loader checks zone, size,
scale and coordinate-space tags, but cannot detect an unreported coordinate shift.

`Zone.load(..., learned=False)` loads generated assets alone for isolated fixtures;
the default continues to load live evidence. Tests forbid opening the repo's runtime file.

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
Call `note_walk(now)` after each actual forward walk dispatch to arm fast zero-motion
detection; do not call it for a proposal that was skipped or failed to dispatch.
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
