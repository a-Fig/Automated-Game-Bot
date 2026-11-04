# Light fixes (branch `claude/clever-hamilton-nt7r40`)

Written 2026-11-04 by Claude, working only from the source code. None of this has been
run against the game: there was no Albion client, no Windows machine and no Roboflow key
in the environment where it was written. The structure, the behaviour order in `main.py`
and all of the tuned timings are unchanged on purpose, so this branch can be compared
line by line with the original.

## Bugs fixed

**Spell slots could shift (`spell_slots.py`, `objectFinder.create_spell_image`).**
The old code sorted spell detections by x and padded the end of the list with
`cooldown`. When the model missed a slot in the middle, every spell to its right
moved down one index, so `spells[4]` (shield, D) could be reading F. Now, whenever
the model returns exactly six detections, their x positions are recorded as the slot
positions, and every later detection snaps to its nearest slot. A slot with nothing
on it reads as `cooldown`. Before the first six-detection frame, all slots read as
`cooldown`, so the bot holds its spells until it has a layout. Calibration repeats on
every full frame, so a change of UI scale corrects itself.

**Possible `IndexError` in the main loop (`main.py`).** The close-mob check read the
latest scan twice in one expression. If a scanner thread appended a mob-free scan
between the two reads, `mobs[0]` raised. The list is now read once. This is rare
(the gap is microseconds), and it is the only place in the code where this pattern
can crash. The rest of the code reading the newest scan each time is deliberate and
unchanged.

**Obstacle outlines had an open side (`obstacle_instance_detection.py`).** The
intersection check never tested the edge from the last outline point back to the
first, so a path through that edge counted as reachable. `tests/test_fixes.py` has a
test that fails on the old code and passes now.

**Wander point check (`player_actions.wander`).** The retry loop ran while a point
was "blocked AND far", which accepted blocked points close by and clear points right
next to the player. It now retries while a point is "blocked OR too close", and gives
up after 50 tries instead of looping forever when no clear point exists. This is a
reading of intent: if the old condition was meant to work some other way, this is the
change to look at.

**Scanner threads died silently (`multiprocessing_class.repeated_execution`).** An
exception in a scan (a bad frame, a model error) ended that thread. The main loop
kept running on the last scan it produced, so the bot acted on a frozen picture. The
error is now printed and the scan retried.

**Out-of-order scans (`objectFinder.general_scanner`).** Two general scanner threads
run at once, so a slow scan could finish after a newer one and become "latest". A
scan is now only appended if it is newer than the current latest.

**Pause and resume could get out of step (`globals.py`, `helper.pause`).** Pausing
used `switch()`, which flips the current state. Any extra flip left the scanners in
the opposite state from the one the main loop assumed. It now calls explicit
`pause_scanners()` and `resume_scanners()`.

**Healing never waited (`player_actions.leash_and_heal`).** `danger()` returns True
whenever health is low, so `while LOW_HEALTH and not danger()` never ran its body, and
`wait_unless(5, danger)` returned at once. After the leash, the bot went straight back
to work while still low. `danger()` now takes `count_low_health`, which the heal wait
sets to False, so it waits for health to recover unless a real threat shows up. The
wait is capped at 60 seconds.

**Farming could hang (`player_actions.farm`).** The loading-bar wait had no time
limit, so a false loading-bar detection held the bot in place forever. It now gives
up after 30 seconds and sleeps briefly between checks.

## Smaller changes

- The image stacks are `deque(maxlen=50)`. The manual reset every 300 loops in
  `main.py` is gone, because the stacks can no longer grow without limit.
- Roboflow keys are read from the `ROBOFLOW_KEY1` and `ROBOFLOW_KEY2` environment
  variables, so they don't need to be in the source.
- Renamed `sheild` to `shield` and `wonder` to `wander`, and fixed two typos in
  comments and messages.
- Removed code nothing called: the deprecated scan loops, `parsedScannedImage`,
  `ScannedImage`, `farmableOb` (which also compared confidence to `55` instead of
  `.55`), `AutoObstacleImage`, `constantObstacleScanning`, the commented-out
  `ProcessHandler`, and unused imports. The other weapon rotations (`longbow`,
  `mistpiercer`, `nature`) and helpers like `attemptArmorCast` were kept, since they
  look like options to switch to, not leftovers.

## Tests

`python -m pytest tests` runs the slot mapping and obstacle intersection tests. They
stub out the game modules, so they run on any machine.
