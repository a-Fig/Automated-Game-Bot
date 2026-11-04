# Fractured Ground: live minimap coordinates

Every point uses the live 400x360 minimap crop, window rectangle
`[1840,1040,2240,1400]` at 2240x1400. The original map was roughly eight pixels too
high in Y. This revision registers Tab terrain to the minimap visible in the same
captures and independently checks the Tab-closed player fixes. Navigation
calibration, visited.json and runtime arrow fixes remain unchanged.

- `map.png`: re-registered Tab image. Moving player icons are removed before
  combining captures; terrain behind them is inpainted, not known.
- `mask.png`: binary 400x360 walkability. Both flat ground levels are open. Their
  boundaries, dark rock blobs, grey patches and the exterior rim are blocked.
  Roads form corridors through boundaries (ramps). This remains a visual hypothesis
  requiring live validation, not ground-truth collision geometry.
- `mask_hires.png`: 1600x1440 planning grid, four cells per map pixel. The scaled
  grid API converts all points, distances, clearance, blocks and waypoints to/from
  the original minimap coordinates. mask.png remains a 400x360 summary. The fine
  grid preserves passages around small grey rocks that disappear when downsampled.
- `road.png`: road/ramp corridor, for debugging and route checks.
- `mask_overlay.png`: red = blocked, green = road/ramp, cyan = independent live
  fixes, white = NE route, magenta = portal approaches. Display coordinates are x4.
- `transform.json`: feature statistics, independent arrow errors, live/asset drift
  checks, parameters and route check. Fixes validate the fit; they never adjust it.
- `zone.json`: corrected portal positions, locator polygon and Tab transform.
  Its clearance default is 1.0; an explicit navigation.clearance overrides it.
- `override.png`: neutral grey. The old override mixed misregistered road colours
  with live visited discs and could reopen cliffs. Keep visited.json as evidence,
  not an instruction to paint every nearby cell open. Override support remains:
  0 = blocked, 255 = open, 128 = retain the generated mask. Disable antialiasing.

Rebuild from the original captures (reads images only, never game input):

```powershell
python tools/zone/tabmap_mask.py --captures "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_01.png" "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_02.png" "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_03.png" --out-dir assets/zones/fractured_ground --reset-override
```

Each adjacent JSON supplies minimap_before, minimap_after and window. Use --fixes
to name them separately. To reproduce the initial correction from old assets, add
`--reference tests/fixtures/tabmap/reference_old.png --metadata
tests/fixtures/tabmap/reference_old.json`. Already-live portal coordinates are
preserved on subsequent rebuilds, avoiding repeated re-projection.

Thresholds and widths are CLI parameters: --level-value 205, --ground-value 165,
--boundary-width 0.5, --dark-rock-value 160, --rock-max-area 10,
--water-saturation 75, --road-saturation 135, --road-value 210, --road-gap 4,
--road-half-width 2.8, --rim-margin 10 and --scale 4. Widths use map pixels, areas map pixels squared.
Road width includes planning clearance. Lower thresholds or wider roads can reopen
cliffs: inspect the overlay before live use. Poor feature fits, independent arrow
errors over 1.2px and asset/live drift over 1px are rejected. Drift means median
terrain displacement and drift at the matched terrain centre, not a similarity's
extrapolated translation at the off-map origin.

The route from (240,130) first recovers to a nearby clear cell, takes the plateau
ramp near (241,142), then follows the road to NE. Clearance 2.0 over-inflates fine
rocks and disconnects it; remove a legacy explicit clearance=2.0 to inherit 1.0.
Zones without their own clearance still default to 2.0.

Bushes, trees and mobs still require local avoidance. Unmarked ramps may be omitted.
Icon anchors have subpixel uncertainty; overrides remain useful for verified
corrections. The old low-resolution Zone.build_mask helper is disabled for these
assets: use full-resolution Tab captures with the generator.
