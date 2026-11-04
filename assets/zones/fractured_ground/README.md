# Fractured Ground map

All positions are pixels in the 400x360 minimap crop, window rectangle
`[1840,1040,2240,1400]` at 2240x1400. These assets are specific to the current UI
scale, minimap zoom and zone. There is no automatic zone/zoom detection.

- `map.png`: larger Tab-map image resampled into minimap coordinates. Four yellow
  portal markers fit the affine transform in `zone.json` within 0.5 minimap pixels.
  The exterior is blacked out; no HUD or account information is included.
- `mask.png`: white = provisionally walkable, black = blocked. Tan and orange road
  pixels are open; dark brown and grey-blue are blocked. **Terrain semantics are
  unverified**, not a substitute for live collision testing.
- `override.png`: paint **0 black to block**, **255 white to open**, **128 grey to
  retain the automatic mask**. Keep it grayscale, 400x360, with these exact values
  (disable antialiasing). The override is applied before wall clearance.
- `zone.json`: coordinate transform, colour thresholds, terrain boundary and portal
  markers. Portal `approach` points are inside the zone, not a command to transition.
  Only NE's destination (Bridgewatch) is known; other destinations are unverified.

To rebuild the automatic mask from a new saved minimap or Tab image:

```python
import cv2
from albot.navigation import Zone, WalkGrid
zone = Zone.load("fractured_ground")
mask = zone.build_mask(cv2.imread("tabmap_open_2.png"), kind="tab")
WalkGrid(mask).save(zone.folder / "mask.png")
# kind="minimap" takes a full 2240x1400 frame; kind="map" takes a 400x360 crop.
```

The recorded arrow is blue: RGB `(97,179,255)` corresponds to OpenCV HSV hue 104.
Hue around 10-16 results from interpreting RGB data as BGR. Both the castle and
compass contain blue pixels. The locator checks a small bright component's shape,
dark border, map bounds and continuity; ambiguity/overlays yield no location.
