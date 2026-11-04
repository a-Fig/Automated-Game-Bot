# Tab-map registration fixtures

The PNGs are unscaled crops of fg_01/fg_02/fg_03, window rectangle
[300,230,1950,1400]. image_origin retains full-window coordinates. The separate
400x360 minimap crops are from the same captures. Before/after fixes were measured
independently with Tab closed and define runtime coordinates.

reference_old.png/json preserve the previous registration so the eight-pixel Y
error cannot be hidden by rebuilding. Expected live fixes: (240.87,130.51),
(239.86,130.30), (237.71,130.50). No fixture requires the game or sends input.
