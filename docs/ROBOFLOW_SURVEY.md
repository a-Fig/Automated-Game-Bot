# Roboflow account survey

Surveyed 2026-11-04 through the Roboflow REST API (read-only: project/version metadata and the image-search endpoint; nothing uploaded, labelled, trained or changed). API keys are not recorded here. Raw responses are in `C:\Users\smash\.t3\scratch\2026-11-04-handoff-df2y4o-md-t3-context-1ece968d\rf\raw.json`, `search_counts.json`.

## 1. Accounts

| Key | Workspace url | Members | Projects |
|---|---|---|---|
| `ROBOFLOW_KEY1` | `albiondetector` | 3 | 8 |
| `ROBOFLOW_KEY2` | `albion-edu-workspace` | 2 | 10 |

Two different workspaces, as suspected. Plan and remaining training credits are **not exposed** by the key-based REST API (`/usage`, `/billing`, `/plan` are treated as project names and return junk or 404). Check them in the web app (Settings > Plan). Every project is `public: true` (CC BY 4.0), which usually means the free Public plan, and the datasets are world-readable. Training service seen on the old runs: GCP Tesla T4.

## 2. Models the bot uses

All are in `albot/config.py` (`ModelConfig`). Detection thresholds: `min_confidence = 0.65` for every class, 0.5 for obstacles.

| Model | Workspace | Type | Trained | Architecture | mAP50 / P / R | Notes |
|---|---|---|---|---|---|---|
| `albion_detector-l4o1a/5` | edu (key 2) | object detection, 11 classes | 2024-11-10 | YOLOv8n "Roboflow 3.0 Fast", fine-tuned from v4, stopped after 16 of 300 epochs | 84.99 / 89.78 / 79.89 (valid); test 87.7 / 93.5 / 82.4 | Resize 920x920 fit-with-black-edges, 13,299 images (12,306 train) |
| `obstacle_instance_detection/5` | edu (key 2) | instance segmentation, 1 class (`obstacle`) | 2024-11-09 | YOLOv8n-seg, 118 epochs | mask mAP50 66.35 (valid), 76.5 test; P 78.0 / R 63.2 (valid) | Weakest model in the stack |
| `spell_detector/4` | albiondetector (key 1) | object detection, `up` / `cooldown` | 2024-11-09 | YOLOv8n, 82 epochs | 97.36 / 95.48 / 95.26 (valid); test 98.5 / 98.1 / 97.5 | Good |
| `loading_bar/1` | edu (key 2) | object detection, `loading_bar` (+ empty `gathering_bar`) | 2024-10-31 | YOLOv8n, 227 epochs | 99.50 / 99.21 / 100 | Valid set has only 7 boxes, test 5, so the score says little |
| `low_detector/1` | edu (key 2) | classification (`low` / `good`), ViT-base | 2024-11-09 | ViT-B/16, 300 epochs | accuracy 1.0 | 46 images in total (224 train after 7x augmentation of 32); 5 test images |

All of the bot's models are the newest trained version of their project, except `spell_detector`: v5 (2025-04-10, 1,543 images, mAP 97.90 / P 95.56 / R 96.23) exists and was never wired in, and v6 (2026-11-03) was just trained with RF-DETR NAS (see section 5).

Dataset export: yes. `albion_detector-l4o1a/5` offers `yolov8`, `yolov11`, `yolo26`, `yolov5pytorch`, `yolov7pytorch`, `yolov5-obb`, `coco`, `openai`, `benchmarker`. Both the `yolov8` and `yolov11` exports are already built and are **977.4 MB** each (13,299 augmented images). That is over the 200 MB cut-off, so **nothing was downloaded** (C: has 17.7 GB free, so disk was not the limit). A smaller export is possible: the un-augmented source set is 2,768 images and could be exported by making a new version with no augmentation, but that would create a version, which this survey was told not to do.

Image sizes in `albion_detector-l4o1a`: 2,622 are 1920x1080, 509 are 2560x1440, plus a few odd small sizes (1366x768, 1360x768, 1024x768, 1152x864, ...). Training resizes to 920x920 with black bars, so a 1920x1080 frame is scaled to 0.48 and a 2560x1440 frame to 0.36 before the network sees it.

## 3. Class inventory of the scene detector (`albion_detector-l4o1a`)

The project-level class counts in the metadata are stale (ore, silver, hide and mob show `null`); the numbers below were counted by paging the search API over every image. 3,203 images are in the project (2,768 are in the dataset splits, 447 have no split).

| Class | Boxes | Images with it | v5 valid P / R | v5 test P / R | Bot use |
|---|---|---|---|---|---|
| player | 3,812 | 2,775 | .91 / .92 | .93 / .94 | player position |
| mob | 3,632 | 1,952 | .87 / .72 | .85 / .73 | `mob_classes` |
| fiber | 2,216 | 1,095 | .95 / .93 | .96 / .95 | `farm_classes` |
| hide | 1,423 | 682 | .81 / .80 | .87 / .83 | `farm_classes` |
| ore | 756 | 471 | .85 / .79 | .93 / .80 | none |
| tree | 398 | 261 | .89 / .84 | .95 / .92 | none |
| stone | 371 | 239 | .88 / .81 | .94 / .60 | none |
| red | 295 | 207 | .93 / .61 | .94 / .70 | `threat_classes` |
| silver | 275 | 232 | .96 / .70 | 1.0 / .80 | `pickup_classes` |
| cup of piss | 62 | 62 | 1.0 / .88 | 1.0 / .94 | `pickup_classes` |
| wisp caged | 49 | 48 | .84 / .80 | .92 / .88 | `wisp_classes` |

Only 389 of the 3,203 images (12%) contain no annotation at all, so the set has few pure background frames.

Splits are by image: 1,762 train / 601 valid / 393 test (+ 447 unassigned). The 117 images added after v5 was generated (91 mob, 92 fiber, 79 hide, 21 ore boxes) are not in the trained model. All images were uploaded in Oct-Nov 2024; nothing has been added since.

## 4. What could explain the in-game problems

These are inferences from the metadata and the bot config. The API does not show images, so nothing here has been checked by eye; section 6 lists what to look at.

**Gathering nodes that were already gathered: there is no class for it.** The 11 classes are cup of piss, fiber, hide, mob, ore, player, red, silver, stone, tree, wisp caged. Nothing like `fiber_depleted`, `used`, `empty` or `stump`. The bot's `farm_classes` are `hide` and `fiber`, with no state attached to a detection. If a harvested node or skinned carcass still looks enough like a live one, the only way the model can tell them apart is if depleted ones were deliberately left unlabelled as hard negatives, and the dataset gives no sign of that (only 389 of 3,203 images are empty, and no tags are used anywhere: `tags` is empty on every image). A node the bot just gathered would keep being detected (fiber is the model's best resource class, recall .93-.95) and the bot would walk back to it. This fits the symptom well. A fix needs either new classes (for example `fiber_depleted`, `hide_skinned`) with a few hundred examples each, or bot-side memory of coordinates already gathered; Roboflow cannot express "used" with the current labels.

**Detector misses things.** At v5's evaluation threshold, recall is 80% overall (valid) / 82% (test), and the weak classes are:
- `mob` recall .72-.73 (3,632 labelled boxes, so this is not a data-volume problem; the class is probably visually noisy, with many sizes and states)
- `red` (the bot's threat class) recall .61 valid / .70 test, and only 295 boxes
- `stone` recall .60 on test (371 boxes)
- `silver` recall .70 valid / .80 test (275 boxes), `ore` .79-.80, `hide` .80-.83
- The bot also applies `min_confidence = 0.65` on top of the model, which lowers recall further for these classes.
- The network is YOLOv8-nano, and 1920x1080 and 2560x1440 frames are shrunk to 0.48x and 0.36x to fit 920x920, which hurts small objects such as silver, cup of piss and wisp caged.
- mob, silver and ore have the lowest mAP50-95 (.45-.48), so even found boxes are loosely placed.

**Detects mobs that are not there.** `mob` precision is .87 (valid) and .85 (test), i.e. roughly one mob detection in seven at evaluation time is wrong, among the lowest precisions in the model (only hide is similar). Contributing factors visible in the data:
- v5 was fine-tuned for only 16 epochs from the v4 checkpoint and is slightly *worse* than v4 on mob precision (v4 test .910, valid .892; v5 test .848, valid .866) and on overall test mAP50 (v4 .889, v5 .877). v4 was trained 2024-11-06 in the same project (`albion_detector-l4o1a/4`, `yolov8` export available), so it is a cheap comparison to run.
- Only 12% of the images are empty, giving the model few examples of "scenery that is not a mob".
- Augmentation includes vertical flips and 90/180/270-degree rotation (plus up to 10 degrees of rotation). The game camera is fixed, so upside-down or sideways sprites and trees never occur in play; the 7x copies spend capacity on poses that cannot appear, and rotated trees, rocks and fibre may be easier to confuse with mobs.
- Validation and test images appear to come from the same video-frame sessions as training (file names like `000_<uuid>_15.png`, `screenshot_<uuid>_17.png`, i.e. numbered frames of the same recordings). If splitting was random per frame, neighbouring frames leak across splits and the 85-88 mAP50 overstates how well the model does on a new map or session. This cannot be confirmed through the API; it matches the pattern of good metrics but worse play.
- The `mob` class may have been labelled inconsistently across sessions. A separate `mob_detector` project (key 1) has `mob`, `red` and `hide` too (3,118 mob boxes), and `hide` counts differ widely between the copies: 1,423 in the scene project, 801 in key-1 `albion_detector`, 388 in `mob_detector`. If the labelling criteria changed between sessions, the class boundaries are inconsistent.

**Obstacles.** `obstacle_instance_detection/5` is the weakest model: mask mAP50 .66 valid, recall .63; 402 of its 736 images have no obstacle annotation and 417 sit outside the splits, so the usable set is 319 images (1,238 masks). It is trained on a 720x720 stretch (not fit), which distorts 16:9 frames. Plausibly responsible for path-blocking or stuck behaviour, but that was not among the reported symptoms.

**Older duplicate project.** Key-1 `albion_detector` (2,298 images, 10,670 in its v19) is an earlier copy of the scene dataset with a different class mix (an `oer` typo class with 0 boxes; ore 616, silver 239). The edu-workspace project is the superset the bot uses.

## 5. Other things worth knowing

- `spell_detector/6` (created 2026-11-03, 337 images, no preprocessing or augmentation, `coco` export only) was trained on 2026-11-03/04 as an **RF-DETR NAS** job: a search group of 80 candidate models (`albiondetector/spell_detector-6-rfdetr-nas-pecoret-t1--*`), status finished. The version endpoint reports no mAP for it. The bot does not call it (`spell_detector/4`). The project has 441 images, 108 with no boxes and 44 unassigned; 990 `up` / 1,013 `cooldown` boxes, balanced.
- `spell_detector/5` (2025-04-10, 1,543 images, YOLOv8, mAP50 97.90 / P 95.56 / R 96.23, exports yolov8, yolov11, yolov5pytorch) is the best classic version and is not in use.
- `low_detector` is a 46-image, 2-class classifier (27 `low`, 19 `good`). Accuracy 1.0 on 5 test images means nothing statistically.
- `loading_bar` lists two classes in the metadata but `gathering_bar` has 0 boxes; the effective set is 47 boxes in 145 images (98 empty).
- Other projects the bot does not use: `goldmarket` (16 UI classes, mAP 99.3, 765 images in v5), `slave_stuff` (17 UI classes, 16 versions, mAP 99.5), `find_items` (1 class `item`, mAP 99.5), `items-rxdzn` (40-class classification of inventory items, 543 images, never trained), `labelroads` / `labelroads-3inbp` (road/zone icons, mAP up to 98), `roadsofavalon` (map UI, mAP up to 98.1), `albion-map-swrfl`, `albion-semantic-test` (semantic segmentation, 5 images), `obstacle_fork` (instance segmentation, 1,667 images, no versions), and `billing`, an unrelated public project that the API resolves under both workspaces.

## 6. Suggested next checks (not done: this survey was read-only)

1. Open `albion_detector-l4o1a` in the web UI, filter to `fiber` / `hide`, and look at whether depleted nodes appear and how they were labelled.
2. Run v4 against v5 on recorded play footage; v4 has the better mob precision and test mAP.
3. Check how the valid/test splits were made (frame leakage).
4. Decide whether to add depleted-state classes or track gathered coordinates in the bot.

# Appendix: every project

Counts and metrics are what the API returns. mAP50 / P / R are Roboflow's reported values in percent for the version's trained model.

## Workspace `albiondetector` (name "albiondetector", 3 members, secret key 1)

8 projects. All projects report `public: true` (license CC BY 4.0).

### `albion-map-swrfl`

- Type: **object-detection**; created 2024-07-01, last updated 2026-10-28; 2 versions
- Images (dataset, API metadata): 38 (train 31 / valid 4 / test 3); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): cooldown: 133, walker: 121, up: 95

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 2 | 2024-07-06 | 199 (193/3/3) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; rot90 cw ccw 180; flip HV | 100.00 / 87.18 / 96.67 | yolov5pytorch, yolov8 |
| 1 | 2024-07-06 | 204 (198/3/3) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; rot90 cw ccw 180; flip HV | 98.61 / 98.78 / 93.33 | yolov5pytorch |

### `albion-semantic-test`

- Type: **semantic-segmentation**; created 2024-07-01, last updated 2024-10-22; 2 versions
- Images (dataset, API metadata): 5 (train 1 / valid 2 / test 2); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): obstacle: 244, map_road: 36, exit: 11, large_map: 3, player: 2, fiber-5ZBA: 0, \: 0

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 2 | 2024-10-22 | 11 (7/2/2) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 4px; rotate +-15deg; crop 50% | - / - / - | png-mask-semantic |
| 1 | 2024-10-22 | 17 (14/1/2) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 4px; rotate +-15deg; crop 50% | not trained | - |

### `albion_detector`

- Type: **object-detection**; created 2024-07-01, last updated 2026-10-22; 18 versions
- Images (dataset, API metadata): 2298 (train 1440 / valid 510 / test 348); unannotated reported: 1
- Class annotation counts (project metadata, may be stale / null): mob: 3352, player: 3316, fiber: 1720, hide: 801, ore: 616, tree: 422, stone: 413, red: 315, silver: 239, cup of piss: 72, wisp caged: 48, oer: 0
- **Live count from the image-search API** (5055 images in project incl. unassigned; splits {'none': 2756, 'valid': 510, 'train': 1441, 'test': 348}; 2610 images with zero annotations):
  - boxes per class: mob: 3352, player: 3316, fiber: 1720, hide: 801, ore: 616, tree: 422, stone: 413, red: 315, silver: 239, cup of piss: 72, wisp caged: 48
  - images containing class: player: 2408, mob: 1777, fiber: 870, ore: 375, hide: 356, tree: 283, stone: 278, red: 228, silver: 197, cup of piss: 72, wisp caged: 47

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 19 | 2024-10-30 | 10670 (9828/499/343) | auto-orient; resize Fit (black edges) in 840x840 | x7 outputs/img; rotate +-10deg; rot90 cw ccw 180; noise 0.23%; flip HV | 85.26 / 89.99 / 78.17 | coco, yolov11, yolov12, yolov5pytorch, yolov8 |
| 18 | 2024-10-25 | 9951 (9142/479/330) | auto-orient; resize Fit (black edges) in 1080x1080 | x7 outputs/img; rotate +-10deg; flip H; rot90 cw ccw 180; noise 0.23% | 84.89 / 88.92 / 78.32 | yolov11, yolov5pytorch |
| 17 | 2024-10-25 | 9425 (8645/460/320) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; rotate +-10deg; flip H; rot90 cw ccw 180; noise 0.23% | not trained | - |
| 16 | 2024-10-25 | 9425 (8645/460/320) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; blur 1px; rotate +-10deg; noise 0.51%; crop 20%; flip H | 81.82 / 88.81 / 74.54 | yolov5pytorch |
| 15 | 2024-10-24 | 8986 (8232/443/311) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 1px; rotate +-10deg; noise 0.51% | 84.23 / 90.66 / 76.14 | yolov11, yolov5pytorch |
| 14 | 2024-10-24 | 8986 (8232/443/311) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 1px; rotate +-10deg; noise 0.51% | not trained | - |
| 12 | 2024-10-23 | 7871 (7203/389/279) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 1px; rotate +-10deg; noise 0.51% | 82.81 / 85.36 / 76.80 | yolov11, yolov5pytorch |
| 11 | 2024-10-23 | 7326 (6706/359/261) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; x7 outputs/img; blur 1px; rotate +-10deg; noise 0.51% | not trained | - |
| 10 | 2024-10-22 | 5256 (4881/210/165) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; x7 outputs/img | 80.21 / 89.24 / 70.36 | yolov5pytorch |
| 9 | 2024-10-22 | 5252 (4877/210/165) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; x7 outputs/img | 82.01 / 86.87 / 75.91 | yolov5pytorch |
| 8 | 2024-10-21 | 3247 (3073/97/77) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; x7 outputs/img; shear 10/10 | not trained | yolov8 |
| 7 | 2024-10-21 | 3276 (3171/73/32) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; x7 outputs/img; shear 10/10 | 68.02 / 67.08 / 66.65 | yolov5pytorch, yolov8 |
| 6 | 2024-10-21 | 3259 (3154/73/32) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; x7 outputs/img | not trained | yolov5pytorch |
| 5 | 2024-07-03 | 969 (898/54/17) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; rot90 cw ccw 180; flip HV | 77.42 / 81.34 / 71.24 | yolov5pytorch, yolov8, yolov9 |
| 4 | 2024-07-02 | 400 (363/28/9) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; rot90 cw ccw 180; flip HV | 79.06 / 80.98 / 71.84 | yolov5pytorch, yolov8 |
| 3 | 2024-07-02 | 412 (375/28/9) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; rot90 cw ccw 180; noise 1.05%; flip HV | 79.06 / 85.04 / 69.87 | yolov5pytorch, yolov8, yolov8-obb |
| 2 | 2024-07-01 | 76 (72/2/2) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; rot90 cw ccw 180; noise 1.05%; flip HV | not trained | yolov5pytorch, yolov8 |
| 1 | 2024-07-01 | 63 (60/1/2) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; rot90 cw ccw 180; noise 1.05%; flip HV | 82.03 / 45.99 / 76.67 | yolov11, yolov5pytorch, yolov8 |

### `labelroads`

- Type: **object-detection**; created 2025-07-08, last updated 2025-07-08; 3 versions
- Images (dataset, API metadata): 79 (train 52 / valid 17 / test 10); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): water: 635, green: 321, zone_name: 146, groupdungeon: 106, ho: 88, blue: 63, gold: 59, large_water: 57, hide: 29, stone: 20, blank: 19, large_ore: 19, large_log: 17, cotton: 17, ore: 16, solo: 14, log: 11, large_hide: 8, large_stone: 7, large_cotton: 4, portalbrec: 1, -blue-green-gold-stone-log-hide-: 0, wa: 0, road_name: 0

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 3 | 2025-07-08 | 133 (126/4/3) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; blur 0.7px; noise 0.1%; exposure {"percent": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; saturation {"percent": 25}; x7 outputs/img | not trained | - |
| 2 | 2025-07-08 | 133 (126/4/3) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; rotate +-15deg; blur 0.7px; noise 0.1%; exposure {"percent": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; saturation {"percent": 25}; x7 outputs/img | 79.10 / 72.71 / 75.96 | yolov5pytorch |
| 1 | 2025-07-08 | 17 (12/3/2) | auto-orient | none | not trained | yolov5pytorch |

### `mob_detector`

- Type: **object-detection**; created 2024-07-01, last updated 2025-11-08; 1 versions
- Images (dataset, API metadata): 2118 (train 1308 / valid 480 / test 330); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): mob: 6236, hide: 776, red: 622
- **Live count from the image-search API** (4782 images in project incl. unassigned; splits {'train': 3972, 'valid': 480, 'test': 330}; 3072 images with zero annotations):
  - boxes per class: mob: 3118, hide: 388, red: 311
  - images containing class: mob: 1626, red: 227, hide: 226

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 1 | 2024-10-27 | 8658 (7848/480/330) | auto-orient; resize Fit (black edges) in 720x720 | rotate +-10deg; flip H; rot90 cw ccw 180; noise 0.31%; x6 outputs/img | 79.92 / 86.50 / 69.28 | yolo26, yolov11, yolov5pytorch, yolov7pytorch, yolov8 |

### `obstacle_fork`

- Type: **instance-segmentation**; created 2024-06-15, last updated 2024-10-28; 0 versions
- Images (dataset, API metadata): 1667 (train 1300 / valid 246 / test 121); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): obstacle: 29, a: 0, o: 0

### `roadsofavalon`

- Type: **object-detection**; created 2025-06-23, last updated 2025-07-13; 5 versions
- Images (dataset, API metadata): 206 (train 140 / valid 42 / test 24); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): zone_current: 257, ava_connection: 86, map_menu: 81, open_connection: 46, portal: 0

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 5 | 2025-07-13 | 786 (735/32/19) | auto-orient; resize Stretch to 900x900 | flip HV; blur 0.4px; rot90 cw ccw 180; noise 0.5%; exposure {"percent": 5}; saturation {"percent": 7}; x7 outputs/img; brightness {"percent": 8, "brighten": true, "darken": true} | not trained | - |
| 4 | 2025-07-13 | 526 (490/22/14) | auto-orient; resize Stretch to 900x900 | flip HV; blur 0.4px; rot90 cw ccw 180; noise 0.5%; rgrayscale {"percent": 2}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 5}; saturation {"percent": 7}; x7 outputs/img | not trained | - |
| 3 | 2025-06-24 | 526 (490/22/14) | auto-orient; resize Stretch to 900x900 | flip HV; blur 0.4px; rot90 cw ccw 180; noise 0.5%; rgrayscale {"percent": 2}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 5}; saturation {"percent": 7}; x7 outputs/img | 81.33 / 73.88 / 81.67 | yolov5pytorch |
| 2 | 2025-06-24 | 200 (189/7/4) | auto-orient; resize Stretch to 900x900 | flip HV; blur 0.4px; rot90 cw ccw 180; noise 0.5%; rgrayscale {"percent": 2}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 5}; saturation {"percent": 7}; x7 outputs/img | 74.63 / 96.57 / 75.00 | yolov5pytorch |
| 1 | 2025-06-24 | 38 (27/7/4) | auto-orient | none | not trained | yolov5pytorch |

### `spell_detector` (**used by the bot**, v4)

- Type: **object-detection**; created 2024-07-01, last updated 2026-11-03; 6 versions
- Images (dataset, API metadata): 337 (train 201 / valid 84 / test 52); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): cooldown: 1013, up: 990
- **Live count from the image-search API** (441 images in project incl. unassigned; splits {'valid': 84, 'train': 261, 'test': 52, 'none': 44}; 108 images with zero annotations):
  - boxes per class: cooldown: 1013, up: 990
  - images containing class: cooldown: 301, up: 257

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 6 | 2026-11-03 | 337 (201/84/52) | none | none | not trained | coco |
| 5 | 2025-04-10 | 1543 (1407/84/52) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; blur 0.8px; noise 0.46%; hue {"degrees": 10}; saturation {"percent": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 10} | 97.90 / 95.56 / 96.23 | yolov11, yolov5pytorch, yolov8 |
| 4 | 2024-11-09 | 1298 (1176/74/48) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; noise 0.34%; blur 0.9px | 97.36 / 95.48 / 95.26 | coco, yolov11, yolov5pytorch, yolov8 |
| 3 | 2024-10-24 | 1239 (1120/72/47) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; noise 0.18%; saturation {"percent": 15}; blur 1px | 97.73 / 96.06 / 95.36 | yolov5pytorch |
| 2 | 2024-10-23 | 463 (441/14/8) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; noise 0.18%; saturation {"percent": 15}; blur 1px | 99.33 / 96.15 / 97.44 | yolov5pytorch |
| 1 | 2024-10-22 | 327 (315/7/5) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; noise 0.18%; saturation {"percent": 15}; blur 1px | 99.50 / 99.64 / 100.00 | yolov5pytorch |

## Workspace `albion-edu-workspace` (name "albion edu workspace", 2 members, secret key 2)

10 projects. All projects report `public: true` (license CC BY 4.0).

### `albion_detector-l4o1a` (**used by the bot**, v5)

- Type: **object-detection**; created 2024-07-01, last updated 2026-10-28; 5 versions
- Images (dataset, API metadata): 2768 (train 1770 / valid 603 / test 395); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): player: 4017, fiber: 2267, tree: 421, stone: 413, red: 318, cup of piss: 72, wisp caged: 49, ore: None, silver: None, hide: None, mob: None
- **Live count from the image-search API** (3203 images in project incl. unassigned; splits {'train': 1762, 'valid': 601, 'test': 393, 'none': 447}; 389 images with zero annotations):
  - boxes per class: player: 3812, mob: 3632, fiber: 2216, hide: 1423, ore: 756, tree: 398, stone: 371, red: 295, silver: 275, cup of piss: 62, wisp caged: 49
  - images containing class: player: 2775, mob: 1952, fiber: 1095, hide: 682, ore: 471, tree: 261, stone: 239, silver: 232, red: 207, cup of piss: 62, wisp caged: 48

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 5 | 2024-11-09 | 13299 (12306/600/393) | resize Fit (black edges) in 920x920; auto-orient | flip HV; rot90 cw ccw 180; noise 0.1%; x7 outputs/img; rotate +-10deg | 84.99 / 89.78 / 79.89 | benchmarker, coco, openai, yolo26, yolov11, yolov5-obb, yolov5pytorch, yolov7pytorch, yolov8 |
| 4 | 2024-11-06 | 13055 (12075/591/389) | resize Fit (black edges) in 920x920; auto-orient | flip HV; rot90 cw ccw 180; noise 0.1%; x7 outputs/img; rotate +-10deg | 85.51 / 89.06 / 79.79 | yolov5pytorch, yolov8 |
| 3 | 2024-11-04 | 12825 (11858/583/384) | resize Fit (black edges) in 920x920; auto-orient | flip HV; rot90 cw ccw 180; noise 0.1%; x7 outputs/img; rotate +-10deg | 85.47 / 89.89 / 78.86 | coco, yolov5pytorch |
| 2 | 2024-11-01 | 11793 (10885/544/364) | resize Fit (black edges) in 920x920; auto-orient | flip HV; rot90 cw ccw 180; rotate +-15deg; noise 0.1%; x7 outputs/img | 83.83 / 88.27 / 77.65 | yolov5pytorch, yolov8 |
| 1 | 2024-10-31 | 10938 (10080/510/348) | resize Fit (black edges) in 840x840 | flip HV; rot90 cw ccw 180; rotate +-15deg; noise 0.1%; x7 outputs/img | 84.34 / 86.47 / 79.26 | yolov11, yolov5pytorch, yolov8 |

### `find_items`

- Type: **object-detection**; created 2025-03-14, last updated 2026-10-28; 2 versions
- Images (dataset, API metadata): 140 (train 99 / valid 27 / test 14); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): item: 4445

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 2 | 2025-03-14 | 338 (297/27/14) | auto-orient; resize Stretch to 840x640 | rot90 cw ccw 180; flip HV; rgrayscale {"percent": 10}; noise 0.5%; blur 1px; exposure {"percent": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; saturation {"percent": 20}; x3 outputs/img | 99.50 / 99.50 / 100.00 | yolo26, yolov5pytorch, yolov8 |
| 1 | 2025-03-14 | 89 (84/3/2) | auto-orient; resize Stretch to 840x640 | rot90 cw ccw 180; flip HV; rgrayscale {"percent": 10}; blur 1.5px; noise 0.5%; x7 outputs/img | 99.50 / 99.88 / 100.00 | yolov5pytorch |

### `goldmarket`

- Type: **object-detection**; created 2025-04-19, last updated 2026-10-28; 5 versions
- Images (dataset, API metadata): 309 (train 228 / valid 53 / test 28); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): market_button: 302, gold_market: 286, setup_gold_order: 283, setup_gold_amount: 258, setup_silver_amount: 237, place_order_button: 235, orders_button: 228, amount_gold: 227, amount_silver: 227, get_gold_button: 227, open_gold_order: 194, cancel_order: 181, popup_menu: 139, ok_button: 127, gold_price: 78, confirm_button: 41

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 5 | 2025-05-17 | 765 (684/53/28) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; blur 0.5px; noise 0.58%; hue {"degrees": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 4}; saturation {"percent": 25}; flip HV | 99.34 / 98.88 / 99.48 | yolo26, yolov5pytorch |
| 4 | 2025-05-14 | 765 (684/53/28) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; blur 0.5px; noise 0.58%; hue {"degrees": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 4}; saturation {"percent": 25}; flip HV | 99.79 / NaN / NaN | coco |
| 3 | 2025-04-20 | 654 (573/53/28) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; blur 0.5px; noise 0.58%; hue {"degrees": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 4}; saturation {"percent": 25}; flip HV | not trained | - |
| 2 | 2025-04-20 | 317 (279/24/14) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; blur 0.5px; noise 0.58%; hue {"degrees": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 4}; saturation {"percent": 25}; flip HV | 99.06 / 98.50 / 99.38 | yolov5pytorch |
| 1 | 2025-04-19 | 193 (171/13/9) | auto-orient; resize Stretch to 640x640 | x3 outputs/img; blur 0.5px; noise 0.58%; hue {"degrees": 10}; brightness {"percent": 10, "brighten": true, "darken": true}; exposure {"percent": 4}; saturation {"percent": 25}; flip HV | 99.35 / 97.20 / 100.00 | yolov5pytorch |

### `items-rxdzn`

- Type: **classification**; created 2025-03-14, last updated 2026-10-28; 6 versions
- Images (dataset, API metadata): 543 (train 381 / valid 109 / test 53); unannotated reported: 809
- Class annotation counts (project metadata, may be stale / null): Q0nm: 671, E0: 450, T4: 275, cBARS: 268, T5: 200, E1: 175, cARMOR: 157, E3: 154, T6: 128, E2: 125, T8: 123, Q3ex: 121, T7: 120, Q1gd: 97, T2: 79, Unlabeled: 77, cPLANKS: 75, cWEAPON: 75, E4: 72, Q2os: 58, cMISC: 50, cORE: 44, cPOTION: 34, cCape: 32, cFOOD: 30, cLOGS: 28, cMOUNT: 27, T3: 22, cSKIN: 22, cBAG: 19, cCLOTH: 19, Q4mp: 17, cLEATHER: 11, incomplete: 10, cOFFHAND: 10, T1: 9, cHIDE: 4, cHEART: 3, cSTONE: 2, cCOTTON: 1

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 6 | 2025-04-14 | 2835 (2674/108/53) | auto-orient; resize Stretch to 256x256 | flip HV; rot90 cw ccw 180; blur 1px; x7 outputs/img; hue {"degrees": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 6}; saturation {"percent": 25} | accuracy NaN | multiclass |
| 5 | 2025-04-11 | 2375 (2240/91/44) | auto-orient; resize Stretch to 256x256 | flip HV; rot90 cw ccw 180; blur 1px; x7 outputs/img; hue {"degrees": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 6}; saturation {"percent": 25} | accuracy NaN | multiclass |
| 4 | 2025-03-17 | 2375 (2240/91/44) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; blur 1px; x7 outputs/img; hue {"degrees": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 4} | accuracy NaN | multiclass |
| 3 | 2025-03-14 | 1729 (1631/66/32) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; blur 1px; noise 0.54%; x7 outputs/img; hue {"degrees": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 4} | accuracy NaN | multiclass |
| 2 | 2025-03-14 | 772 (728/30/14) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; blur 1px; noise 0.54%; x7 outputs/img; hue {"degrees": 10}; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 4} | accuracy NaN | multiclass |
| 1 | 2025-03-14 | 289 (273/11/5) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; blur 1px; noise 0.54%; x7 outputs/img | accuracy NaN | multiclass |

### `labelroads-3inbp`

- Type: **object-detection**; created 2025-07-08, last updated 2025-07-21; 5 versions
- Images (dataset, API metadata): 330 (train 227 / valid 67 / test 36); unannotated reported: -4
- Class annotation counts (project metadata, may be stale / null): water: 1951, green: 1179, ho: 326, zone_name: 322, gold: 233, groupdungeon: 169, blue: 163, log: 91, stone: 87, cotton: 83, ore: 75, hide: 73, solo: 71, large_log: 50, large_water: 39, large_cotton: 38, large_hide: 38, avadungeon: 35, large_stone: 31, portalbrec: 28, large_ore: 26, blank: 25

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 5 | 2025-07-21 | 784 (681/67/36) | resize Stretch to 640x640; auto-orient | x3 outputs/img; rot90 cw ccw 180; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 5}; hue {"degrees": 6}; saturation {"percent": 10}; noise 1.01% | 97.76 / 95.72 / 96.70 | yolo26, yolov5pytorch |
| 4 | 2025-07-18 | 610 (528/52/30) | resize Stretch to 640x640; auto-orient | x3 outputs/img; rot90 cw ccw 180; brightness {"percent": 15, "brighten": true, "darken": true}; exposure {"percent": 5}; hue {"degrees": 6}; saturation {"percent": 10}; noise 1.01% | 98.05 / 95.63 / 96.48 | yolov5pytorch |
| 3 | 2025-07-14 | 475 (405/44/26) | resize Stretch to 640x640; auto-orient | x3 outputs/img; flip HV; rot90 cw ccw 180; saturation {"percent": 25}; brightness {"percent": 15, "brighten": true, "darken": true}; noise 0.26%; exposure {"percent": 5}; hue {"degrees": 9} | 97.75 / 95.92 / 95.53 | yolov5pytorch |
| 2 | 2025-07-09 | 255 (219/23/13) | resize Stretch to 640x640; auto-orient | x3 outputs/img; flip HV; rot90 cw ccw 180; saturation {"percent": 25}; noise 0.1%; blur 0.5px; exposure {"percent": 10}; hue {"degrees": 16}; brightness {"percent": 15, "brighten": true, "darken": true} | 93.59 / 90.66 / 83.46 | yolov5pytorch |
| 1 | 2025-07-08 | 183 (156/17/10) | resize Stretch to 640x640; auto-orient | x3 outputs/img; flip HV; rot90 cw ccw 180; saturation {"percent": 25}; noise 0.1%; blur 0.5px; exposure {"percent": 10}; hue {"degrees": 16}; brightness {"percent": 15, "brighten": true, "darken": true} | 94.04 / 73.18 / 89.44 | yolov5pytorch |

### `loading_bar` (**used by the bot**, v1)

- Type: **object-detection**; created 2024-10-31, last updated 2026-10-28; 1 versions
- Images (dataset, API metadata): 145 (train 102 / valid 29 / test 14); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): loading_bar: 47, gathering_bar: 0
- **Live count from the image-search API** (145 images in project incl. unassigned; splits {'valid': 29, 'train': 102, 'test': 14}; 98 images with zero annotations):
  - boxes per class: loading_bar: 47
  - images containing class: loading_bar: 47

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 1 | 2024-10-31 | 757 (714/29/14) | auto-orient; resize Fit (black edges) in 720x720 | flip H; rot90 cw ccw 180; noise 0.1%; rotate +-15deg; x7 outputs/img | 99.50 / 99.21 / 100.00 | yolo26, yolov11, yolov5pytorch, yolov8 |

### `low_detector` (**used by the bot**, v1)

- Type: **classification**; created 2024-11-09, last updated 2026-10-28; 1 versions
- Images (dataset, API metadata): 46 (train 32 / valid 9 / test 5); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): low: 27, good: 19
- **Live count from the image-search API** (46 images in project incl. unassigned; splits {'train': 32, 'valid': 9, 'test': 5}; 46 images with zero annotations):
  - boxes per class: none (classification project)

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 1 | 2024-11-09 | 238 (224/9/5) | auto-orient; resize Stretch to 640x640 | flip HV; rot90 cw ccw 180; blur 2px; rotate +-15deg; noise 0.14%; x7 outputs/img | accuracy NaN | clip, folder |

### `obstacle_instance_detection` (**used by the bot**, v5)

- Type: **instance-segmentation**; created 2024-10-27, last updated 2026-10-28; 5 versions
- Images (dataset, API metadata): 319 (train 225 / valid 61 / test 33); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): obstacle: 1238, \: 0
- **Live count from the image-search API** (736 images in project incl. unassigned; splits {'test': 33, 'train': 225, 'valid': 61, 'none': 417}; 402 images with zero annotations):
  - boxes per class: obstacle: 1238
  - images containing class: obstacle: 334

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 5 | 2024-11-09 | 1669 (1575/61/33) | auto-orient; resize Stretch to 720x720 | flip HV; noise 0.1%; x7 outputs/img; rotate +-10deg | 66.35 / - / - | yolo26, yolov11, yolov5pytorch, yolov7pytorch, yolov8, yolov8-obb |
| 4 | 2024-11-06 | 1483 (1400/54/29) | auto-orient; resize Stretch to 720x720 | flip HV; noise 0.1%; x7 outputs/img; rgrayscale {"percent": 10}; rotate +-10deg | 66.35 / - / - | yolov5pytorch, yolov8, yolov8-obb |
| 3 | 2024-11-06 | 1075 (1015/39/21) | auto-orient; resize Stretch to 720x720 | flip HV; noise 0.1%; x7 outputs/img; rgrayscale {"percent": 10}; rotate +-10deg | 65.04 / - / - | yolov5pytorch |
| 2 | 2024-10-28 | 660 (623/24/13) | auto-orient; resize Stretch to 720x720 | rgrayscale {"percent": 15}; flip HV; rotate +-15deg; noise 0.1%; x7 outputs/img | 76.33 / - / - | yolov5pytorch |
| 1 | 2024-10-28 | 267 (252/10/5) | auto-orient; resize Stretch to 720x720 | rgrayscale {"percent": 15}; flip HV; rotate +-15deg; noise 0.1%; x7 outputs/img | 82.66 / - / - | coco, multiclass, yolov5pytorch, yolov8-obb |

### `roadsofavalon-1ueto`

- Type: **object-detection**; created 2025-06-23, last updated 2026-10-28; 3 versions
- Images (dataset, API metadata): 250 (train 171 / valid 50 / test 29); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): zone_current: 314, ava_connection: 106, map_menu: 97, open_connection: 58, portal: 0

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 3 | 2025-07-14 | 592 (513/50/29) | auto-orient | x3 outputs/img; flip HV; hue {"degrees": 10}; saturation {"percent": 15}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; noise 0.26%; rot90 180 | 98.08 / 94.81 / 97.91 | yolo26, yolov5pytorch |
| 2 | 2025-07-14 | 506 (438/43/25) | auto-orient | x3 outputs/img; flip HV; rot90 cw ccw 180; hue {"degrees": 10}; saturation {"percent": 15}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; noise 0.1%; blur 0.3px | not trained | yolov5pytorch |
| 1 | 2025-07-13 | 486 (420/42/24) | resize Stretch to 640x640 | x3 outputs/img; flip HV; rot90 cw ccw 180; hue {"degrees": 10}; saturation {"percent": 15}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; noise 0.1%; blur 0.3px | 97.63 / 97.10 / 98.33 | yolov5pytorch |

### `slave_stuff`

- Type: **object-detection**; created 2025-01-05, last updated 2026-10-28; 16 versions
- Images (dataset, API metadata): 498 (train 346 / valid 98 / test 54); unannotated reported: 0
- Class annotation counts (project metadata, may be stale / null): chat_box: 331, popup_menu: 172, yes_button: 170, no_button: 168, trade_menu: 101, trader_name: 101, trade_value: 100, trade_pending: 72, accept_button: 61, login_menu: 52, x_button: 46, character_menu: 45, login_button: 44, ad_menu: 41, trade_accepted: 29, enter_world_button: 28, ok_button: 27

| ver | created | images (train/valid/test) | preprocessing | augmentation | mAP50 / P / R | exports |
|---|---|---|---|---|---|---|
| 16 | 2025-04-19 | 2574 (2422/98/54) | auto-orient; resize Stretch to 840x840 | x7 outputs/img; hue {"degrees": 6}; brightness {"percent": 5, "brighten": true, "darken": true}; exposure {"percent": 5}; saturation {"percent": 10}; blur 0.5px; noise 1.08%; flip V; rot90 cw ccw 180 | 99.50 / 98.57 / 99.78 | yolo26, yolov5pytorch |
| 15 | 2025-04-15 | 2218 (2086/85/47) | auto-orient; resize Stretch to 840x840 | x7 outputs/img; hue {"degrees": 6}; brightness {"percent": 5, "brighten": true, "darken": true}; exposure {"percent": 5}; saturation {"percent": 10}; blur 0.5px; noise 1.08%; flip V; rot90 cw ccw 180 | 99.50 / 99.15 / 99.84 | yolov5pytorch |
| 14 | 2025-04-13 | 1972 (1855/75/42) | auto-orient; resize Stretch to 720x480 | x7 outputs/img; saturation {"percent": 20}; blur 1px; hue {"degrees": 6}; noise 1.48%; brightness {"percent": 5, "brighten": true, "darken": true}; exposure {"percent": 5} | 99.46 / 98.54 / 98.49 | yolov5pytorch |
| 13 | 2025-04-12 | 1830 (1722/70/38) | auto-orient; resize Stretch to 940x640 | x7 outputs/img; saturation {"percent": 20}; brightness {"percent": 10, "brighten": true, "darken": true}; blur 1px; exposure {"percent": 10}; hue {"degrees": 6}; noise 0.81% | 99.45 / 99.08 / 99.43 | yolov5pytorch |
| 12 | 2025-04-12 | 1634 (1540/61/33) | auto-orient; resize Stretch to 940x640 | x7 outputs/img; noise 0.26%; saturation {"percent": 20}; brightness {"percent": 10, "brighten": true, "darken": true}; blur 1px; exposure {"percent": 10}; hue {"degrees": 6} | 99.49 / 99.11 / 99.43 | yolov5pytorch |
| 11 | 2025-03-20 | 1490 (1407/54/29) | auto-orient; resize Stretch to 640x640 | x7 outputs/img; blur 0.5px; noise 0.26%; saturation {"percent": 20}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; hue {"degrees": 4}; flip H | 99.50 / 99.35 / 99.33 | yolov5pytorch |
| 10 | 2025-03-17 | 1372 (1295/50/27) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; saturation {"percent": 20}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; hue {"degrees": 4}; flip HV | 99.42 / 98.24 / 99.68 | yolov5pytorch |
| 9 | 2025-03-17 | 1186 (1120/43/23) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180; saturation {"percent": 20}; exposure {"percent": 6}; brightness {"percent": 10, "brighten": true, "darken": true}; hue {"degrees": 4} | 99.50 / 98.91 / 99.56 | yolov5pytorch |
| 8 | 2025-03-17 | 1186 (1120/43/23) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180 | 99.50 / 97.97 / 98.93 | yolov5pytorch |
| 7 | 2025-01-24 | 1074 (1015/38/21) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180 | 99.50 / 97.69 / 98.61 | yolov5pytorch |
| 6 | 2025-01-07 | 891 (840/35/16) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180 | 99.50 / 99.32 / 100.00 | yolov5pytorch |
| 5 | 2025-01-06 | 558 (525/23/10) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180 | 99.50 / 97.92 / 99.95 | yolov5pytorch |
| 4 | 2025-01-06 | 446 (420/18/8) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26%; flip V; rot90 cw ccw 180 | 99.50 / 96.92 / 99.17 | yolov5pytorch |
| 3 | 2025-01-05 | 313 (294/13/6) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26% | 99.50 / 96.31 / 100.00 | yolov5pytorch |
| 2 | 2025-01-05 | 268 (252/11/5) | auto-orient; resize Stretch to 940x940 | x7 outputs/img; blur 0.5px; noise 0.26% | 99.50 / 98.04 / 99.48 | yolov5pytorch |
| 1 | 2025-01-05 | 179 (168/8/3) | auto-orient; resize Stretch to 940x940 | blur 2px; flip HV; rot90 cw ccw 180; noise 0.1%; x7 outputs/img | 99.50 / 95.57 / 97.38 | yolov5pytorch |
