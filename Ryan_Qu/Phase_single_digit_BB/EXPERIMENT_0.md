# Phase II Experiment 0 — zero-shot OBB digit detection

100 human-reviewed orientation-corrected Dryad samples. The scenario-based sample is deliberately stratified and is not a population estimate.

Model: `ntarekp/watermeter-yolov8-obb/stage2_digits.pt`; SHA-256 `f5553aaba25b49b8cab4b74d5b87f0ad68779a889fe19dcbc3139439a0f3ef02`.
Input size 416; confidence threshold 0.1; Ultralytics OBB NMS IoU 0.7. No extra grouping was applied.

Total predictions: 671; median confidence 0.770, mean 0.643, range 0.101–0.968.

## Predicted box counts

| Boxes per image | Images |
|---:|---:|
| 3 | 1 |
| 4 | 5 |
| 5 | 26 |
| 6 | 20 |
| 7 | 19 |
| 8 | 9 |
| 9 | 10 |
| 10 | 8 |
| 11 | 1 |
| 12 | 1 |

## Confidence distribution

| Confidence | Boxes |
|---|---:|
| 0.10-0.25 | 119 |
| 0.25-0.50 | 107 |
| 0.50-0.75 | 100 |
| 0.75-1.00 | 345 |

Selection groups: {'dial-stained': 20, 'reflective': 12, 'blurry': 16, 'soil-covered': 25, 'clear': 20, 'dark': 7}. These are priority groups; source scenario flags overlap.

## Review status

Manual localization review has not been performed: good, partial and failed counts are pending. No localization-performance comparison across meter types or image-quality groups is available. There are no digit-level ground-truth boxes, so mAP and IoU are not measured.

Initial visual spot checks (not a systematic review): `sample_000019` is severely obscured yet receives several boxes; `sample_000193` has overlapping predictions near a transition digit; some visually readable registers, such as `sample_000147`, show close alignment. These observations do not establish error rates.

The `u` class means unclear according to the publisher; it is not a physical digit slot. Multiple boxes in a slot can be legitimate for transition states or may be false positives; review is needed.

This checkpoint is worth reviewing as a possible fine-tuning starting point, but do not select it or claim reliable localization before completing the manual gallery review and obtaining digit-level ground truth.
