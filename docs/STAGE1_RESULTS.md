# Stage 1 baseline — October 22, 2026

YOLO11n was fine-tuned for **one-class numerical-register localization** on real Dryad images, using the existing `cs539-stage1` NVIDIA Tesla T4 runtime. Training and held-out evaluation completed, and the checkpoints and experiment artifacts were retrieved into this repository's Git-ignored `outputs/stage1/` directory.

## Data and validation

Source: [Dryad Word-Wheel Water Meter Dataset](https://doi.org/10.5061/dryad.7d7wm3860), archive version `422392`, file `4579939`, SHA-256:

```text
3b7d3d8168ab924314a7c71987ce09f54dec95818a9cb484b46ea804fc1d90e6
```

An existing local download was independently verified against this published checksum. Actual archive inspection found 32,707 training and 13,391 test image/mask pairs. Photographs are JPEG-encoded despite their original `.png` names; masks are grayscale PNG with values 0 and 255. Original image bytes were preserved, with extensions normalized to `.jpg`.

| Split | Images | Source |
|---|---:|---|
| Train | 2,400 | Official detection training pool |
| Validation | 400 | Disjoint portion of the official training pool |
| Held-out test | 800 | Official detection test pool |

Selection seed: `20261022`. Exact decoded image hashes are unique across all 3,600 selected images; no selected duplicate or empty-mask exclusions were needed. Every selected image/mask pair has matching dimensions. All boxes contain the complete nonzero mask foreground and round-trip through YOLO coordinates within `1e-5` pixels. All 3,600 images lack EXIF rotation metadata, preventing loader-induced rotation mismatches.

**All 32 original/overlay comparisons were visually inspected before training.** The hash-bound approval record was rechecked on Colab after transferring the subset. The test subset was not used to fit weights, select checkpoints, or select a deployment confidence threshold. No synthetic images or synthetic image-composition augmentation were used.

## Training

- Model: COCO-pretrained `yolo11n.pt`, one output class `register`.
- Runtime: Tesla T4; PyTorch `2.11.0+cu130`, CUDA `13.0`, Python `3.13.15`; Ultralytics `8.4.0`.
- 30 completed epochs; **best checkpoint selected at epoch 28** using validation mAP@0.50–0.95.
- Image size 640, batch 32, AdamW, initial learning rate 0.001, three warmup epochs, `warmup_bias_lr=0.0`, deterministic settings, FP32.
- Geometry, HSV, mosaic, MixUp, CutMix, copy/paste, erasing and flip augmentation were disabled for this baseline. Standard resizing/letterboxing remained enabled.
- Reloaded best-checkpoint validation mAP@0.50–0.95: **0.8245**.

An initial run using the framework's generic 0.1 bias warmup was stopped early and excluded from the reported experiment. Its evidence is retained under `artifacts/attempts/adamw_default_bias_warmup/`. The corrected run restarted from pretrained weights with the same reviewed data.

The framework's generic single-class display name `item` was normalized to `register` in saved checkpoints. Every model/EMA tensor was compared bitwise before and after serialization; this metadata correction changed no learned tensors. The final checkpoint hashes and verification report are saved with the experiment.

## Held-out results

Evaluated on **800 real images / 800 register instances**, with the validation-selected best checkpoint.

| Metric | Value |
|---|---:|
| mAP@0.50 | **0.994810** |
| mAP@0.50–0.95 | **0.783845** |
| Ultralytics PR-curve precision | 0.992544 |
| Ultralytics PR-curve recall | 0.998360 |

The framework's precision/recall above come from its PR-curve summary. They should not be interpreted as the fixed deployment-threshold result below.

At the separately fixed **confidence 0.25 / IoU 0.50** operating point:

| Quantity | Value |
|---|---:|
| Correctly localized registers | **795 / 800** |
| Localization recall | **0.993750** |
| Total predictions | 805 |
| Precision, counting extra predictions as false positives | 795 / 805 = **0.987578** |
| Missed or insufficient-overlap registers | 5 |

The difficult-example contact sheet includes the lowest-IoU cases, including dark/obstructed views and a prediction covering only the visible part of an obscured register. A second sheet contains eight seeded random test examples. Both were visually checked after export. A separate real-image inference check successfully saved a register crop and original-image box coordinates.

## Saved outputs

All paths below are local and **ignored by Git**:

```text
outputs/stage1/stage1_results.zip
outputs/stage1/artifacts/
  runs/register_yolo11n/weights/best.pt
  runs/register_yolo11n/weights/last.pt
  runs/register_yolo11n/weights/epoch*.pt
  runs/register_yolo11n/results.csv
  runs/register_yolo11n/results.png
  heldout_metrics.json
  operating_point.json
  test_predictions.json
  test_predictions_01.jpg
  test_predictions_02.jpg
  example_prediction/
  annotations/review_01.jpg ... review_04.jpg
  split_manifest.json
  validation.json
  annotation_review.json
  checkpoint_class_names.json
  source_snapshot/
  evaluation_source_snapshot/
  pip-freeze.txt
  gpu.json
  artifact_checksums.json
```

The result archive and every extracted file were checksum-verified after retrieval from Colab. The complete source archive and prepared subset are excluded from the result bundle. See [STAGE1.md](STAGE1.md) for preparation, review, training, evaluation and retrieval commands.

## Interpretation limits

This is a reproducible **subset baseline**, not an evaluation on the entire published test set. The source does not supply physical-meter identities establishing group-disjoint splits; exact-image deduplication does not exclude all near-duplicates or repeated photographs of the same meter. The model returns a register box and confidence. Digit localization and digit-state recognition remain later stages.
