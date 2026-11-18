# Stage I OBB Experiment Results

## Purpose and method

Localize the complete digit register in a mechanical water-meter photo with a single-class YOLO11n-OBB detector. The Dryad Word-Wheel real-image subset had 2,400 training, 400 validation and 800 held-out test images. The official test pool was kept separate. Training ran for 30 epochs on an NVIDIA T4 with image size 640, batch size 32, AdamW, initial learning rate 0.001 and seed 20261022. Validation OBB fitness selected the checkpoint; the test set was not used for selection.

## Results

| Held-out test metric | Value |
| --- | ---: |
| Precision | 99.42% |
| Recall | 99.00% |
| mAP@0.50 | 99.49% |
| mAP@0.50–0.95 | 85.27% |

Ultralytics 8.4.0 calculates these OBB metrics using probabilistic rotated IoU. They measure register localization, not digit-reading accuracy or orientation classification accuracy.

The highest recorded validation mAP@0.50–0.95 was 88.236% at epoch 24 (mAP@0.50: 99.50%). The selected checkpoint's reported validation metrics are stored separately in final_metrics.json. The checkpoint's embedded epoch metadata was stripped, so epoch 24 cannot be independently verified from that checkpoint alone.

## Files

- stage1_obb_results.png: training loss and validation curves.
- final_metrics.json: split sizes, training configuration, selected-checkpoint validation metrics, held-out test metrics and checkpoint checksum.
- Model weights, per-epoch CSV and full logs remain local under ../outputs/stage1_obb/final_run/artifacts/ and are excluded from Git.

See [the Stage I guide](../docs/STAGE1_OBB.md) for preparation and reproduction.
