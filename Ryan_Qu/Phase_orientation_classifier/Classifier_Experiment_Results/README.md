# Orientation Classifier Experiment Results

## Purpose and method

The ViT-Small classifier determines whether a geometrically rectified full
water-meter image is upright or upside down. It was fine-tuned from the
pretrained vit_small_patch16_224.augreg_in21k_ft_in1k model on Apple MPS using
AdamW (learning rate 0.0001, batch size 16, seed 42). Training stopped after
11 epochs; the best checkpoint was selected by validation accuracy at epoch 6.

The grouped split contains 2,394 training files, 513 validation files, and
513 held-out test files. Each file contributes two orientation samples: the
original and its 180-degree counterpart. Source groups and paired orientations
remain within one split.

## Results

| Metric | Result |
| --- | ---: |
| Best validation accuracy | 98.6355% |
| Held-out test accuracy | 99.7076% |
| Held-out test samples | 1,026 |
| Correct / incorrect | 1,023 / 3 |

The test set contains 513 files from 359 distinct source groups. Roboflow v3
files had already been resized and augmented before release, so these results
describe this particular dataset and do not establish accuracy on every meter.
The model's confidence scores are not calibrated probabilities.

## Files

- training_curves.png: training and validation loss and accuracy by epoch.
- final_metrics.json: selected checkpoint, configuration and held-out results.
- The original checkpoint, per-epoch metrics, detailed report and evaluation
  artifacts remain local under
  ../outputs/vit_small_orientation_train_report_11.12/.
