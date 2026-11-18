# Water Meter Reader

A CS539 computer vision project for reading mechanical water meters from photographs. The project separates the problem into register localization, image rectification and orientation correction, individual digit localization, and digit recognition. Register detection and orientation classification have been trained and evaluated; a complete end-to-end meter reader has not yet been established.

## Project pipeline

1. **Locate the register:** A YOLO11n oriented bounding-box (OBB) detector identifies the digit register in a meter photograph.
2. **Rectify and correct orientation:** OpenCV rotates the image using the register geometry. A trained ViT-Small classifier predicts whether the rectified image is upright or upside down; a 180° correction can then be applied before cropping the register.
3. **Locate individual digit wheels:** Annotation tools and exploratory experiments support the preparation of individual-digit bounding boxes. This phase is still under development.
4. **Read the digits:** Recognize the digit on each wheel and assemble the meter reading. This remains future work.

The detector and orientation classifier have been exercised together in a pilot. Component metrics below should not be interpreted as end-to-end reading accuracy.

## Results so far

| Component | Experiment | Reported result |
| --- | --- | --- |
| Register detection | YOLO11n-OBB trained on 2,400 images, with 400 validation and 800 held-out test images | Test precision **99.42%**, recall **99.00%**, mAP@0.50 **99.49%**, and mAP@0.50–0.95 **85.27%** |
| Orientation classification | ViT-Small trained on rectified full-meter images; best checkpoint selected on validation | Validation accuracy **98.64%** and held-out test accuracy **99.71%** (1,023 / 1,026 orientation samples) |
| Single-digit localization | Third-party zero-shot detector pilot and manual annotation workflow | Exploratory work; digit-localization accuracy has not been established |

The orientation test includes paired orientations and data with prior resizing and augmentation; its result describes that specific test set. See the experiment reports for the splits, evaluation methods, and limitations.

- [Register detection results and training curves](Ryan_Qu/Phase_detection/Stage_1_OBB_Experiment_Results/README.md)
- [Orientation classifier results and training curves](Ryan_Qu/Phase_orientation_classifier/Classifier_Experiment_Results/README.md)

## Repository guide

| Directory or file | Purpose |
| --- | --- |
| [Ryan_Qu/Phase_detection/](Ryan_Qu/Phase_detection/) | Register detection data preparation, training, evaluation, documentation, and results. |
| [Ryan_Qu/Phase_orientation_classifier/](Ryan_Qu/Phase_orientation_classifier/) | Orientation classification, training documentation, evaluation, and orientation dataset records. |
| [Ryan_Qu/Phase_single_digit_BB/](Ryan_Qu/Phase_single_digit_BB/) | Individual-digit localization experiments, annotation tools, and synthetic-reference selection. |
| [Ryan_Qu/Phase_reading_from_single_BB/](Ryan_Qu/Phase_reading_from_single_BB/) | Space for the subsequent digit-reading phase. |
| [Ryan_Qu/shared_dataset/](Ryan_Qu/shared_dataset/) | Shared input datasets and source records, including orientation-corrected register crops. |
| [Ryan_Qu/LoRA_training/](Ryan_Qu/LoRA_training/) | Local register crops prepared for LoRA-related work. |
| [Jiwoo_Hong/](Jiwoo_Hong/), [Manus_Kumer/](Manus_Kumer/), [Reilly_Tom/](Reilly_Tom/) | Teammate directories. |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Earlier project overview; some paths and dataset counts may reflect an older layout. |

## Getting started

Start with the [Stage I detection guide](Ryan_Qu/Phase_detection/docs/STAGE1_OBB.md), the [orientation classifier README](Ryan_Qu/Phase_orientation_classifier/README.md), and the [single-digit experiment README](Ryan_Qu/Phase_single_digit_BB/README.md). These documents explain the relevant workflows and their requirements. The [shared dataset README](Ryan_Qu/shared_dataset/README.md) describes the current crop categories and source IDs.

Large datasets, model checkpoints, and some generated experiment outputs are kept outside Git or ignored. A fresh clone therefore may not contain all inputs or weights needed to reproduce a run. The experiment result folders retain concise metrics and plots; consult the component documentation for the original artifact locations.

## Next steps

Complete and review the individual-digit annotations, evaluate a digit localizer, and develop digit recognition, including handling rolling-wheel transitions. Evaluate the combined system on held-out meter photographs once the remaining components are ready.
