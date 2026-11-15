# Water Meter Digit Reader — Project Overview

The project aims to read mechanical water meters from photos. We have trained and evaluated models for **register localization** and **orientation classification**, and built a workflow for labeling **individual digit-wheel positions**. The individual-digit detector and final meter-reading system are not yet trained or evaluated end to end.

The intended flow is: **meter photo → YOLO-OBB register detection → OpenCV full-frame rotation to make the register horizontal → our trained ViT-Small for upright vs. upside-down classification → apply a 180° correction if needed and crop the register → locate individual digit wheels → recognize digits and assemble the reading**. The detector, full-frame rotation, and local ViT-Small have been exercised together on new images. The final two steps are still open.

## Completed work

| Area | What we built and verified |
| --- | --- |
| **Register localization** | Converted Dryad segmentation masks into four-corner OBB labels and reviewed them before training. Fine-tuned a one-class YOLO11n-OBB detector on a Colab NVIDIA T4. The frozen split has **2,400 train / 400 validation / 800 held-out test images** and preserves the official Dryad train/test boundary. On the held-out test set, precision was **99.42%**, recall **99.00%**, and OBB mAP was **0.9949 at IoU 0.50** and **0.8527 at IoU 0.50–0.95**. These measure register localization. |

| **Geometric rectification and crop** | Implemented OpenCV full-frame rotation from the detected register's long axis. The pipeline carries the OBB coordinates through the rotation and any later 180° correction before extracting the register crop. A 10-image pilot tested this path with the historical hosted Roboflow classifier. The local ViT was subsequently tested with Stage I detection and full-frame rectification in a separate pilot. |

| **Local orientation classifier** | Trained ViT-Small to classify **geometrically rectified full frames**, including the meter housing, as upright or upside down. Recovered Dryad source IDs and audited historical rotations so crop labels could be mapped to the correct stage of the full frame. Trained on reviewed Dryad-derived images plus Roboflow v3 files, using grouped splits. The epoch-6 checkpoint reached **98.64% validation accuracy** and **99.71% held-out accuracy** (1,023/1,026 orientation samples, from 513 files and 359 source groups). Roboflow's released files include prior resizing and augmentation; the result applies to this experiment. |
| **Unseen-ID orientation review** | Ran Stage I detection and rectification followed by the local ViT on **100 Dryad IDs excluded from previously traced usage**. All 100 processed successfully. The model predicted **73 upright** and **27 upside down**. A standalone review gallery is available; these counts are predictions, and the saved pilot does not report human-verified accuracy. |
| **Shared crop dataset and review tools** | Prepared **1,500 crops with the original Dryad clear label and no noise flags**, plus **500 clear-labeled crops with exactly one noise flag**. Kept source IDs and selection provenance and reused existing crops where possible. Also built separate human review tools for orientation, LoRA candidate quality, and the dataset author's clear/blurry labels. Scenario flags are a selection aid; they are not visual quality guarantees. |
| **Single-digit localization preparation** | Ran a **100-image zero-shot OBB detector pilot** as an exploratory baseline; no digit localization mAP or IoU was measured. Built an offline annotation tool covering **1,996 images**. Six points generate equal-spacing proposals; saving point seven removes those proposals and switches the remaining wheels to manual four-corner annotation. The latest local export inspected contains **250 complete records: 216 automatic proposals and 34 manual fallback records**. The automatic proposals still need visual verification before they can be treated as training labels. |

## Where to look

- [Stage I detector report](Ryan_Qu/Phase_detection/docs/STAGE1_OBB.md) and [training plot](Ryan_Qu/Phase_detection/training%20loss%20plot/stage1_obb_results.png)
- [Orientation classifier overview](Ryan_Qu/Phase_orientation_classifier/README.md) and [input provenance audit](Ryan_Qu/Phase_orientation_classifier/PROVENANCE_AUDIT.md)
- [Shared dataset selection](Ryan_Qu/shared_dataset/README.md)
- [Single-digit zero-shot experiment](Ryan_Qu/Phase_single_digit_BB/EXPERIMENT_0.md) and [annotation tool instructions](Ryan_Qu/Phase_single_digit_BB/manual_five_point/README.md)

The implemented work and reports available in this checkout are under `Ryan_Qu/`. Checkpoints, generated galleries, and some datasets and metrics are local artifacts; a fresh clone alone does not contain everything needed to rerun the experiments. The component READMEs identify the relevant paths.

## What remains

Verify and consolidate the digit-wheel labels, then train and evaluate a dedicated digit localizer. After that, build and measure digit recognition, including how to interpret a wheel between two digits, and evaluate complete meter readings on held-out photos. The reported detector and orientation metrics are for individual components; there is no end-to-end reading accuracy yet.
