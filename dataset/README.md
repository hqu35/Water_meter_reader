# Datasets

Stage-specific datasets are stored in separate subdirectories.

- [stageOne/](stageOne/README.md): reviewed Dryad real-image register-localization dataset (2,400 train / 400 validation / 800 test), including images, YOLO labels, original masks, source archive and annotation-review metadata.

Actual data is ignored by Git. Model checkpoints and experiment results are in `outputs/stage1/`.


<!-- recursive-summary:start -->
## Directory overview

Real-data storage is organized by project stage. This overview aggregates existing descendants without adding README files to reviewed image, label or mask leaves.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |

### Subdirectories

| Directory | Role |
|---|---|
| [stageOne/](stageOne/README.md) | Reviewed Dryad register-localization subset and its source/audit/transfer material; the existing stage-specific README gives the layout. |

### Aggregated contents

Counts exclude README files and internal caches; leaf directories are unchanged.

| Path | Contents | Role |
|---|---|---|
| [stageOne/images/](stageOne/images/) | .jpg: 3600 | Original encoded real photographs, separated into train/val/test. |
| [stageOne/labels/](stageOne/labels/) | .txt: 3600 | Class-0 register boxes in normalized YOLO text format, paired by basename. |
| [stageOne/masks/](stageOne/masks/) | .png: 3600 | Original PNG segmentation masks for the selected examples. |
| [stageOne/metadata/](stageOne/metadata/) | .jpg: 4, .json: 8 | Manifest, conversion checks, approval record, statistics and four review sheets. |
| [stageOne/raw/](stageOne/raw/) | .zip symlink: 1 | Source ZIP entry; the existing archive link is not followed by the inventory. |
| [stageOne/transfers/](stageOne/transfers/) | .part000: 1, .part001: 1, .part002: 1, .part003: 1, .part004: 1, .part005: 1, .zip: 1 | Existing subset ZIP and bounded upload chunks. |
| [stageOne/](stageOne/) | .yaml: 1 | Stage-level data.yaml and existing README; file counts here are non-recursive. |

### Summary

Stage I keeps 2,400 training, 400 validation and 800 held-out real examples with matching labels and masks. The existing source archive, exact split manifest and approval records remain unchanged. Future stage datasets can be sibling directories; this summary neither creates them nor changes the data.

<!-- recursive-summary:end -->
