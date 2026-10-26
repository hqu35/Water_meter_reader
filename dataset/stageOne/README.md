# Stage One Dataset

All active Stage 1 data is stored here, separately from model outputs.

```text
dataset/stageOne/
├── data.yaml                 # YOLO entry point (absolute root refreshed per machine)
├── images/
│   ├── train/                # 2,400 real images
│   ├── val/                  # 400 validation images
│   └── test/                 # 800 held-out real images
├── labels/{train,val,test}/  # YOLO labels: class 0 = register
├── masks/{train,val,test}/   # Matching original segmentation masks
├── raw/                      # Full verified source ZIP or local symlink
├── metadata/                 # Split manifest, checks and visual-review approval
│   └── annotations/          # Four sheets: 32 reviewed examples
└── transfers/                # Temporary subset ZIP and upload chunks
```

A photograph, its label and its mask have the same basename. `val` means
validation. The extracted directories contain the existing 3,600-image subset;
the complete published source is available through the ZIP in `raw/`.

Only this README is tracked in Git. Actual data, generated metadata and transfer
files are ignored. Do not edit labels or replace images after visual approval:
the training gate checks the original review, every image/label/mask hash and
exact split-directory membership.

Source: Dryad DOI `10.5061/dryad.7d7wm3860`, frozen archive version `422392`,
SHA-256 `3b7d3d8168ab924314a7c71987ce09f54dec95818a9cb484b46ea804fc1d90e6`.
Selection seed: `20261022`. Train and validation originate from the official
training pool; test originates from its official test pool. No synthetic images.

See [the Stage 1 workflow](../../docs/STAGE1.md) for preparation, migration and Colab
commands. Checkpoints and experiment results stay in `outputs/stage1/`.
Historical experiment snapshots can contain duplicate audit records; they are
preserved for reproducibility and are not the active data source.
