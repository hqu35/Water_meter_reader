# Stage 1: real-image register localization

This implementation uses the **existing `cs539-stage1` Colab CLI session**. It never calls `colab new`, `colab run`, changes the accelerator, restarts the runtime, or stops the session. Only source, configuration and documentation are tracked in Git. Raw data, pretrained/trained weights, and experiment artifacts are ignored by Git.

## Architecture and reading order

`scripts/stage1.py` is a small launcher. Start with `stage1/cli.py` to see the
phase dispatch, then follow the module for the phase you want to understand:

```text
scripts/stage1.py                 CLI launcher; existing commands are unchanged
stage1/
  cli.py                         Argument parsing and explicit phase dispatch
  config.py                      Experiment JSON, shared paths and source hashes
  data/
    download.py                  Original Dryad download and archive verification
    archive.py                   Exact image/mask pairing, decoding and inspection
    preparation.py               Mask boxes, YOLO export and deterministic splits
    validation.py                Coordinate, file-hash and split-membership checks
    review.py                    32 review examples, manual approval and train gate
  training/
    trainer.py                   T4 check, unchanged YOLO settings and training
    checkpoints.py               Class-name normalization with tensor comparisons
  evaluation/
    evaluator.py                 Held-out AP, bounded predictions and fixed IoU matching
    visualization.py             Worst-IoU and seeded random prediction sheets
  utils/
    io.py                        JSON read/write and streaming SHA-256
    packaging.py                 Result ZIP, audit snapshot and file checksums
```

Data flow: **verify archive → pair images/masks → deterministic split and YOLO
export → inspect review sheets → approve → recheck reviewed files → train →
evaluate selected checkpoint → package → fetch**. Preparation never approves
itself, and training checks approval before invoking the GPU or YOLO.

The Colab bridge remains `scripts/colab_stage1.py`. Its `sync` command now uploads
all package modules as well as the launcher, experiment JSON and requirements.
Run `sync` before calling phases on a previously configured runtime; replacing
only the old single script is no longer enough. It uses the existing named
session and does not allocate or restart a runtime.

The refactor preserves all phase names: `gpu`, `download`, `inspect`, `prepare`,
`approve`, `train`, `normalize-checkpoints`, `evaluate`, and `package`.
For example, these remote-root entry commands are unchanged:

```bash
python3 scripts/stage1.py --root /content/cs539_stage1 prepare
python3 scripts/stage1.py --root /content/cs539_stage1 train
python3 scripts/stage1.py --root /content/cs539_stage1 evaluate
```

These examples are executed **on Colab**; use the bridge below to launch them
from the local computer. GPU libraries are imported by the relevant phase,
so CLI help and data checks do not require a local PyTorch installation.

## Local directory layout

All active Stage 1 data lives in the repository-root `dataset/stageOne/` directory:

```text
dataset/stageOne/
  data.yaml                         # YOLO dataset entry point
  images/{train,val,test}/           # Original encoded photographs
  labels/{train,val,test}/           # Class-0 YOLO bounding boxes
  masks/{train,val,test}/            # Original PNG segmentation masks
  raw/                              # Verified source ZIP (or its local symlink)
  metadata/                         # Frozen split manifest and review records
    annotations/review_*.jpg         # 32 reviewed image/mask/box comparisons
  transfers/                        # Ignored data-transfer ZIP and chunks
outputs/stage1/                      # Checkpoints, metrics and experiment logs
```

`val` is the validation split. Images, labels and masks share the same basename.
Only `dataset/stageOne/README.md` is tracked; actual data and generated metadata are ignored.
The full source archive remains in `raw/`; the extracted directories contain the
existing 2,400/400/800 subset, not the entire published dataset.

For an older checkout with data under `outputs/stage1/prepared/` or the earlier flat `dataset/` layout, run once:

```bash
python3 scripts/migrate_stage1_dataset.py
```

This moves images and labels without resampling, extracts matching original masks,
updates `data.yaml`, and checks file hashes, all mask-to-box alignments, split
membership and the original visual-review gate. Historical experiment artifacts,
source snapshots, checkpoint contents and result ZIPs are preserved unchanged;
their embedded old paths describe the completed run, not the active dataset.

The normal local root is `dataset/stageOne/`. A custom `--root /path/to/run` uses
`/path/to/run/dataset/stageOne/`, and the existing Colab bridge uses
`/content/cs539_stage1/dataset/stageOne/`. Training/evaluation refresh the absolute root in
`data.yaml` for the current environment. Re-sync the updated scripts and transfer
the reviewed subset before using the new layout on an existing Colab runtime.
No remote data migration or training is performed by the local migration command.

## Experiment definition

- Dataset: [Dryad Word-Wheel Water Meter Dataset](https://doi.org/10.5061/dryad.7d7wm3860), published January 30, 2026. Dryad metadata specifies CC0-1.0.
- Frozen archive: version ID `422392`, file ID `4579939`, 2,613,710,367 bytes.
- SHA-256: `3b7d3d8168ab924314a7c71987ce09f54dec95818a9cb484b46ea804fc1d90e6`.
- Task: one class, `0: register`. This stage does not recognize readings or individual digits.
- Planned subset: 2,400 train, 400 validation, 800 held-out test images; seed `20261022`.
- Train and validation come exclusively from the official detection training pool. Test comes exclusively from the official detection test pool. The recognition crops are excluded.
- Selected decoded image pixels are SHA-256 deduplicated across all three splits. The final manifest records exact archive member names, image/mask hashes, dimensions, foreground values, and boxes. Empty masks and duplicates are explicitly recorded as exclusions and replaced deterministically.
- This is image-level evaluation. The source does not provide identities that establish that photographed physical meters are disjoint; exact-image deduplication cannot exclude all near-duplicate or same-meter leakage.
- Model: COCO-pretrained `yolo11n.pt`, Ultralytics `8.4.0`; 640 px, batch 32, AdamW, learning rate 0.001, three warmup epochs with **`warmup_bias_lr=0.0`**, up to 30 epochs, early-stop patience 10. Validation alone selects the checkpoint. The explicit bias warmup avoids applying the generic 0.1 default to this AdamW baseline.
- No synthetic training images. Mosaic, MixUp, CutMix, copy/paste, erasing, flips, perspective/affine transforms and HSV augmentation are disabled for this baseline. Resizing/letterboxing remains part of the model input pipeline.
- The CUDA PyTorch package supplied by Colab is retained, and its exact version and the installed environment are saved. A seed and deterministic settings improve repeatability; exact bitwise agreement across different CUDA/PyTorch versions is not promised.

The counts above describe the configured experiment, not a claim that training has completed. Consult the local `outputs/stage1/STATUS.json` and generated artifacts for execution state.

## Connect and verify

Install the official Colab CLI beforehand and authenticate it in your normal terminal. Do not commit session state or credentials. If `colab` is not on PATH, the bridge also checks `~/.local/bin/colab`.

```bash
python3 scripts/colab_stage1.py status
python3 scripts/colab_stage1.py sync
python3 scripts/colab_stage1.py gpu
```

The GPU phase asserts CUDA availability, checks for an NVIDIA T4, performs a small CUDA matrix multiplication, compares its result against CPU, and saves the GPU, driver, Python and PyTorch report. It deliberately fails rather than silently training on CPU.

Install dependencies **inside the existing session**, retaining its CUDA PyTorch:

```bash
colab exec --session cs539-stage1 <<'PY'
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
               '/content/cs539_stage1/project/requirements-stage1.txt'], check=True)
PY
```

## Download and inspect

```bash
python3 scripts/colab_stage1.py download --background
```

Check `/content/cs539_stage1/download.log` using `colab exec`. Do not start the next phase until the process succeeds. Dryad may reject programmatic downloads with HTTP 403; its API download endpoint can require a bearer token. The script does not circumvent those checks. If the normal [public browser download](https://datadryad.org/downloads/file_stream/4579939) succeeds, upload the unmodified archive instead:

```bash
colab upload --session cs539-stage1 /path/to/Word-Wheel_Water_Meter_Dataset.zip \
  /content/cs539_stage1/dataset/stageOne/raw/Word-Wheel_Water_Meter_Dataset.zip
python3 scripts/colab_stage1.py inspect
```

The inspection phase verifies the published archive checksum before reading it. It inspects image/mask pairs inside the ZIP rather than executing downloaded dataset code or indiscriminately extracting archive contents. It records actual official split counts, example paths, dimensions and mask values in `dataset/stageOne/metadata/inspection.json`.

Observed in the checksum-verified archive: **32,707 train and 13,391 test pairs**, rather than the 32,714 / 13,398 listed in the dataset description. Its actual folders are `train/train_img`, `train/train_seg_label`, `test/test_img`, and `test/test_seg_label`. The photographs have JPEG content despite `.png` source filenames; masks are grayscale PNG with values 0 and 255. Preparation preserves each photograph's original encoded bytes and normalizes its extension to `.jpg`, without recompression. The original archive filenames and exact bytes remain traceable through the manifest.

### Browser-download / local-preparation route

If the archive is already on the local computer, a verified local copy can be prepared without uploading the entire 2.61 GB archive. Place it or an ignored symlink at `dataset/stageOne/raw/Word-Wheel_Water_Meter_Dataset.zip`. With NumPy, Pillow and PyYAML available locally:

```bash
python3 scripts/stage1.py inspect
python3 scripts/stage1.py prepare
# Inspect all four local dataset/stageOne/metadata/annotations/review_*.jpg sheets.
python3 scripts/stage1.py approve --reviewer 'Your name; all 32 pairs inspected'
python3 scripts/colab_stage1.py sync
python3 scripts/colab_stage1.py push-prepared
```

`push-prepared` transfers only selected real images, YOLO labels, original masks and audit metadata, in bounded 32 MB Colab CLI uploads. It verifies the chunks and archive remotely, safely extracts the subset, and updates only the dataset root in `data.yaml`. The training gate re-verifies image pixels, label and mask bytes, exact split membership, review sheets and manifest on Colab before fitting. Preparation refuses to mix files with an existing dataset directory; use a fresh output root when reproducing a new subset.

## Conversion and mandatory visual review

```bash
python3 scripts/colab_stage1.py prepare --background
```

The converter pairs the observed `<split>_img/<name>` with `<split>_seg_label/<name>` under the **same official split and parent**. It also recognizes the `det_img` / `seg_label` names used in Dryad's description. Positional sorting or cross-split basename matching is never used. Every selected image must have the same dimensions as its mask. Unexpected mask colors/values fail conversion rather than being silently thresholded.

For the binary nonzero foreground:

```text
x0 = minimum foreground x       x1 = maximum foreground x + 1
y0 = minimum foreground y       y1 = maximum foreground y + 1
cx = (x0 + x1) / (2W)           cy = (y0 + y1) / (2H)
bw = (x1 - x0) / W              bh = (y1 - y0) / H
YOLO label: 0 cx cy bw bh
```

The union of **all** foreground pixels is retained. Small disconnected regions are not silently removed. Boxes must be in bounds, contain all foreground pixels, and round-trip to the original half-open pixel coordinates within `1e-5` pixels.

Preparation creates four contact sheets containing **32 comparisons**, including original images, mask overlays and red YOLO boxes. They cover train/validation/test and the smallest/largest mask areas. Download all four sheets and `validation.json` into `dataset/stageOne/metadata/`:

```bash
mkdir -p dataset/stageOne/metadata/annotations
colab download --session cs539-stage1 /content/cs539_stage1/dataset/stageOne/metadata/validation.json dataset/stageOne/metadata/validation.json
# Repeat for review_01.jpg, review_02.jpg, review_03.jpg, review_04.jpg:
colab download --session cs539-stage1 /content/cs539_stage1/dataset/stageOne/metadata/annotations/review_01.jpg dataset/stageOne/metadata/annotations/review_01.jpg
```

Inspect **all 32** comparisons: mask polarity, filename correspondence, orientation, register coverage, and alignment. A dimension check cannot establish semantic alignment by itself. If any example is wrong, fix conversion, regenerate the sheets and review again. Only after actual visual review:

```bash
python3 scripts/colab_stage1.py approve --reviewer 'Your name; all 32 pairs inspected'
```

Approval hashes the manifest and review sheets. Training verifies those hashes, verifies every selected image, label and mask again, checks exact split-directory membership, checks for duplicate decoded images, and requires at least 30 reviewed examples. It refuses to run when approval is absent or the reviewed dataset has changed.

## Train and evaluate

```bash
python3 scripts/colab_stage1.py train --background
```

Monitor `/content/cs539_stage1/train.log` in the existing session. Background launch alone is not successful training. Require `artifacts/training_complete.json` and nonempty `best.pt` / `last.pt` before evaluating:

After training, checkpoint class-name metadata is normalized to `{0: register}`. Ultralytics' `single_cls` option can otherwise use the generic display name `item`. Normalization round-trips every checkpoint and verifies every model/EMA tensor is bitwise identical; no weights or optimizer values are changed. Its before/after hashes are saved in `checkpoint_class_names.json`. For already-trained checkpoints, this can also be run explicitly with `python3 scripts/colab_stage1.py normalize-checkpoints`.

```bash
python3 scripts/colab_stage1.py evaluate --background
```

Evaluation uses the untouched 800-image test subset and the validation-selected best checkpoint. It saves AP50, AP50–95, precision/recall, speed, prediction data, and visualizations. A separate fixed operating point (`confidence=0.25`, `IoU=0.50`) reports image localization recall. The prediction contact sheets include the eight worst-IoU examples as well as eight seeded random examples. Test data is not used for training or early stopping.

## Retrieve results into this repository

```bash
python3 scripts/colab_stage1.py package
python3 scripts/colab_stage1.py fetch
```

`fetch` downloads the bundle to **this local repository** at `outputs/stage1/`, verifies the remote bundle checksum, safely extracts its paths, and verifies each file hash. The bundle contains:

- `artifacts/runs/register_yolo11n/weights/best.pt` and `last.pt`, plus periodic checkpoints;
- epoch-level `results.csv`, training curves, arguments and validation plots;
- `heldout_metrics.json`, `operating_point.json` and per-image test predictions;
- reviewed annotation sheets and held-out prediction visualizations;
- exact split manifest, conversion checks, approval record, training configuration, GPU report, environment freeze and artifact checksums.

Training and evaluation source snapshots are retained separately with their hashes: `source_snapshot/` records the code used by the running training process, and `evaluation_source_snapshot/` records the evaluation implementation. Held-out inference is explicitly chunked into at most 32 images per call, with filename/order checks against the manifest.

The result bundle excludes the raw archive and image/mask/label dataset. New runs snapshot dataset audit metadata under `artifacts/dataset_metadata/`; these are historical records, not the live dataset. Reproduce the subset using the pinned source archive and manifest. Before any Git commit:

```bash
git check-ignore dataset/stageOne/images/train/example.jpg
git check-ignore outputs/stage1/artifacts/runs/register_yolo11n/weights/best.pt
git status --short
```

For the browser/local-preparation route, the raw archive remains locally at the verified download location; only the selected subset is sent to Colab. In either route, raw images are excluded from the retrieved result bundle.

## Use the trained detector

With PyTorch and the pinned Ultralytics version available, run:

```bash
python3 scripts/predict_register.py \
  --weights outputs/stage1/artifacts/runs/register_yolo11n/weights/best.pt \
  --image /path/to/a/real_meter_photo.jpg \
  --output outputs/stage1/prediction
```

This saves a visualization, all predicted boxes/confidences in the original image coordinate system, and the highest-confidence register crop for a future Stage 2. A missing detection is reported explicitly. This does not implement digit recognition or claim end-to-end reading accuracy.

## Local implementation checks

```bash
python3 -m unittest discover -s tests -v
```

These tests cover one-pixel boxes, edge pixels, disconnected masks, empty/full masks, invalid bounds, exact filename pairing, dimension mismatch, missing visual review, and invalidated review hashes. Their tiny array fixtures are not training data. Passing them does **not** replace inspection of the real Dryad annotations.

### Refactor verification (October 26, 2026)

The tests retain the original conversion/boundary checks and cover seeded split
selection, empty/duplicate exclusions, approval revocation, every CLI phase,
Colab package synchronization, transfer paths, checkpoint-normalization control
flow and result-bundle contents. JSON fixtures freeze the pre-refactor training
and evaluation arguments and the ordered subset/review selection.

A one-time differential check against the saved original script compared a small
fixture end to end: 103 preparation files matched byte for byte (excluding the
machine-specific root in `data.yaml`), as did inspection/download reports,
approval records with a fixed timestamp, fixed-threshold predictions and review
visualizations. Simulated YOLO evaluation produced matching metrics and prediction
sheets; result ZIP member names and non-provenance payloads matched. This is a
regression check on unit fixtures, not a new real-data experiment.

Existing datasets, checkpoints and experiment results are checked separately by
file hashes and left unchanged. Code-provenance hashes naturally change in future
runs: `source_checksums.json` now includes every package module, and the existing
checkpoint/evaluation source-hash fields identify their implementing modules.
No real GPU training, full-data redownload or real PyTorch checkpoint rewrite was
performed for this refactor. The local checkpoint test uses a serialization stub;
full T4 numerical/serialization equivalence has not been re-established by a new
experiment. Public CLI commands and artifact paths are preserved; import helpers
from the package rather than from the former monolithic script.
