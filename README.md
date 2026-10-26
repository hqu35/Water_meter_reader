# CS539 — Water Meter Localization and Recognition

A multi-stage computer vision pipeline for automatic localization and recognition of mechanical water meter readings.

**Course:** CS 539 — Introduction to Artificial Neural Networks, University of Wisconsin–Madison  
**Team:** Ryan Qu, Reilly Tom, Manas Kumar, Jiwoo Hong  
**Status:** Stage 1 baseline trained and evaluated; Stages 2–3 in feasibility development.

Stage 1 used YOLO11n on the existing `cs539-stage1` T4 session, with 2,400 real training images, 400 validation images and 800 held-out test images. All 32 annotation-review examples were inspected before training. Held-out **mAP@0.50: 0.9948; mAP@0.50–0.95: 0.7838**. Checkpoints, metrics and figures were retrieved into the Git-ignored `outputs/stage1/` directory. See [results and limitations](docs/STAGE1_RESULTS.md) and the [reproducible Colab workflow](docs/STAGE1.md).

## Current code structure

All executable Stage I code lives in one package, `stage1_python_worker/`. Its only entry point
is **`python -m stage1_python_worker`**, run from the repository root. There is no `scripts/`
implementation or compatibility-wrapper directory.

```text
stage1_python_worker/
  __main__.py                 Package entry point
  cli.py                      All local/remote command parsing and dispatch
  config.py                   Experiment JSON, paths and source inventory
  data/                       Source inspection, conversion, splitting and review
  training/                   T4 checks, YOLO training and checkpoint metadata
  evaluation/                 Held-out metrics and prediction sheets
  inference.py                Single-photo detection and register crop
  colab.py                    Upload/download and launch the same worker remotely
  utils/                      Shared JSON, hashes, ZIP extraction and result packaging
configs/stage1.json            Frozen Stage I experiment settings
requirements-stage1.txt        Dependencies; retain Colab's CUDA PyTorch build
tests/                        Data, execution and Colab regression tests
```

The one-time data-migration script has been removed because migration is already
complete. Colab source synchronization uploads this package, configuration and
requirements; remote phases also execute `python -m stage1_python_worker`.

- [`dataset/stageOne/`](dataset/stageOne/README.md): existing train/validation/test images, labels, masks and review metadata. Data stays ignored by Git; future stage datasets can be sibling directories.
- `outputs/stage1/`: existing weights, metrics, plots, logs and immutable source snapshots (ignored).
- `docs/`: preserved experiment records and results.
- `notes/`: preserved study notes.

`dataset/`, `docs/`, `notes/` and existing experiment outputs were left unchanged
by this consolidation. Historical commands or source copies inside those records
can refer to removed scripts; use the current commands below for execution.

## Running Stage I

```bash
# Inspect the available local phases and remote operations.
python3 -m stage1_python_worker --help
python3 -m stage1_python_worker colab --help

# Read the existing data source and apply an existing checkpoint locally.
python3 -m stage1_python_worker inspect
python3 -m stage1_python_worker predict \
  --weights outputs/stage1/artifacts/runs/register_yolo11n/weights/best.pt \
  --image /path/to/real_meter_photo.jpg \
  --output outputs/stage1/prediction

# Use only the existing named Colab session.
python3 -m stage1_python_worker colab status
python3 -m stage1_python_worker colab sync
python3 -m stage1_python_worker colab push-prepared
python3 -m stage1_python_worker colab train --background
python3 -m stage1_python_worker colab evaluate --background
python3 -m stage1_python_worker colab package
python3 -m stage1_python_worker colab fetch
```

The remote environment must have `requirements-stage1.txt` installed while keeping
its CUDA-enabled PyTorch. These are individual commands, not an automatically
executed sequence: wait for each remote job to finish successfully before starting
the next. `sync` uploads the full package before it is used remotely.

Data phases remain `download`, `inspect`, `prepare`, and `approve --reviewer NAME`.
The existing subset is already prepared and reviewed; `prepare` refuses to merge
with it. For a fresh run, use `python3 -m stage1_python_worker --root /path/to/run prepare`
after placing the verified ZIP under that run's `dataset/stageOne/raw/`.
Inspect all 32 review examples before recording approval. Training rechecks the
reviewed manifest, files and split membership before using the GPU.

`--root` precedes the command. The default local data path remains
`dataset/stageOne/` and outputs remain `outputs/stage1/`; a custom run root stores
data under `<root>/dataset/stageOne/`. Colab uses `/content/cs539_stage1` as its
run root. Training settings and the real-data split are unchanged.

```bash
python3 -m unittest discover -s tests -v
```

Tests use temporary fixtures and simulated Colab/YOLO calls. They do not provision
a GPU, launch real training, or replace existing experiment results.

## 1. Project Overview

Mechanical water meters are challenging to read automatically due to variations in lighting, viewing angles, reflections, and partially rotating digit wheels.

This project explores a modular neural network pipeline that identifies a water meter's numerical register, localizes individual digit positions, and recognizes stable or transitional digit states.

Our initial objective is to establish a reproducible baseline using **real-world datasets only**. Synthetic data augmentation may be investigated in a later phase but is excluded from the current prototype.

## 2. Proposed Architecture

```text
Input: Mechanical Water Meter Photograph
                   |
                   v
       [NN1] Register Localization
                 YOLO
                   |
                   v
           Register Region Crop
                   |
                   v
       [NN2] Digit Localization
                 YOLO
                   |
                   v
          Individual Digit Crops
                   |
                   v
       [NN3] Digit-State Recognition
           CNN / ResNet18
                   |
                   v
       Stable / Transitional States
                   |
                   v
      Optional Rule-Based Decoder
                   |
                   v
     Predicted Reading / Review Flag
```

### Stage 1 — Register Localization

Detect the numerical reading region within a photograph of a mechanical water meter.

- **Model:** Pretrained YOLO, fine-tuned for register detection.
- **Dataset:** Dryad Word-Wheel Water Meter Dataset.
- **Output:** Register bounding box and detection confidence.

The dataset provides segmentation masks. `stage1_python_worker/data/preparation.py` converts them into register bounding boxes after verifying image-mask alignment.

### Stage 2 — Individual Digit Localization

Detect the positions of individual digits within the cropped register.

- **Model:** A second, independently fine-tuned YOLO detector.
- **Dataset:** RITIKA Water Meter Digit Detection.
- **Output:** Individual digit bounding boxes and confidence scores.

The first two models can be trained independently using different datasets and connected during inference.

An important challenge is that character-level bounding boxes may not fully capture partially rotating mechanical digit wheels.

### Stage 3 — Digit-State Recognition

Classify individual digit-wheel images into stable or transitional states.

- **Proposed model:** ResNet18-based classifier.
- **Dataset:** SCUT Water Meter Number Dataset.
- **Output:** One of 20 digit-state classes.

The proposed classes are:

- **0–9:** Stable digit states.
- **10–19:** Transitional states between consecutive digits, including 9 → 0.

Unlike continuous rotation-angle estimation, this approach learns the visual states directly from labeled examples.

SCUT provides digit-sequence labels rather than individual wheel bounding boxes. Extracting reliable digit crops and matching them to the corresponding labels remains a key feasibility question.

## 3. Datasets and Resources

| Resource | Purpose |
|---|---|
| [Dryad Word-Wheel Water Meter Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.7d7wm3860) | Register localization |
| [RITIKA Water Meter Digit Detection](https://universe.roboflow.com/ritika-1gfha/water-meter-jbktv-7vz5k-axdg-r8weg) | Individual digit localization |
| [SCUT Water Meter Number Dataset](https://github.com/HCIILAB/Water-Meter-Number-DataSet) | Stable/transitional digit-state recognition |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO model training and inference |

Dataset licenses and redistribution restrictions should be checked before downloading, modifying, or publishing any data.

Raw datasets will not be committed to this repository.

## 4. Implementation Plan

### Phase 1 — Environment and Data Preparation

- Configure PyTorch and Ultralytics in Google Colab.
- Inspect dataset formats and annotation quality.
- Convert Dryad segmentation masks into YOLO detection labels.
- Verify RITIKA's bounding-box annotations.
- Investigate SCUT's label-to-digit-crop alignment.

### Phase 2 — Model Training

- Fine-tune a lightweight YOLO model for register detection.
- Fine-tune another YOLO model for digit localization.
- Train a 20-class digit-state classifier if reliable training crops can be obtained.

### Phase 3 — Integration

Connect the trained models into a unified inference pipeline:

1. Detect the numerical register.
2. Crop the detected register region.
3. Detect individual digit positions.
4. Extract digit-wheel images.
5. Predict stable or transitional states.
6. Optionally apply reading rules and flag uncertain predictions for review.

### Phase 4 — Evaluation

We plan to evaluate each neural component separately before analyzing end-to-end performance.

**Detection metrics:** mAP, precision, recall.

**Classification metrics:** Accuracy, macro-F1, and transition-state recall.

**Integration diagnostics:** Missing digits, incorrect crops, uncertain transitions, and error propagation between stages.

Quantitative end-to-end reading accuracy will only be reported if suitable jointly annotated evaluation images are available.

## 5. Current Milestones

| Date | Goal |
|---|---|
| October 2026 | Dataset inspection and YOLO familiarization |
| November 1, 2026 | Project proposal and preliminary feasibility results |
| November 2026 | Model training, integration, and controlled experiments |
| December 9, 2026 | Final project submission and demonstration |

## 6. Technical Challenges

The main anticipated challenges include:

- **Dataset compatibility:** Different annotation conventions across datasets.
- **Digit-wheel transitions:** Partially rotating digits may show two characters simultaneously.
- **Synthetic-to-real domain gap:** Relevant to possible future augmentation experiments.
- **Error propagation:** An inaccurate register crop may affect all later predictions.
- **Evaluation:** Standalone model accuracy does not necessarily reflect full-pipeline reliability.

## 7. Reproducibility

Stage I uses Python, PyTorch, Ultralytics and the existing Google Colab T4 session. Its fixed configuration is `configs/stage1.json`; the current execution interface and test command are documented above. Checkpoints and results remain in `outputs/stage1/`, and experiment records remain in `docs/`.

The eventual goal is to provide a runnable Colab notebook demonstrating the full inference pipeline.

---

*This repository is a course project under active development. Model architecture, datasets, and implementation details may change based on preliminary experiments.*
