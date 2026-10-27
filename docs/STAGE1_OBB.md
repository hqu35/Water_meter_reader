# Stage I — oriented register localization

Status: **3,600 real examples converted; explicit human approval pending; no GPU
training started.** Only the oriented detector is implemented. Future stages are
outside this workflow.

## Paths and entry point

| Path | Purpose |
|---|---|
| `configs/stage1_obb.json` | Frozen architecture, source identity and training/crop settings. |
| `dataset/wordWheelSource/` | Original Dryad ZIP plus format-neutral `selection.json`. |
| `dataset/stageOneOBB/` | Exported images/masks/OBB labels and audit metadata. |
| `artifacts/annotation/` | Four JPEG review sheets and browser gallery; Git-visible. |
| `outputs/stage1_obb/` | Ignored training outputs, metrics, checkpoints and implementation evidence. |
| `/content/cs539_stage1_obb` | Remote run root on the existing `cs539-stage1` session. |

Run from the repository root. Commands default to OBB; the optional `obb` prefix
remains accepted for compatibility with earlier OBB commands.

```bash
python -m pip install -r requirements-stage1-obb.txt
python -m stage1_python_worker --help
python -m stage1_python_worker validate
open artifacts/annotation/index.html   # macOS browser gallery
```

No model, raw dataset or generated metric files are committed. Review images and
source/config/test code are Git-visible. Source data and output directories remain
ignored. No documentation is inserted into exact-membership dataset leaves.

## Rebuilding from the original masks

The source inventory preserves the same deterministic 2,400 training / 400
validation / 800 test identities, seed 20261022, original image/mask byte and pixel
hashes, dimensions and official split identities. Validation uses the official
training pool; test remains exclusively official test data. No generated training
examples are introduced. The original ZIP is retained via a link to the downloaded
Dryad archive; deleting converted labels never deletes that external archive.

```bash
# A fresh root is required; existing converted data are never overwritten.
python -m stage1_python_worker --root /path/to/new-run prepare \
  --archive /path/to/Word-Wheel_Water_Meter_Dataset.zip \
  --selection /path/to/selection.json
python -m stage1_python_worker --root /path/to/new-run validate
```

The source inventory is also copied into `metadata/source_selection.json` and
bound by checksum to the derived manifest. Remote integrity checks therefore need
no raw ZIP. Changing source identities, official pool, source hashes, labels,
photos, masks or exact directory membership fails validation/review checks.
`download` and `inspect` remain available for the original segmentation source.

Each foreground pixel `(x,y)` occupies the cell `[x,x+1] × [y,y+1]` in edge
coordinates. The converter expands the foreground convex hull by a unit square
and applies OpenCV `minAreaRect`/`boxPoints`. This is the minimum-area rectangle of
all foreground cells, including disconnected and single-pixel components. Nothing
is discarded by a largest-component heuristic.

Corners are clockwise in image coordinates, with the long edge first and pointing
rightward (downward at the vertical tie). Corner zero is local top-left. A label is
one line `0 x1 y1 x2 y2 x3 y3 x4 y4`, normalized by exported width/height and written
to ten decimal places. Validation checks rectangle geometry, winding, normalized
range, serialization round-trip and coverage of all foreground cell corners.

89 real examples require minimal integer image padding because a fitted rotated
rectangle extends outside the original frame. RGB borders are replicated; masks
are zero-padded and labels shifted identically. Unpadded images/masks preserve
original encoded bytes. Padded photos use lossless PNG. Source sizes, offsets,
source hashes and derived hashes are recorded. No corners are clipped into a
non-rectangular quadrilateral just to normalize them.

## Human review — mandatory stopping point

Open **all four** sheets in `artifacts/annotation/`, directly or through
`index.html`: 16 train, 8 validation and 8 test examples, 32 total. Rows show
original photo, green mask overlay/red oriented rectangle with corner indices,
and the aligned crop. Deterministic selection includes the largest padding cases
in each split before randomly selecting other examples.

Check image/mask registration, edge digits and crop content. Some source images
already truncate the register at the frame (for example `train_train17696` on
sheet 01). Padding cannot reconstruct missing source information. A mask is the
original annotation, not proof that every printed digit was correctly annotated.

Only after explicit **human** inspection and approval:

```bash
python -m stage1_python_worker approve --reviewer 'YOUR NAME' --confirm-reviewed
```

`metadata/annotation_review.json` is absent while pending. It binds configuration,
manifest, validation and relocated sheet hashes, reviewer attestation and reviewed
count. Any later change invalidates the gate. An assistant's visual inspection
alone is not approval. Training checks the full evidence **before** GPU access,
checkpoint download or model creation. Remote training checks local approval
before launch and repeats the gate on the remote machine. This is a human workflow
attestation, not cryptographic proof of the person's identity.

Default local review sheets live at repository-level `artifacts/annotation/`.
For custom/remote roots they live at `<root>/artifacts/annotation/`. Approved subset
transfers include the sheets at that portable location and retain their hashes.

## Training and existing-session Colab workflow

Architecture: `yolo11n-obb.pt`, explicit `task='obb'`, Ultralytics 8.4.0. Training
settings: 30 epochs, image size 640, batch 32, workers 2, seed 20261022, deterministic,
AdamW, LR 0.001, warmup 3 epochs, warmup bias LR 0, patience 10, AMP off, one class,
checkpoint save interval 5. All configured spatial/color/mixing/flipping
augmentations are disabled. No new rotation augmentation or angle-loss override
is enabled. YOLO's own OBB task supplies rotated assignment, losses and validation.
Best checkpoint selection uses validation fitness, never held-out test performance.

The following are **future commands**, not completed training:

```bash
python -m stage1_python_worker colab status
python -m stage1_python_worker colab sync
colab exec --session cs539-stage1 <<'PY'
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
    '/content/cs539_stage1_obb/project/requirements-stage1-obb.txt'], check=True)
PY
# After explicit human approval, retaining Colab's CUDA-enabled PyTorch:
python -m stage1_python_worker colab push-prepared
python -m stage1_python_worker colab gpu
python -m stage1_python_worker colab train --background
# Wait for training to finish before evaluating:
python -m stage1_python_worker colab evaluate --background
# Wait for evaluation to finish before packaging:
python -m stage1_python_worker colab package
python -m stage1_python_worker colab fetch
```

Only `cs539-stage1` is used; no runtime provisioning/restart occurs. `gpu` requires
NVIDIA T4 and compares a small PyTorch CUDA matrix product to CPU. No CPU training
fallback exists. Source sync uploads the OBB package/config/dependencies. Approved
data are transferred in checksum-verified chunks and safely extracted. Background
logs live in the remote root. Fetch verifies the bundle and artifact hashes before
retrieving results locally. Existing datasets, bundles, run logs and prediction
outputs are protected against overwrite.

## Evaluation and inference contract

Evaluation invokes the native Ultralytics OBB validator on the held-out test split.
In version 8.4.0, native OBB AP matching uses `batch_probiou` (Gaussian/probabilistic
rotated similarity). Separately, exact float64 convex polygon intersection-over-
union at confidence 0.25 / IoU 0.5 produces precision/recall and per-image matches.
At most one prediction matches each register; duplicates count as false positives.
The exact polygon diagnostic is distinct from native OBB AP. It uses no enclosing
rectangle overlap. Model metrics/checkpoints do not yet exist.

```bash
python -m stage1_python_worker predict \
  --weights outputs/stage1_obb/artifacts/runs/register_yolo11n_obb/weights/best.pt \
  --image /path/to/real_meter.jpg --output outputs/stage1_obb/prediction/example
```

`prediction.json` returns each detection's confidence, class and ordered corners
in pixels plus unclipped normalized prediction corners. The highest-confidence
register supplies `register_crop.png`; `prediction.jpg` is a polygon preview.
No detections yields an empty list and `crop: null`. Non-oriented checkpoints and
non-normalized EXIF orientation are rejected.

The crop uses positive winding, a horizontal long axis, half-pixel sampling
corrections and a 3% margin on each side. Out-of-image sampling replicates borders
and is reported. Geometry includes the homography, sampling corners, dimensions,
near-square status and `orientation_ambiguous_180: true`. This avoids reflection
and arbitrary transposition, but masks/rectangles alone do not identify upright
text versus a 180-degree inversion. Incorrect predicted localization can still
miss digits and must be assessed after approved training.

## Validation evidence

`tests/test_stage1_obb.py` covers OBB conversion, normalized coordinates, corner
ordering, border padding, arbitrary-angle crop markers and edge contents, polygon
IoU, explicit hyperparameters, source-only reproducibility, stale approval and
Colab payloads. Generated test patterns never enter model training.

```bash
python -m unittest discover -s tests -v
python -m ruff check stage1_python_worker tests
```

Local model/Colab interfaces are stub-tested; no native-model accuracy, live GPU
execution or live transfer result is claimed. Updated evidence lives in ignored
`outputs/stage1_obb/implementation_validation/`.

References: [OBB labels](https://docs.ultralytics.com/datasets/obb/),
[pinned OBB validator](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/models/yolo/obb/val.py),
[OpenCV geometry](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
