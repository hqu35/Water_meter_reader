# Stage I full-image orientation integration pilot

This independent experimental entry point runs the trained OBB detector once per
image, rotates the entire image on an expanded canvas, sends that full image to
Roboflow `meter-orientation-classification-ugjff/3`, carries the predicted OBB
through any 180-degree correction, and extracts a crop from the corrected image.
The existing `predict` command and production inference are unchanged. No ground
truth is used during inference, no model is trained, and no labels are modified.

## Setup and usage

Install `requirements-stage1-integration.txt` in a Python environment with a
compatible PyTorch installation. Set `ROBOFLOW_API_KEY` in your environment
using your preferred secret manager. Do not put credentials in source files.
Each detected image may trigger a hosted API request and associated usage.

```bash
python -m pip install -r requirements-stage1-integration.txt
python -m stage1_python_worker.obb.integrated \
  --image /path/to/meter.jpg \
  --weights outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt \
  --output outputs/stage1_integration/my_single_run
```

Use `--input-dir /path/to/images` instead of `--image` for a directory. An output
path must be new. Optional `--device`, `--confidence`, and `--margin` parameters
control inference and padding. Missing detections, API failures, malformed
responses, and unrecognized classes produce failure metadata rather than silently
assuming an upright orientation. Additional detections are retained in the JSON;
the highest-confidence detection supplies the crop. Input EXIF orientation must
already be normalized. Only `Correct` and `upside_down` trigger crop generation;
other classifier labels remain unresolved. Confidence is not calibrated accuracy.

Each sample directory contains unannotated full-resolution rectified and corrected
images, annotated previews, the final crop, and a JSON record of detector output,
model/checkpoint hash, affine matrices, transformed corners, orientation response,
and status. Image and mask datasets and checkpoints are never overwritten.

## Ten-image verification

Open `outputs/stage1_integration/pilot_10/index.html` for the four-column visual
comparison. The selected validation IDs are recorded in `selection_manifest.json`.
Examples were chosen by evenly spaced orientation quantiles from the frozen
validation manifest; those GT angles were used for selection only, not detector
inference. The run uses source images from the Dryad archive. `results.csv`,
per-sample JSON, full-resolution PNGs, and `independent_crop_checks.json` provide
inspection evidence. Results remain ignored output artifacts.

The geometric tests cover 0, 45, 89, 90, 92, 135, and 179 degrees, expanded
canvas bounds, non-reflection, asymmetric markers, 180-degree pixel mapping,
unknown classes and missing detections:

```bash
python -m unittest discover -s tests -p 'test_stage1_integration.py' -v
```

The classifier may make mistakes, and a valid detector crop can still omit digits
if the detector predicts an incomplete box or the original image truncates them.
Border replication and padding cannot recover information outside the source.
A near-square box makes the long-axis convention ambiguous. A later 1,000-image
review set was prepared; the ten-image pilot does not establish orientation
accuracy. See [Stage I results](STAGE1_OBB.md#orientation-evaluation-and-limits).
