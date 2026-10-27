# Stage I PaddleOCR orientation pilot

This is a standalone CPU experiment using the official pretrained
`PP-LCNet_x0_25_textline_ori` model, with no training or production integration.
It does not modify Stage I training, evaluation, packaging, splits or labels.

## Run with one CLI command

Use Python 3.12 and a separate environment; do not install these dependencies over
the Stage I CUDA environment. PaddleX 3.1 imports older LangChain module paths,
so the optional pilot requirements pin compatible versions explicitly.

```bash
python3.12 -m venv /tmp/cs539-orientation-pilot-env
/tmp/cs539-orientation-pilot-env/bin/python -m pip install -r requirements-orientation-pilot.txt
# From the repository root, run the complete 100-sample experiment:
/tmp/cs539-orientation-pilot-env/bin/python -m stage1_python_worker orientation-pilot \
  --crop-source ground-truth
```

Every run creates a fresh timestamped directory under
`outputs/pilots/paddle_orientation/`. `--output /path/to/new-pilot` is optional;
existing destinations are rejected. The command explicitly requires
`--crop-source ground-truth`. No OBB inference results/checkpoints were available
for this pilot, so no inference-box source is supported or silently substituted.

## Selection, rectification and prediction

Read the existing manifest, keep training examples only, sort by ID using Unicode
lexicographic ascending order and take the first 100. This is string ordering,
not natural numeric ordering. The observed selected IDs start at
`train_train10005` and end at `train_train11225`. No split or label is rewritten.

Read the actual four-corner GT label paired with each photo, check its hash and
photo hash/dimensions, and call the existing `geometry.aligned_crop` with the Stage I
3% margin. The manifest checksum and selected identity/hash records are saved.

Invoke PaddleOCR's `TextLineOrientationClassification` explicitly with model name
`PP-LCNet_x0_25_textline_ori`, CPU, two threads and MKL-DNN disabled. The official
PaddleX BOS model download/cache is inside this pilot's `model_cache/`. The model
labels `0_degree` and `180_degree` must agree with class IDs 0 and 1 respectively;
unknown labels or invalid scores fail that sample.

For 180°, use Pillow `ROTATE_180`, an exact pixel permutation without interpolation.
For 0°, preserve all pixels. Save separate lossless PNGs and verify each saved
correction equals its expected pixel transform. There is no confidence threshold:
every valid 180° prediction is rotated, exactly as specified.

Sample-level exceptions are written to CSV and processing continues without
replacing the failed identity. Setup failures are separate from sample inference
failures. `status=success` means the inference/save operation completed, not that
the orientation is correct.

## Outputs and manual review

The delivered run is `outputs/pilots/paddle_orientation/gt_train100_003/`:

| File/directory | Contents |
|---|---|
| `index.html` | Browser gallery linking all ten comparison sheets. |
| `comparison/comparison_01.jpg` … `10.jpg` | Ten rows each: ID, original/corrected crop, predicted orientation and confidence. |
| `original/` | 100 native-resolution rectified GT crops. |
| `corrected/` | 100 corresponding PNGs after the prescribed correction. |
| `predictions.csv` | All 100 attempts, image paths, explicit crop source, raw orientation/score, crop paths, status/error. Crop paths are relative to the pilot directory. |
| `raw_predictions.json` | Native PaddleOCR result data and crop geometry for successful samples. |
| `selection.json` | Exact source manifest hash, selection rule and selected file hashes. |
| `summary.json` | Success/failure and predicted-flip counts, versions, configuration and implementation hashes. |
| `model_checksums.json`, `model_cache/` | Model file hashes and the downloaded official pretrained model. |
| `environment-lock.txt` | Complete installed Python distribution versions for this run. |
| `verification.json` | Final artifact checks and input-preservation verification. |

Open locally on macOS:

```bash
open outputs/pilots/paddle_orientation/gt_train100_003/index.html
```

Observed execution: **100 successful, 0 failed; 24 predicted 180°, 76 predicted 0°**.
These counts are not accuracy. No manually verified orientation ground truth is
available, and the scores are not calibrated probabilities of correctness.
Inspect all 100 pairs before deciding whether the model is useful for water meters.
For manual attention, sheet 01 includes `train_train10115` and sheet 08 includes
`train_train10889`; these retained crops warrant checking for inversion. These
notes are inspection prompts, not ground-truth labels or a scored evaluation.

All pilot artifacts remain together in the dedicated, Git-ignored output tree.
The existing Git-visible Stage I annotation sheets are unchanged. An earlier
`gt_train100_001/` contains a failed dependency setup; `002/` is the initial successful
run. Both are preserved rather than overwritten; `003/` is the final deliverable.

The complete test suite includes deterministic selection, orientation parsing,
exact rotation, forced per-sample failures and no-overwrite behavior. Production
model operations remain unchanged. Stop here for manual inspection; this command
is not called automatically by Stage I prediction or training.

Official reference: [PaddleOCR text-line orientation classifier](https://www.paddleocr.ai/v3.1.0/en/version3.x/module_usage/textline_orientation_classification.html).
