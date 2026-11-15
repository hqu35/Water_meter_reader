# Phase II Experiment 0 — zero-shot digit OBBs

This is a feasibility experiment, not a training run or a digit-reading metric.
See [EXPERIMENT_0.md](EXPERIMENT_0.md) for the initial results and limitations.
The publisher's [model card](https://huggingface.co/ntarekp/watermeter-yolov8-obb)
describes the stage-two model as a YOLOv8-OBB digit detector trained on cropped,
upscaled register windows. The checkpoint identifies 11 classes: `0`–`9` and
`u` (unclear); its training arguments record `imgsz=416`. Predictions use
Ultralytics 8.4.0, `imgsz=416`, confidence 0.10 and standard class-aware OBB NMS
IoU 0.70. Coordinates refer to the original rectified crop dimensions. No extra
box grouping or deduplication is applied after Ultralytics inference; these are
post-NMS predictions, not raw network logits.

The checkpoint is [stage2_digits.pt](https://huggingface.co/ntarekp/watermeter-yolov8-obb/blob/main/stage2_digits.pt),
pinned to model repository revision `66b856c`; SHA-256:
`f5553aaba25b49b8cab4b74d5b87f0ad68779a889fe19dcbc3139439a0f3ef02`.
It is a third-party pickle-based checkpoint, not a trusted source-code artifact.
The experiment loaded it only in an OS sandbox with network access and user-file
reads blocked, an empty inherited environment, and writes limited to a scratch
directory. The model's file hash was checked before loading. The original run denied access
to user files and the network but allowed reads of other temporary files; a stricter
verification run restricted temporary-file contents to the experiment and Python
environment. This is not a formal security guarantee. Docker was unavailable, so
macOS `sandbox-exec` was used.

## Dataset and outputs

The 100 inputs were selected with seed 20261028 from 375 previously human-reviewed
orientation outputs marked `Correct`, not from the detector's own train/val/test
subset. We reused the already orientation-corrected full images; a trusted Phase I
checkpoint was then run once on each selected full image to obtain its register
crop. Sampling deliberately prioritizes rare dark, reflective, blurry, soil,
stained and clear scenario flags, which overlap; results are not representative
population estimates. A perceptual dHash filter rejects near-identical full images
at Hamming distance <= 2; it does not guarantee all near duplicates are removed.
The model is not trained on these samples in this experiment. Selection relies on
a saved human-review export, and human verdicts are not digit-level ground truth.

`outputs/exp0_pretrained_obb/` is ignored by Git and holds the model checkpoint,
`inputs/`, `selected_samples.json`, `raw_predictions.json`, `visualizations/`,
`evaluation_summary.md`, and `gallery/index.html`. The manifest records original
source paths, hashes, Phase I predicted OBB and affine/crop geometry. The gallery
supports good/partial/failed verdicts, notes, filtering, JSON export/import and
browser-local autosave. Manual review counts begin at zero; do not report mAP or
IoU without independently verified digit OBBs.

## Reproduce on macOS

Run from the repository root with the original Phase I assets and the previously
exported human review JSON available. Use an isolated Python 3.11 environment
outside the user home directory, so the sandbox can deny access to `/Users`.
The commands below create a **new** output and sandbox directory; no prior results
are overwritten. Replace the review/CSV paths if your copies differ.

```bash
python3.11 -m venv /private/tmp/phase2-repro-venv
/private/tmp/phase2-repro-venv/bin/pip install -r PhaseOne/requirements-stage1-obb.txt
/private/tmp/phase2-repro-venv/bin/python PhaseTwo/prepare_inputs.py \
  --output PhaseTwo/outputs/exp0_reproduction \
  --review-export /path/to/label-studio-export.json \
  --scenario-csv /path/to/detection/train/train_class_label_CSV.csv

PHASE2_DIR=$(mktemp -d /private/tmp/phase2-repro.XXXXXX)
mkdir -p "$PHASE2_DIR/runtime"
cp -R PhaseTwo/outputs/exp0_reproduction/inputs "$PHASE2_DIR/inputs"
curl -fsSL -o "$PHASE2_DIR/stage2_digits.pt" \
  https://huggingface.co/ntarekp/watermeter-yolov8-obb/resolve/66b856c/stage2_digits.pt
shasum -a 256 "$PHASE2_DIR/stage2_digits.pt"
cp PhaseTwo/sandbox_infer.py "$PHASE2_DIR/sandbox_infer.py"
cd "$PHASE2_DIR"
env -i PATH=/usr/bin:/bin OMP_NUM_THREADS=1 PHASE2_SANDBOX_ROOT="$PHASE2_DIR" \
  YOLO_CONFIG_DIR="$PHASE2_DIR/runtime" MPLCONFIGDIR="$PHASE2_DIR/runtime" \
  sandbox-exec -p "(version 1) (allow default) (deny network*) (deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") (subpath \"/private/tmp\")) (allow file-read-metadata (subpath \"/private/tmp\")) (allow file-read* (subpath \"$PHASE2_DIR\") (subpath \"/private/tmp/phase2-repro-venv\")) (deny file-write*) (allow file-write* (subpath \"$PHASE2_DIR\"))" \
  /private/tmp/phase2-repro-venv/bin/python "$PHASE2_DIR/sandbox_infer.py"
```

From the repository root, copy the sandbox's `raw_predictions.json` and
`visualizations/` into `PhaseTwo/outputs/exp0_reproduction/`, then run:

```bash
/private/tmp/phase2-repro-venv/bin/python PhaseTwo/build_gallery.py \
  --output PhaseTwo/outputs/exp0_reproduction
```

`sandbox_infer.py` verifies the SHA-256 itself and refuses an existing visualization
directory. Sample selection and crop preparation use local Phase I data and do not
invoke the hosted orientation API. Review all predictions before selecting a
checkpoint for fine-tuning; no generalization or digit accuracy is established here.
