# Stage I: Register localization with YOLO-OBB

Stage I finds the digit register in a full mechanical water-meter photo. The
detector returns four oriented corners and a confidence score. OpenCV uses the
register's long axis to make it horizontal; the axis alone does not tell us
which way the digits face.

The intended pipeline is **detect the register → rotate the full frame with
OpenCV → classify its orientation with our locally trained ViT-Small → apply a
180° correction if needed → crop the register**. The team tested detection,
full-frame rectification, and local orientation classification together in a
[100-image pilot](../../Phase_orientation_classifier/README.md#separate-100-image-review).
The pilot saved predictions without applying the classifier's correction. The
detector's `predict` command is also standalone: it returns a rotation-aligned
crop, but does not resolve the 180° ambiguity. Single-digit detection and meter
reading are outside the scope of this detector.

## Data and labels

The frozen subset contains **2,400 train**, **400 validation**, and **800
held-out test** images from the Dryad Word-Wheel Water Meter Dataset. Training
and validation were selected from the official training pool; test used the
official test pool. The selection seed was `20261022`. No synthetic images
were used.

For each exact image/mask pair, OpenCV fits a minimum-area rectangle enclosing
the foreground pixels. Boundary cases are padded before normalization. Each
label is one class-0 register followed by four ordered, normalized corners:
`0 x1 y1 x2 y2 x3 y3 x4 y4`. The frozen source inventory and split manifest
record the identities used in the experiment.

Preparation checks the source splits, image/mask pairing and dimensions,
and OBB geometry as it converts the data. Before training or evaluation, the
worker checks that the requested splits contain the intended image/label pairs. Neither requires approval files. Historical
[annotation sheets](../register_BB_annotation_inspection/annotation/README.md)
remain available for inspection.

## Training and held-out results

The team fine-tuned `yolo11n-obb.pt` on a Colab **NVIDIA T4** with Ultralytics
**8.4.0** for **30 epochs**. The [configuration](../configs/stage1_obb.json)
records the settings: 640-pixel inputs, batch size 32, AdamW with initial
learning rate 0.001, three warmup epochs, and patience 10. AMP and the configured
augmentations were disabled. Validation OBB fitness selected `best.pt`; the
held-out test set did not select the checkpoint.

| Test images | Precision | Recall | OBB mAP@0.50 | OBB mAP@0.50–0.95 |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 0.9942 | 0.9900 | 0.9949 | 0.8527 |

Ultralytics computes these AP metrics with probabilistic rotated IoU. The
saved exact polygon-IoU operating-point report is a separate diagnostic.
Neither measures orientation accuracy or digit-reading accuracy. See the
[training and validation curves](../Stage%201%20OBB%20Experiment%20Results/stage1_obb_results.png).

## Reproducibility and local artifacts

Run the commands below from `Ryan_Qu/Phase_detection/`. The original detector
checkpoint is located at
`outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt`.
The same run's `artifacts/` directory holds the test metrics, logs, GPU and
environment details, and frozen dataset inventory. Back up the raw archive,
source inventory, split manifest, and checkpoint separately: they are not
included in a fresh Git clone.

**Prepare a fresh run.** Supply the original ZIP and frozen `selection.json`;
`prepare` reuses those identities. To regenerate an existing prepared dataset,
rerun the same command with `prepare --replace`. It builds the new dataset first
and replaces the generated image, mask, and label directories, manifests, and
YOLO YAML. The input ZIP, selection file, and other files stay in place. Put
`--root` before the subcommand.

```bash
python -m pip install -r requirements-stage1-obb.txt
RUN=outputs/stage1_obb/reproduction
SOURCE=dataset/Dryad_archive_original_dataset

python -m stage1_python_worker --root "$RUN" prepare \
  --archive "$SOURCE/Word-Wheel_Water_Meter_Dataset.zip" \
  --selection "$SOURCE/selection.json"
python -m stage1_python_worker --root "$RUN" validate
# Optional: render annotation and crop previews to a new directory.
python -m stage1_python_worker --root "$RUN" preview --output outputs/obb_preview
```

Adjust `SOURCE` to your local archive location. The prepared data for this
command goes in `$RUN/dataset/stageOneOBB/`. Without `--root`, the worker still
expects `dataset/stageOneOBB/`; it does not automatically find the renamed
`dataset/stage_I_Register_OBB/` folder. This guide rebuilds from the frozen
inventory rather than changing the archived dataset.

**Train and evaluate on Colab.** Use the existing authorized `cs539-stage1`
T4 session. After syncing, make sure the runtime has the dependencies in
`requirements-stage1-obb.txt` while retaining Colab's CUDA PyTorch build. The
worker uses `/content/cs539_stage1_obb` on Colab; it refuses to overwrite an
existing prepared dataset or run. Preserve old results and use a runtime where
that remote working path is unused for a fresh run.

```bash
python -m stage1_python_worker colab status
python -m stage1_python_worker colab sync
python -m stage1_python_worker --root "$RUN" colab push-prepared
python -m stage1_python_worker colab gpu
python -m stage1_python_worker colab train --background
# After training completes successfully:
python -m stage1_python_worker colab evaluate --background
# After evaluation completes successfully:
python -m stage1_python_worker colab package
python -m stage1_python_worker --root "$RUN" colab fetch
```

Background commands write `train.log` and `evaluate.log` under the Colab
working path. Fetch saves the result bundle and extracted `artifacts/` under
`$RUN`.

**Run detector-only inference.** Use an EXIF-normalized image and a new output
directory.

```bash
python -m stage1_python_worker predict \
  --weights outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt \
  --image /path/to/meter.jpg \
  --output outputs/stage1_obb/example
```

The command writes `prediction.jpg`, `prediction.json`, and, when a register is
detected, `register_crop.png`. For the local orientation model and its input
contract, see the [orientation classifier guide](../../Phase_orientation_classifier/README.md).
