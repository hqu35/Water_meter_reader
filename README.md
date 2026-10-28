# CS539 — Mechanical Water Meter Reading

Stage I localizes a mechanical meter's digit register with a trained YOLO11n-OBB
model. An independent integration experiment rotates the full image using the
predicted register, asks a hosted orientation classifier whether a 180° correction
is needed, and extracts the final register crop. Human review of orientation
predictions is ongoing; Stage II digit localization is future work.

Start with the [Stage I guide](docs/STAGE1_OBB.md) for architecture, experiments,
results and reproduction. The [integration guide](docs/STAGE1_INTEGRATION.md)
describes the separate full-image orientation experiment.

| Path | Purpose |
|---|---|
| `stage1_python_worker/obb/` | OBB geometry, preparation, review gate, training, evaluation and inference. |
| `stage1_python_worker/data/` | Dryad source inventory and dataset checks. |
| `configs/stage1_obb.json` | Frozen Stage I configuration. |
| `dataset/stageOneOBB/` | Local 2,400 train / 400 validation / 800 test subset, labels and audit metadata. |
| `dataset/wordWheelSource/` | Local Dryad ZIP link and frozen source-identity inventory. |
| `training loss plot/` | Git-tracked Stage I training and validation plots. |
| `outputs/` | Local experiment outputs, checkpoint, human-review data and results. |
| `tests/` | Geometry, data integrity, workflow and integration tests. |
| `notes/` | Supplementary study notes. |

Raw data, checkpoints, visual review images and human annotation files are kept
locally and excluded from Git. See the [dataset description](dataset/README.md)
for what must be backed up; a fresh clone alone does not contain the exact data,
review evidence, or trained weights.

Stage II and Stage III are planned for digit localization and digit-state
recognition respectively. No digit-reading accuracy is claimed by Stage I.

Potential future data sources include [RITIKA Water Meter Digit Detection](https://universe.roboflow.com/ritika-1gfha/water-meter-jbktv-7vz5k-axdg-r8weg)
for digit localization and [SCUT Water Meter Number Dataset](https://github.com/HCIILAB/Water-Meter-Number-DataSet)
for digit-state recognition. A proposed 20-class scheme distinguishes ten stable
and ten transitional wheel states; extracting and aligning individual digit crops
from sequence labels remains a feasibility question.
