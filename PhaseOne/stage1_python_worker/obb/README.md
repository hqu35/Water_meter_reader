# Oriented register worker

`geometry.py` defines rectangle/crop/IoU coordinates. `data.py` converts original
source ZIP image/mask pairs from a deterministic source inventory and validates
all derived files. `review.py` renders to `artifacts/annotation/` and enforces
explicit human approval. `training.py` fits YOLO11n-OBB; `evaluation.py` uses native
OBB AP plus exact polygon-IoU diagnostics. `inference.py` returns ordered corners,
confidence and aligned crops. `config.py` defines paths, `cli.py` dispatches commands,
and `colab.py` transfers approved data/results using the existing named session.

Read [the guide](../../docs/STAGE1_OBB.md) before training.

`orientation_pilot.py` is an explicitly invoked, standalone CPU experiment. It uses GT crops and pretrained PaddleOCR, without changing the normal detector operations.

`orientation_review.py` builds a standalone review page from existing pilot outputs using `orientation_review.html`. The HTML template and review outputs are kept locally and excluded from Git; restoring that template is required to run this optional legacy UI on a fresh clone. Model predictions and dataset labels stay unchanged.

`integrated.py` is the independent full-image rectification and hosted-orientation
experiment. `orientation_evaluation.py` prepares or summarizes local human-review
data; its Label Studio configuration and generated artifacts are Git-ignored.
