# Oriented register worker

`geometry.py` defines rectangle/crop/IoU coordinates. `data.py` converts original
source ZIP image/mask pairs from a deterministic source inventory and validates
all derived files. `review.py` renders to `artifacts/annotation/` and enforces
explicit human approval. `training.py` fits YOLO11n-OBB; `evaluation.py` uses native
OBB AP plus exact polygon-IoU diagnostics. `inference.py` returns ordered corners,
confidence and aligned crops. `config.py` defines paths, `cli.py` dispatches commands,
and `colab.py` transfers approved data/results using the existing named session.

Read [the guide](../../docs/STAGE1_OBB.md) before training.
