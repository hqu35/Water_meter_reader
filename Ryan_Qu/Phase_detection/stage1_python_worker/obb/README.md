# Stage I OBB worker

Run `python -m stage1_python_worker --help` from `Ryan_Qu/Phase_detection/`.

- `data.py` converts the frozen Dryad image/mask selection to YOLO OBB labels
  and checks prepared image/label pairs for the requested splits.
- `preview.py` renders optional annotation sheets with the `preview` command.
- `training.py` fine-tunes YOLO11n-OBB. `evaluation.py` reports Ultralytics OBB
  AP and separate exact polygon IoU diagnostics. Neither requires review files.
- `geometry.py` handles corner ordering, padding, rectification and polygon IoU.
  `inference.py` runs single-image register prediction.
- `colab.py` syncs the worker, transfers prepared data/results, and runs the
  same training and evaluation commands in the configured NVIDIA T4 session.

See the [Stage I guide](../../docs/STAGE1_OBB.md) for results and reproduction.
