<!-- recursive-summary:start -->
## Directory overview

Run the configured YOLO fine-tuning experiment on a verified T4 and maintain saved checkpoint metadata.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [\_\_init\_\_.py](__init__.py) | Declares the training subpackage. |
| [checkpoints.py](checkpoints.py) | Normalizes the single-class display name to register, round-trips model/EMA tensors to check equality and atomically replaces the checkpoint while recording hashes. |
| [trainer.py](trainer.py) | Checks the review gate and T4 CUDA calculation, builds explicit training arguments, runs pretrained YOLO11n fine-tuning and saves configuration, environment, provenance and completion records. |

### Summary

The trainer owns orchestration and reproducible arguments; Ultralytics owns the network, losses, backpropagation and best-checkpoint selection. Metadata normalization is a separate post-training operation and is not another training run.

<!-- recursive-summary:end -->
