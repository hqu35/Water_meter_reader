# Training support

`trainer.py` verifies NVIDIA T4 CUDA against a CPU matrix product and emits epoch
markers. `checkpoints.py` normalizes the single-class display name while comparing
all model tensors before/after serialization. Training itself is in `../obb/training.py`.
