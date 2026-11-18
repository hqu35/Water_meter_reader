# Python Environment Setup

These files define dependencies for different Stage I workflows. Install only the environment needed for the task; they are not all GPU settings.

- `requirements-stage1.txt`: shared Stage I dependencies. In Colab, retain its preinstalled CUDA-enabled PyTorch.
- `requirements-stage1-obb.txt`: Stage I OBB preparation and training dependencies; includes the shared requirements and OpenCV. Use this for the Colab OBB workflow.

From `Ryan_Qu/Phase_detection/`, install the OBB environment with:

```bash
python -m pip install -r environment_setup/requirements-stage1-obb.txt
```
