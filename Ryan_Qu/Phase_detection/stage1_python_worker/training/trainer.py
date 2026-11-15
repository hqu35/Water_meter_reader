"""T4 verification and epoch reporting used by the oriented detector."""

import json
import subprocess
import sys
import time
from pathlib import Path
from .. import config
from ..utils.io import write_json


def verify_gpu(run_root: Path) -> None:
    """Require a T4 and compare a minimal CUDA matrix product against CPU."""
    import torch

    assert torch.cuda.is_available(), (
        "CUDA unavailable; CPU fallback is intentionally disabled"
    )
    gpu_name = torch.cuda.get_device_name(0)
    assert "T4" in gpu_name, gpu_name
    cpu_matrix = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    gpu_product = cpu_matrix.cuda() @ cpu_matrix.cuda().T
    torch.cuda.synchronize()
    assert torch.allclose(
        gpu_product.cpu(), cpu_matrix @ cpu_matrix.T, rtol=1e-5, atol=1e-3
    )
    report = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "cuda_matmul_passed": True,
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(config.artifacts_dir(run_root) / "gpu.json", report)
    print(json.dumps(report, indent=2))


def report_completed_epoch(trainer) -> None:
    """Emit the existing log marker used to monitor background Colab training."""
    print("STAGE1_EPOCH", trainer.epoch + 1, flush=True)
