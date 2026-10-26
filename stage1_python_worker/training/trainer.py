"""Verify the T4 runtime and fine-tune YOLO with the original experiment settings."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .. import config
from ..data.review import require_annotation_review
from ..utils.io import sha256_file, write_json
from .checkpoints import normalize_checkpoints


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


def training_arguments(run_root: Path) -> dict:
    """Keep every YOLO argument explicit so experiment settings are auditable."""
    settings = config.CONFIG
    return {
        "data": str(config.write_dataset_yaml(run_root)),
        "epochs": settings["epochs"],
        "imgsz": settings["imgsz"],
        "batch": settings["batch"],
        "device": 0,
        "workers": settings["workers"],
        "seed": settings["seed"],
        "deterministic": True,
        "optimizer": settings["optimizer"],
        "lr0": settings["lr0"],
        "warmup_bias_lr": settings["warmup_bias_lr"],
        "warmup_epochs": settings["warmup_epochs"],
        "patience": settings["patience"],
        "project": str(config.artifacts_dir(run_root) / "runs"),
        "name": "register_yolo11n",
        "exist_ok": False,
        "pretrained": True,
        "amp": False,
        "cache": False,
        "plots": True,
        "save": True,
        "save_period": 5,
        "single_cls": True,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
    }


def report_completed_epoch(trainer) -> None:
    """Emit the existing log marker used to monitor background Colab training."""
    print("STAGE1_EPOCH", trainer.epoch + 1, flush=True)


def train_model(run_root: Path) -> None:
    """Enforce review, verify GPU, fit once and normalize checkpoint names."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["YOLO_CONFIG_DIR"] = str(run_root / "ultralytics-settings")
    require_annotation_review(run_root)
    verify_gpu(run_root)

    import ultralytics
    from ultralytics import YOLO

    assert ultralytics.__version__ == "8.4.0", ultralytics.__version__
    artifacts = config.artifacts_dir(run_root)
    write_json(artifacts / "training_config.json", config.CONFIG)
    environment = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (artifacts / "pip-freeze.txt").write_text(environment)

    model = YOLO(config.CONFIG["model"])
    model.add_callback("on_train_epoch_end", report_completed_epoch)
    write_json(artifacts / "source_checksums.json", config.source_checksums())
    write_json(
        artifacts / "pretrained.json",
        {
            "model": config.CONFIG["model"],
            "sha256": sha256_file(model.ckpt_path),
        },
    )
    result = model.train(**training_arguments(run_root))
    write_json(
        artifacts / "training_complete.json",
        {
            "save_dir": str(model.trainer.save_dir),
            "best": str(model.trainer.best),
            "last": str(model.trainer.last),
            "metrics": result.results_dict,
        },
    )
    normalize_checkpoints(run_root)
    print("TRAINING_COMPLETE", flush=True)
