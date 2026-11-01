"""OBB training, intentionally unreachable without separate human approval."""

import os
import subprocess
import sys

from .. import config as repository
from ..training.checkpoints import normalize_checkpoints
from ..training.trainer import report_completed_epoch, verify_gpu
from ..utils.io import sha256_file, write_json
from . import config
from .review import require_approval


def training_arguments(root):
    """Keep all numerical training and augmentation settings explicit."""
    settings = config.CONFIG
    return {
        "task": "obb",
        "data": str(config.write_yaml(root)),
        **{
            key: settings[key]
            for key in [
                "epochs",
                "imgsz",
                "batch",
                "workers",
                "seed",
                "optimizer",
                "lr0",
                "warmup_bias_lr",
                "warmup_epochs",
                "patience",
            ]
        },
        "device": 0,
        "deterministic": True,
        "project": str(root / "artifacts/runs"),
        "name": "register_yolo11n_obb",
        "exist_ok": False,
        "pretrained": True,
        "amp": False,
        "cache": False,
        "plots": True,
        "save": True,
        "save_period": 5,
        "single_cls": True,
        **{
            key: 0.0
            for key in [
                "mosaic",
                "mixup",
                "cutmix",
                "copy_paste",
                "erasing",
                "fliplr",
                "flipud",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "hsv_h",
                "hsv_s",
                "hsv_v",
            ]
        },
        "close_mosaic": 0,
    }


def train(root):
    require_approval(root)  # BEFORE GPU access, imports or checkpoint download.
    artifacts = root / "artifacts"
    if (artifacts / "training_config.json").exists():
        raise FileExistsError("Use a fresh run root for another experiment")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["YOLO_CONFIG_DIR"] = str(root / "ultralytics-settings")
    verify_gpu(root)
    import ultralytics
    from ultralytics import YOLO

    assert ultralytics.__version__ == "8.4.0"
    write_json(artifacts / "training_config.json", config.CONFIG)
    (artifacts / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    model = YOLO(config.CONFIG["model"], task="obb")
    assert model.task == "obb"
    model.add_callback("on_train_epoch_end", report_completed_epoch)
    write_json(
        artifacts / "pretrained.json",
        {"model": config.CONFIG["model"], "sha256": sha256_file(model.ckpt_path)},
    )
    checksums = repository.source_checksums()
    write_json(artifacts / "source_checksums.json", checksums)
    result = model.train(**training_arguments(root))
    write_json(
        artifacts / "training_complete.json",
        {
            "task": "obb",
            "save_dir": str(model.trainer.save_dir),
            "best": str(model.trainer.best),
            "last": str(model.trainer.last),
            "metrics": result.results_dict,
        },
    )
    normalize_checkpoints(root)
