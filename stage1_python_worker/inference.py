"""Locate a register in one photograph and export its crop and coordinates."""

import json
import math
from pathlib import Path

from . import config
from .utils.io import write_json


def predict_register(
    weights: Path,
    image_path: Path,
    output: Path,
    device: str = "cpu",
    confidence: float = 0.25,
) -> dict:
    """Apply a trained checkpoint and select the highest-confidence register."""
    from PIL import Image
    from ultralytics import YOLO

    assert weights.is_file() and image_path.is_file()
    output.mkdir(parents=True, exist_ok=True)
    prediction = YOLO(str(weights)).predict(
        source=str(image_path),
        conf=confidence,
        imgsz=config.CONFIG["imgsz"],
        device=device,
        verbose=False,
    )[0]
    prediction.save(filename=str(output / "prediction.jpg"))
    image = Image.open(image_path).convert("RGB")
    boxes = prediction.boxes.xyxy.cpu().tolist()
    scores = prediction.boxes.conf.cpu().tolist()
    best = max(range(len(scores)), key=lambda index: scores[index]) if scores else None
    record = {
        "source": str(image_path),
        "confidence_threshold": confidence,
        "boxes_xyxy": boxes,
        "confidences": scores,
        "register_found": best is not None,
    }
    if best is not None:
        left, top, right, bottom = boxes[best]
        crop = [
            max(0, math.floor(left)),
            max(0, math.floor(top)),
            min(image.width, math.ceil(right)),
            min(image.height, math.ceil(bottom)),
        ]
        image.crop(crop).save(output / "register_crop.png")
        record.update(
            selected_box=boxes[best], selected_confidence=scores[best], crop_xyxy=crop
        )
    write_json(output / "prediction.json", record, sort_keys=False)
    print(json.dumps(record, indent=2))
    return record
