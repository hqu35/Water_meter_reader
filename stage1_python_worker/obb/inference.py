"""Return oriented corners and a geometry-aligned crop for a real image."""

import numpy as np
from PIL import Image, ImageDraw

from ..utils.io import sha256_file, write_json
from . import config
from .geometry import aligned_crop, ordered_corners


def detections_from_result(result):
    if result.obb is None:
        raise ValueError(
            "Expected OBB output; non-oriented checkpoints are not accepted"
        )
    return [
        {
            "corners_xy": ordered_corners(corners).tolist(),
            "confidence": float(confidence),
            "class_id": int(class_id),
        }
        for corners, confidence, class_id in zip(
            result.obb.xyxyxyxy.cpu().numpy(),
            result.obb.conf.cpu().numpy(),
            result.obb.cls.cpu().numpy(),
        )
    ]


def predict(weights, image_path, output, device="cpu", confidence=0.25):
    from ultralytics import YOLO
    import ultralytics

    assert ultralytics.__version__ == "8.4.0"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Prediction output must be empty; previous results are preserved"
        )
    model = YOLO(str(weights), task="obb")
    assert model.task == "obb", "An OBB checkpoint is required"
    with Image.open(image_path) as source:
        assert source.getexif().get(274, 1) == 1, (
            "Normalize EXIF orientation before inference"
        )
        image = source.convert("RGB")
    result = model.predict(
        np.asarray(image)[:, :, ::-1].copy(),
        device=device,
        imgsz=config.CONFIG["imgsz"],
        conf=confidence,
        iou=0.7,
        verbose=False,
    )[0]
    detections = detections_from_result(result)
    detections.sort(key=lambda detection: detection["confidence"], reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "obb",
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "weights_sha256": sha256_file(weights),
        "detections": detections,
        "selection": "highest confidence register",
        "crop": None,
        "orientation_ambiguous_180": True,
    }
    painter = ImageDraw.Draw(image)
    for detection in detections:
        q = np.asarray(detection["corners_xy"])
        # Do not clamp out-of-image predictions into invalid rectangles.
        detection["corners_normalized_unclipped"] = (q / image.size).tolist()
        polygon = [tuple(point) for point in q]
        painter.line(polygon + [polygon[0]], fill="red", width=3)
    # Crop from pristine pixels, not the annotated preview.
    if detections:
        with Image.open(image_path) as source:
            crop, geometry = aligned_crop(
                np.asarray(source.convert("RGB")),
                detections[0]["corners_xy"],
                config.CONFIG["crop_margin_fraction"],
            )
        Image.fromarray(crop).save(output / "register_crop.png")
        report["crop"] = {"file": "register_crop.png", **geometry}
    image.save(output / "prediction.jpg", quality=95)
    write_json(output / "prediction.json", report)
    return report
