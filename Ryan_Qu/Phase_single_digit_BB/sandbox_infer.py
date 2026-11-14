"""Run third-party digit OBB inference; execute only inside the restricted sandbox."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, __version__

ROOT = Path(
    os.environ.get("PHASE2_SANDBOX_ROOT", "/private/tmp/phase2_zero_shot")
).resolve()
MODEL = ROOT / "stage2_digits.pt"
EXPECTED = "f5553aaba25b49b8cab4b74d5b87f0ad68779a889fe19dcbc3139439a0f3ef02"
CONFIDENCE = 0.10
assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == EXPECTED
model = YOLO(str(MODEL), task="obb")
assert model.task == "obb" and model.names == {i: str(i) for i in range(10)} | {10: "u"}
files = sorted((ROOT / "inputs").glob("*.png"))
assert len(files) == 100
(ROOT / "visualizations").mkdir(exist_ok=False)
rows = []
for index, path in enumerate(files, 1):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    result = model.predict(
        rgb[:, :, ::-1].copy(),
        imgsz=416,
        conf=CONFIDENCE,
        iou=0.7,
        max_det=300,
        device="cpu",
        verbose=False,
    )[0]
    assert tuple(result.orig_shape) == (h, w)
    detections = []
    if result.obb is not None:
        for corners, score, class_id in zip(
            result.obb.xyxyxyxy.cpu().numpy(),
            result.obb.conf.cpu().numpy(),
            result.obb.cls.cpu().numpy(),
        ):
            corners = np.asarray(corners, dtype=float)
            class_id = int(class_id)
            assert corners.shape == (4, 2) and np.isfinite(corners).all()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": float(score),
                    "corners_xy": corners.tolist(),
                }
            )
    canvas = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(5, round(h * 0.15)))
    for number, item in enumerate(detections, 1):
        pts = [tuple(p) for p in item["corners_xy"]]
        draw.line(pts + [pts[0]], fill=(255, 34, 34), width=max(1, round(h / 50)))
        x = max(0, min(w - 1, min(p[0] for p in pts)))
        y = max(0, min(h - 1, min(p[1] for p in pts)))
        draw.text(
            (x, y),
            str(number),
            fill="yellow",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    canvas.save(ROOT / "visualizations" / path.name)
    rows.append(
        {
            "sample_id": path.stem,
            "input_image": "inputs/" + path.name,
            "visualization": "visualizations/" + path.name,
            "input_width": w,
            "input_height": h,
            "box_count": len(detections),
            "detections": detections,
        }
    )
    print(f"Inference {index}/100: {path.stem} {len(detections)} boxes", flush=True)
(ROOT / "raw_predictions.json").write_text(
    json.dumps(
        {
            "model_sha256": EXPECTED,
            "ultralytics_version": __version__,
            "input_imgsz": 416,
            "confidence_threshold": CONFIDENCE,
            "nms_iou": 0.7,
            "note": "Ultralytics standard class-aware OBB NMS applied; no additional grouping or deduplication.",
            "samples": rows,
        },
        indent=2,
    )
)
