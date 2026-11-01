"""Ultralytics OBB AP plus explicitly separate exact polygon-IoU diagnostics."""

import json

from PIL import Image, ImageDraw

from ..utils.io import read_json, sha256_file, write_json
from . import config
from .data import validate_dataset
from .geometry import rotated_iou
from .inference import detections_from_result
from .review import require_approval


def match_detections(detections, truth, threshold=0.5):
    """One register per image; match at most one prediction, in score order."""
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    ious = [rotated_iou(d["corners_xy"], truth) for d in ordered]
    eligible = [index for index, iou in enumerate(ious) if iou >= threshold]
    tp = int(bool(eligible))
    return {
        "tp": tp,
        "fp": len(ordered) - tp,
        "fn": 1 - tp,
        "polygon_ious": ious,
        "matched_prediction_index": eligible[0] if eligible else None,
    }


def evaluate(root, weights=None):
    require_approval(root)
    manifest = validate_dataset(root)
    artifacts = root / "artifacts"
    weights = weights or __import__("pathlib").Path(
        read_json(artifacts / "training_complete.json")["best"]
    )
    destination = artifacts / "evaluation"
    if destination.exists():
        raise FileExistsError("Evaluation exists; refusing to overwrite results")
    import ultralytics
    from ultralytics import YOLO

    assert ultralytics.__version__ == "8.4.0"
    model = YOLO(str(weights), task="obb")
    assert model.task == "obb"
    args = {
        "data": str(config.write_yaml(root)),
        "task": "obb",
        "split": "test",
        "device": 0,
        "imgsz": config.CONFIG["imgsz"],
        "batch": config.CONFIG["batch"],
        "workers": config.CONFIG["workers"],
        "plots": True,
        "save_json": True,
        "project": str(destination),
        "name": "heldout_test",
        "exist_ok": False,
        "conf": 0.001,
        "iou": 0.7,
    }
    write_json(artifacts / "evaluation_arguments.json", args)
    result = model.val(**args)
    write_json(
        artifacts / "heldout_obb_metrics.json",
        {
            "task": "obb",
            "metrics": result.results_dict,
            "speed": result.speed,
            "matching": "Ultralytics 8.4.0 OBBValidator batch_probiou (probabilistic rotated IoU)",
            "checkpoint_sha256": sha256_file(weights),
            "manifest_sha256": sha256_file(
                config.metadata_dir(root) / "split_manifest.json"
            ),
            "test_images": sum(r["split"] == "test" for r in manifest["records"]),
        },
    )
    records = [r for r in manifest["records"] if r["split"] == "test"]
    predictions = []
    preview = destination / "polygon_diagnostics"
    preview.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 32):
        batch = records[start : start + 32]
        paths = [
            config.dataset_dir(root) / "images/test" / r["image_file"] for r in batch
        ]
        results = list(
            model.predict(
                [str(p) for p in paths],
                stream=True,
                device=0,
                imgsz=config.CONFIG["imgsz"],
                batch=32,
                conf=0.25,
                iou=0.7,
                verbose=False,
            )
        )
        assert len(results) == len(batch)
        for record, path, prediction in zip(batch, paths, results):
            assert __import__("pathlib").Path(prediction.path).name == path.name
            detections = detections_from_result(prediction)
            matched = match_detections(detections, record["corners_xy"])
            predictions.append(
                {"id": record["id"], "detections": detections, **matched}
            )
            if len(predictions) <= 32:
                image = Image.open(path).convert("RGB")
                draw = ImageDraw.Draw(image)
                for q, color in [(record["corners_xy"], "lime")] + [
                    (d["corners_xy"], "red") for d in detections
                ]:
                    polygon = [tuple(p) for p in q]
                    draw.line(polygon + [polygon[0]], fill=color, width=3)
                image.save(preview / (record["id"] + ".jpg"), quality=95)
    tp, fp, fn = [sum(p[key] for p in predictions) for key in ["tp", "fp", "fn"]]
    write_json(artifacts / "test_obb_predictions.json", predictions)
    write_json(
        artifacts / "polygon_operating_point.json",
        {
            "confidence": 0.25,
            "exact_polygon_iou_threshold": 0.5,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "matching": "float64 convex-polygon clipping; separate from native OBB AP",
        },
    )
    print(json.dumps(result.results_dict, indent=2))
