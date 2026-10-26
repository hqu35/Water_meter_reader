"""Evaluate the validation-selected checkpoint on the untouched real test subset."""

import json
from pathlib import Path
from typing import Iterator

from .. import config
from ..utils.io import read_json, sha256_file, write_json
from .visualization import save_prediction_sheets


def prediction_batches(
    model, records: list, image_root: Path, batch_size: int = 32
) -> Iterator:
    """Bound in-memory prediction batches and verify image/ground-truth ordering."""
    for start in range(0, len(records), batch_size):
        group = records[start : start + batch_size]
        predictions = list(
            model.predict(
                source=[str(image_root / record["image_file"]) for record in group],
                device=0,
                imgsz=config.CONFIG["imgsz"],
                batch=batch_size,
                conf=0.25,
                iou=0.7,
                stream=True,
                verbose=False,
            )
        )
        assert len(predictions) == len(group), "Prediction count mismatch"
        for record, prediction in zip(group, predictions):
            assert Path(prediction.path).name == record["image_file"], (
                "Prediction/ground-truth image ordering mismatch"
            )
            yield record, prediction


def match_prediction(record: dict, prediction) -> dict:
    """Compute the original best-box IoU at the separately fixed operating point."""
    import numpy as np

    boxes = prediction.boxes.xyxy.cpu().numpy()
    confidences = prediction.boxes.conf.cpu().numpy()
    ground_truth = np.array(record["box_xyxy"])
    overlaps = []
    for box in boxes:
        intersection = np.maximum(
            0,
            np.minimum(box[2:], ground_truth[2:])
            - np.maximum(box[:2], ground_truth[:2]),
        ).prod()
        union = (
            np.maximum(0, box[2:] - box[:2]).prod()
            + (ground_truth[2:] - ground_truth[:2]).prod()
            - intersection
        )
        overlaps.append(float(intersection / union))
    best_iou = max(overlaps, default=0)
    return {
        "id": record["id"],
        "image_file": record["image_file"],
        "gt_box": record["box_xyxy"],
        "boxes": boxes.tolist(),
        "confidences": confidences.tolist(),
        "best_iou": best_iou,
        "matched": best_iou >= 0.5,
    }


def operating_point_summary(records: list) -> dict:
    """Summarize image localization at confidence .25 and matching IoU .50."""
    localized = sum(record["matched"] for record in records)
    return {
        "confidence": 0.25,
        "iou": 0.5,
        "images": len(records),
        "localized": localized,
        "image_localization_recall": localized / len(records),
        "missed": sum(not record["matched"] for record in records),
        "total_predictions": sum(len(record["boxes"]) for record in records),
    }


def evaluation_arguments(run_root: Path) -> dict:
    """Keep low-confidence AP evaluation separate from the .25 operating point."""
    return {
        "data": str(config.write_dataset_yaml(run_root)),
        "split": "test",
        "device": 0,
        "imgsz": config.CONFIG["imgsz"],
        "batch": config.CONFIG["batch"],
        "workers": config.CONFIG["workers"],
        "plots": True,
        "save_json": True,
        "project": str(config.artifacts_dir(run_root) / "evaluation"),
        "name": "heldout_test",
        "exist_ok": False,
        "conf": 0.001,
        "iou": 0.7,
    }


def evaluate_model(run_root: Path) -> None:
    """Run test AP, fixed-threshold matching and prediction visualizations."""
    from ultralytics import YOLO

    artifacts = config.artifacts_dir(run_root)
    completed = read_json(artifacts / "training_complete.json")
    model = YOLO(completed["best"])
    assert model.names == {0: "register"}, "Run normalize-checkpoints before evaluation"
    result = model.val(**evaluation_arguments(run_root))
    write_json(
        artifacts / "heldout_metrics.json",
        {
            "split": "test",
            "source": "official Dryad detection test pool only",
            "count": config.CONFIG["test_count"],
            "selection": "never used for fitting, early stopping or threshold selection",
            "metrics": result.results_dict,
            "speed": result.speed,
            "checkpoint_sha256": sha256_file(completed["best"]),
            "evaluation_source_sha256": sha256_file(__file__),
        },
    )
    manifest = read_json(config.metadata_dir(run_root) / "split_manifest.json")
    test_records = [
        record for record in manifest["records"] if record["split"] == "test"
    ]
    image_root = config.dataset_dir(run_root) / "images/test"
    predictions = [
        match_prediction(record, prediction)
        for record, prediction in prediction_batches(model, test_records, image_root)
    ]
    assert len(predictions) == len(test_records) == config.CONFIG["test_count"], (
        "Incomplete held-out predictions"
    )
    write_json(
        artifacts / "test_predictions.json",
        {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "records": predictions,
        },
    )
    save_prediction_sheets(predictions, image_root, artifacts)
    write_json(artifacts / "operating_point.json", operating_point_summary(predictions))
    print("EVALUATION_COMPLETE", json.dumps(result.results_dict), flush=True)
