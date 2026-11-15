"""Select reviewed Phase I samples and produce register crops, without network access."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "PhaseOne"
sys.path.insert(0, str(PHASE1))
from stage1_python_worker.obb.geometry import aligned_crop  # noqa: E402
from stage1_python_worker.obb.inference import detections_from_result  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "outputs/exp0_pretrained_obb"
SOURCE = PHASE1 / "outputs/orientation_evaluation/dryad_2000_seed42"
REVIEW = Path("/Users/ququ/Downloads/project-1-at-2026-10-28-04-10-4e3a6a67.json")
SCENARIOS = Path(
    "/Users/ququ/Downloads/doi_10_5061_dryad_7d7wm3860__v20260130/Word-Wheel_Water_Meter_Dataset/detection/train/train_class_label_CSV.csv"
)
SOURCE_DIR = SCENARIOS.parent / "train_img"
WEIGHTS = (
    PHASE1
    / "outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt"
)
SEED = 20261028
LABELS = [
    "clear",
    "blurry",
    "dial-stained",
    "soil-covered",
    "dark",
    "reflective",
    "six-digit",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference_hash(image):
    grey = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    tiny = cv2.resize(grey, (17, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((tiny[:, 1:] > tiny[:, :-1]).ravel()).tobytes()


def main():
    global OUTPUT, REVIEW, SCENARIOS, SOURCE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--review-export", type=Path, default=REVIEW)
    parser.add_argument("--scenario-csv", type=Path, default=SCENARIOS)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    REVIEW = args.review_export.resolve()
    SCENARIOS = args.scenario_csv.resolve()
    SOURCE_DIR = SCENARIOS.parent / "train_img"
    manifest_path = OUTPUT / "selected_samples.json"
    if manifest_path.exists() or (OUTPUT / "inputs").exists():
        raise FileExistsError(
            "Experiment inputs already exist; use a fresh output directory"
        )
    records = json.loads((SOURCE / "manifest_1000.json").read_text())["samples"]
    annotations = json.loads(REVIEW.read_text())
    good = set()
    for task in annotations:
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for result in ann.get("result", []):
                if result.get("from_name") == "prediction_verdict" and result.get(
                    "value", {}
                ).get("choices") == ["Correct"]:
                    good.add(task["data"]["sample_id"])
    with SCENARIOS.open(newline="") as stream:
        labels = {
            row[0]: dict(zip(LABELS, map(int, row[1:])))
            for row in csv.reader(stream)
            if len(row) == 8
        }
    pool = [
        r
        for r in records
        if r["sample_id"] in good
        and r["source_relative_path"] in labels
        and r["status"] == "success"
    ]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    quotas = [
        ("dark", 7),
        ("reflective", 12),
        ("blurry", 16),
        ("soil-covered", 25),
        ("dial-stained", 20),
        ("clear", 20),
    ]
    model = YOLO(str(WEIGHTS), task="obb")
    destination = OUTPUT / "inputs"
    destination.mkdir(parents=True)
    selected, used_ids, hashes, failed = [], set(), [], []

    def consider(record, group):
        if record["sample_id"] in used_ids:
            return False
        image_path = SOURCE / record["prediction_adjusted_path"]
        try:
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB"))
            sig = difference_hash(image)
            if any(
                sum((a ^ b).bit_count() for a, b in zip(sig, old)) <= 2
                for old in hashes
            ):
                return False
            result = model.predict(
                image[:, :, ::-1].copy(),
                imgsz=640,
                conf=0.25,
                iou=0.7,
                device="cpu",
                verbose=False,
            )[0]
            detections = sorted(
                detections_from_result(result),
                key=lambda d: d["confidence"],
                reverse=True,
            )
            if not detections:
                failed.append(
                    {"sample_id": record["sample_id"], "reason": "no Phase I detection"}
                )
                used_ids.add(record["sample_id"])
                return False
            crop, geometry = aligned_crop(image, detections[0]["corners_xy"], 0.03)
            path = destination / (record["sample_id"] + ".png")
            Image.fromarray(crop).save(path)
            selected.append(
                {
                    "sample_id": record["sample_id"],
                    "selection_group": group,
                    "scenario_labels": labels[record["source_relative_path"]],
                    "original_image_path": str(
                        SOURCE_DIR / record["source_relative_path"]
                    ),
                    "original_sha256": record["source_image_sha256"],
                    "phase1_corrected_image_path": str(image_path),
                    "phase1_corrected_sha256": sha(image_path),
                    "phase1_orientation_prediction": record["prediction"],
                    "human_orientation_verdict": "Correct",
                    "phase1_detector_confidence": detections[0]["confidence"],
                    "phase1_detector_obb_xy": detections[0]["corners_xy"],
                    "crop_geometry": geometry,
                    "input_image": f"inputs/{path.name}",
                    "input_sha256": sha(path),
                    "crop_width": crop.shape[1],
                    "crop_height": crop.shape[0],
                    "source_to_rectified_affine": record["calibration_affine"],
                    "orientation_transform": record["applied_transform"],
                }
            )
            used_ids.add(record["sample_id"])
            hashes.append(sig)
            print(f"Prepared {len(selected)}/100", flush=True)
            return True
        except Exception as exc:
            failed.append(
                {"sample_id": record["sample_id"], "reason": type(exc).__name__}
            )
            used_ids.add(record["sample_id"])
            return False

    for group, quota in quotas:
        added = 0
        for record in pool:
            if labels[record["source_relative_path"]][group] and consider(
                record, group
            ):
                added += 1
            if added >= quota:
                break
    for record in pool:
        if len(selected) >= 100:
            break
        consider(record, "remainder")
    if len(selected) != 100:
        raise RuntimeError(f"Only {len(selected)} valid crops; see partial inputs")
    report = {
        "seed": SEED,
        "selection": "Deterministic quota sampling from 375 human-reviewed Correct orientation outputs; scenario flags may overlap, group is selection order. Reject dHash distance <=2 among selected full images.",
        "annotation_export_sha256": sha(REVIEW),
        "scenario_csv_sha256": sha(SCENARIOS),
        "phase1_manifest_sha256": sha(SOURCE / "manifest_1000.json"),
        "phase1_checkpoint_sha256": sha(WEIGHTS),
        "sample_count": 100,
        "excluded_during_crop": failed,
        "samples": selected,
    }
    manifest_path.write_text(json.dumps(report, indent=2))
    print("100 selected crops ready")


if __name__ == "__main__":
    main()
