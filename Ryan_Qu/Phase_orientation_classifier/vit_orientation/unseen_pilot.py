"""Run a frozen, unused-ID Stage I -> ViT orientation pilot without training."""

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "Phase_orientation_classifier"
PHASE = ROOT / "Phase_detection"
ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:train_)?(train\d+)(?!\d)")
TEXT_TYPES = {".json", ".jsonl", ".csv", ".md", ".log", ".py", ".txt", ".html"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output, count, seed, input_dir):
    selection = output / "selection.json"
    if selection.exists():
        result = json.loads(selection.read_text())
        if result["seed"] != seed or len(result["samples"]) != count:
            raise ValueError("Existing pilot selection differs")
        return result
    excluded = collections.defaultdict(set)
    sources = []
    skipped = []
    for path in sorted(ROOT.parent.rglob("*")):
        if (
            not path.is_file()
            or path.is_relative_to(output)
            or any(
                x in path.parts
                for x in (".git", "__pycache__", "node_modules", "site-packages")
            )
        ):
            continue
        ids = set(ID_PATTERN.findall(path.name))
        if path.suffix.lower() in TEXT_TYPES:
            if path.stat().st_size <= 20_000_000:
                content = path.read_bytes()
                ids.update(ID_PATTERN.findall(content.decode("utf-8", errors="ignore")))
            else:
                skipped.append(str(path))
        if ids:
            relative = str(path.relative_to(ROOT.parent))
            for rid in ids:
                excluded[rid].add(relative)
            sources.append({"path": relative, "id_count": len(ids)})
    candidates = [
        p
        for p in sorted(input_dir.glob("train*.png"))
        if re.fullmatch(r"train\d+", p.stem) and p.stem not in excluded
    ]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} unused IDs available")
    picked = candidates[:count]
    output.mkdir(parents=True, exist_ok=False)
    samples = [
        {
            "id": p.stem,
            "index": i + 1,
            "original_path": str(p.resolve()),
            "original_sha256": digest(p),
            "official_split": "Dryad official train, unused IDs",
        }
        for i, p in enumerate(picked)
    ]
    result = {
        "schema": "watermeter-unused-id-pilot-selection-v1",
        "seed": seed,
        "count": count,
        "candidate_unused_images": len(candidates),
        "excluded_id_count": len(excluded),
        "samples": samples,
        "exclusion_scope": "All traceable IDs in repository sibling files, filenames and text artifacts; raw upstream class CSV is not used as usage evidence.",
    }
    (output / "exclusion_audit.json").write_text(
        json.dumps(
            {
                "excluded_ids": {k: sorted(v) for k, v in sorted(excluded.items())},
                "sources": sources,
                "oversized_text_files_not_read": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    selection.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(args):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    from ultralytics import YOLO
    from .model import OrientationPredictor

    sys.path.insert(0, str(PHASE))
    from stage1_python_worker.obb.inference import detections_from_result
    from stage1_python_worker.obb.integrated import rectify_full_image

    selection = freeze(args.output, args.count, args.seed, args.input_dir)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    weights = (
        PHASE
        / "outputs/stage1_obb/final_run/artifacts/runs/register_yolo11n_obb/weights/best.pt"
    )
    checkpoint = HOME / "outputs/vit_small_orientation_20261112/best.pt"
    detector_sha256 = digest(weights)
    assert (
        detector_sha256
        == "e7c03f5e8bd7e09b00fdd0d82f4d232f641ea576566f38b672503b5ad503e07e"
    )
    orientation_sha256 = digest(checkpoint)
    assert (
        orientation_sha256
        == "f87a5344c128f8bae49ef15f9c2345a86c27ef8ff29b95ecd6705712e7ae7bde"
    )
    config = {
        "seed": args.seed,
        "count": args.count,
        "stage1_checkpoint": str(weights),
        "stage1_sha256": detector_sha256,
        "stage1_device": "cpu",
        "stage1_imgsz": 640,
        "stage1_conf": 0.25,
        "stage1_iou": 0.7,
        "fallback": "If no box at 0.25, retry 0.05; record any failure; never use GT masks or silently replace a selected ID.",
        "orientation_checkpoint": str(checkpoint),
        "orientation_sha256": orientation_sha256,
        "orientation_device": "mps",
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "display_stage": "Full geometrically rectified image BEFORE any classifier-predicted 180-degree correction",
        "classes": {"0": "normal", "1": "upside_down"},
        "training_performed": False,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ["images/rectified", "records"]:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(weights), task="obb")
    predictor = OrientationPredictor(checkpoint)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    for row in selection["samples"]:
        record_path = args.output / "records" / (row["id"] + ".json")
        if record_path.exists():
            continue
        record = dict(row, status="failed", predicted_label=None)
        try:
            original = Path(row["original_path"])
            if digest(original) != row["original_sha256"]:
                raise ValueError("Original image changed after selection")
            with Image.open(original) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError("Unexpected EXIF orientation")
                rgb = np.asarray(im.convert("RGB"))
            boxes = []
            for threshold in [0.25, 0.05]:
                result = detector.predict(
                    rgb[:, :, ::-1].copy(),
                    imgsz=640,
                    conf=threshold,
                    iou=0.7,
                    device="cpu",
                    verbose=False,
                )[0]
                boxes = sorted(
                    detections_from_result(result),
                    key=lambda b: b["confidence"],
                    reverse=True,
                )
                if boxes:
                    break
            if not boxes:
                raise ValueError(
                    "Stage I detector returned no register; no GT fallback or replacement"
                )
            register_corners = np.asarray(boxes[0]["corners_xy"])
            rectified_rgb, rectified_corners, affine_matrix, angle = rectify_full_image(
                rgb, register_corners
            )
            image = Image.fromarray(rectified_rgb)
            rectified_path = args.output / "images/rectified" / (row["id"] + ".png")
            image.save(rectified_path)
            orientation, confidence = predictor.predict_orientation(image)
            record.update(
                status="success",
                predicted_label=int(orientation == 180),
                predicted_orientation="upside_down" if orientation == 180 else "normal",
                confidence=confidence,
                stage1_threshold=threshold,
                detection=boxes[0],
                rotation_angle_degrees=angle,
                affine_matrix=affine_matrix.tolist(),
                rectified_obb=rectified_corners.tolist(),
                rectified_image_path=str(rectified_path.relative_to(args.output)),
                rectified_sha256=digest(rectified_path),
                original_size_wh=[rgb.shape[1], rgb.shape[0]],
                rectified_size_wh=list(image.size),
                classifier_correction_applied=False,
            )
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        pending_record = record_path.with_suffix(".tmp")
        pending_record.write_text(json.dumps(record, indent=2) + "\n")
        pending_record.replace(record_path)
        print(
            f"{row['index']}/{args.count} {row['id']}: {record['status']} {record.get('predicted_orientation', '')}",
            flush=True,
        )
    records = [
        json.loads((args.output / "records" / (r["id"] + ".json")).read_text())
        for r in selection["samples"]
    ]
    (args.output / "predictions.json").write_text(
        json.dumps({"config": config, "records": records}, indent=2) + "\n"
    )
    with (args.output / "predictions.csv").open("w", newline="") as f:
        keys = [
            "id",
            "original_path",
            "status",
            "predicted_orientation",
            "predicted_label",
            "confidence",
            "rectified_image_path",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "selected": len(records),
        "successful": sum(r["status"] == "success" for r in records),
        "failed": sum(r["status"] != "success" for r in records),
        "predictions": dict(
            collections.Counter(r.get("predicted_orientation") for r in records)
        ),
        "accuracy": "Not measured: human review pending",
        "unused_ids_available_at_selection": selection["candidate_unused_images"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/Users/ququ/Downloads/doi_10_5061_dryad_7d7wm3860__v20260130/Word-Wheel_Water_Meter_Dataset/detection/train/train_img"
        ),
    )
    run(p.parse_args())
