"""Prepare orientation pairs or summarize a Label Studio JSON export.

The detector and rotation/classifier implementations are imported from Stage I.
Generated runs are isolated in a fresh output directory and can be resumed.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from ..config import REPO
from .integrated import MODEL_ID, run_one


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def select_images(directory, count, seed, frozen_manifest, source_name, split_name):
    """Seeded permutation, filtering identical decoded pixels and frozen IDs."""
    frozen = json.loads(frozen_manifest.read_text())["records"]
    excluded_names = {
        Path(r["source_image"]).name for r in frozen if r["split"] in ("train", "val")
    }
    excluded_pixels = {r["source_pixel_sha256"] for r in frozen}
    candidates = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    random.Random(seed).shuffle(candidates)
    selected = []
    seen_paths = set()
    seen_pixels = set(excluded_pixels)
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen_paths or path.name in excluded_names:
            continue
        seen_paths.add(resolved)
        try:
            with Image.open(path) as image:
                if image.getexif().get(274, 1) != 1:
                    continue
                pixels = np.asarray(image.convert("RGB"))
            pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
            if pixel_hash in seen_pixels:
                continue
            seen_pixels.add(pixel_hash)
            selected.append(
                {
                    "sample_id": f"sample_{len(selected) + 1:06d}",
                    "source_relative_path": str(path.relative_to(directory)),
                    "source_image_sha256": digest(path),
                    "source_pixel_sha256": pixel_hash,
                    "source": source_name,
                    "official_split": split_name,
                    "detector_split": "not_selected_train_or_validation",
                    "status": "pending",
                }
            )
            if len(selected) == count:
                break
        except (OSError, ValueError):
            continue
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} valid unique eligible images found; {count} required"
        )
    return selected


def task_url(output_name, name, base_url):
    relative = f"{output_name}/images/{name}"
    if base_url:
        return base_url.rstrip("/") + "/" + quote(relative, safe="/")
    return "/data/local-files/?d=" + quote(relative, safe="/")


def export_tasks(root, manifest, base_url):
    tasks = []
    for row in manifest["samples"]:
        if row["status"] != "success":
            continue
        sid = row["sample_id"]
        tasks.append(
            {
                "data": {
                    "sample_id": sid,
                    "image_reference": task_url(
                        root.name, f"reference/{sid}.png", base_url
                    ),
                    "image_prediction": task_url(
                        root.name, f"prediction_adjusted/{sid}.png", base_url
                    ),
                    "predicted_orientation": row["prediction"],
                    "applied_transform": row["applied_transform"],
                    "source": row["source"],
                    "official_split": row["official_split"],
                }
            }
        )
    write_json(root / "tasks.json", tasks)


def process(args):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("ROBOFLOW_API_KEY must be set")
    directory = args.input_dir.resolve()
    root = args.output.resolve()
    frozen = args.frozen_manifest.resolve()
    if not directory.is_dir() or not frozen.is_file() or not args.weights.is_file():
        raise ValueError("Input directory, frozen manifest or weights not found")
    if root == directory or directory in root.parents:
        raise ValueError("Output must not be inside the source image directory")
    metadata = {
        "seed": args.seed,
        "count": args.count,
        "input_dir": str(directory),
        "frozen_manifest_sha256": digest(frozen),
        "weights_sha256": digest(args.weights),
        "model": MODEL_ID,
        "url_base": args.url_base,
        "source": args.source,
        "split": args.split,
    }
    if root.exists():
        if not args.resume:
            raise FileExistsError(
                "Output exists; use --resume only for the same manifest"
            )
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["configuration"] != metadata:
            raise ValueError("Resume configuration differs from existing manifest")
    else:
        samples = select_images(
            directory, args.count, args.seed, frozen, args.source, args.split
        )
        root.mkdir(parents=True)
        (root / "images/reference").mkdir(parents=True)
        (root / "images/prediction_adjusted").mkdir(parents=True)
        manifest = {"configuration": metadata, "samples": samples}
        write_json(root / "manifest.json", manifest)
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="obb")
    consecutive_api_errors = 0
    for index, row in enumerate(manifest["samples"], 1):
        if row["status"] != "pending":
            continue
        source = directory / row["source_relative_path"]
        sid = row["sample_id"]
        reference = root / "images/reference" / f"{sid}.png"
        adjusted = root / "images/prediction_adjusted" / f"{sid}.png"
        try:
            if digest(source) != row["source_image_sha256"]:
                raise ValueError("Source image changed since selection")
            with tempfile.TemporaryDirectory(prefix="orientation-eval-") as tmp:
                worker_output = Path(tmp) / "sample"
                result = run_one(
                    model,
                    source,
                    args.weights.resolve(),
                    worker_output,
                    key,
                    device=args.device,
                )
                rectified = worker_output / "rectified_full.png"
                if rectified.exists():
                    if reference.exists() and digest(reference) != digest(rectified):
                        raise ValueError(
                            "Existing reference differs; refusing overwrite"
                        )
                    if not reference.exists():
                        shutil.copyfile(rectified, reference)
                    row.update(
                        reference_path=f"images/reference/{sid}.png",
                        calibration_angle=result.get("rotation_angle_degrees"),
                        calibration_affine=result.get("affine_matrix"),
                        detection_confidence=result.get("selected_detection", {}).get(
                            "confidence"
                        ),
                    )
                if result["status"] == "success":
                    orientation = result["orientation"]
                    result_image = worker_output / "corrected_full.png"
                    if adjusted.exists() and digest(adjusted) != digest(result_image):
                        raise ValueError(
                            "Existing adjusted image differs; refusing overwrite"
                        )
                    if not adjusted.exists():
                        shutil.copyfile(result_image, adjusted)
                    row.update(
                        status="success",
                        prediction=orientation["raw_class"],
                        normalized_prediction=orientation["normalized_class"],
                        model_confidence=orientation["confidence"],
                        prediction_adjusted_path=f"images/prediction_adjusted/{sid}.png",
                        applied_transform="rotate_180"
                        if result["rotation_180_applied"]
                        else "identity",
                    )
                else:
                    row.update(status=result["status"], error=result.get("error"))
                    if "orientation" in result:
                        row.update(
                            prediction=result["orientation"]["raw_class"],
                            normalized_prediction=result["orientation"][
                                "normalized_class"
                            ],
                            model_confidence=result["orientation"]["confidence"],
                        )
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__)
        write_json(root / "manifest.json", manifest)
        export_tasks(root, manifest, args.url_base)
        print(
            f"[{index}/{len(manifest['samples'])}] {sid}: {row['status']}", flush=True
        )
        if row["status"] == "orientation_unresolved" and row.get("error"):
            consecutive_api_errors += 1
        else:
            consecutive_api_errors = 0
        if consecutive_api_errors >= 3:
            print(
                "Stopped after three consecutive classifier errors; remaining samples stay pending.",
                flush=True,
            )
            break
    print(
        "Status counts:",
        dict(Counter(r["status"] for r in manifest["samples"])),
        flush=True,
    )


def evaluate(manifest_path, export_path, output_dir):
    manifest = json.loads(manifest_path.read_text())
    exported = json.loads(export_path.read_text())
    records = {r["sample_id"]: r for r in manifest["samples"]}
    verdicts = {}
    ambiguous = set()
    for task in exported:
        sid = task.get("data", {}).get("sample_id")
        if sid not in records or records[sid]["status"] != "success":
            continue
        observed = []
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                if result.get("from_name") == "prediction_verdict":
                    choices = result.get("value", {}).get("choices", [])
                    if len(choices) == 1 and choices[0] in ("Correct", "Incorrect"):
                        observed.append(choices[0])
        if observed and len(set(observed)) == 1:
            if sid in verdicts and verdicts[sid] != observed[0]:
                ambiguous.add(sid)
            verdicts[sid] = observed[0]
        elif observed:
            ambiguous.add(sid)
    for sid in ambiguous:
        verdicts.pop(sid, None)
    processed = [r for r in records.values() if r["status"] == "success"]
    correct = sum(v == "Correct" for v in verdicts.values())
    grouped = {}
    for label in sorted({r["normalized_prediction"] for r in processed}):
        matching = [r for r in processed if r["normalized_prediction"] == label]
        annotated = [
            verdicts[r["sample_id"]] for r in matching if r["sample_id"] in verdicts
        ]
        count_correct = annotated.count("Correct")
        grouped[label] = {
            "predictions": len(matching),
            "annotated": len(annotated),
            "correct": count_correct,
            "accuracy": count_correct / len(annotated) if annotated else None,
        }
    summary = {
        "total_sampled": len(records),
        "processed": len(processed),
        "annotated": len(verdicts),
        "correct": correct,
        "incorrect": len(verdicts) - correct,
        "accuracy": correct / len(verdicts) if verdicts else None,
        "unannotated": len(processed) - len(verdicts),
        "failed_or_pending": len(records) - len(processed),
        "conflicting_annotations": len(ambiguous),
        "by_predicted_class": grouped,
        "note": "Human judgment measures predictions conditional on successful calibration; calibration is not verified ground truth.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() or (output_dir / "summary.csv").exists():
        raise FileExistsError("Summary already exists")
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "predicted_class",
                "predictions",
                "annotated",
                "correct",
                "accuracy",
            ],
        )
        writer.writeheader()
        for label, values in grouped.items():
            writer.writerow({"predicted_class": label, **values})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--weights", type=Path, required=True)
    prep.add_argument(
        "--frozen-manifest",
        type=Path,
        default=REPO / "dataset/stageOneOBB/metadata/split_manifest.json",
    )
    prep.add_argument("--source", required=True)
    prep.add_argument("--split", required=True)
    prep.add_argument("--count", type=int, default=2000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--device", default="cpu")
    prep.add_argument("--url-base", default="")
    prep.add_argument("--resume", action="store_true")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--export", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 1:
            parser.error("--count must be positive")
        process(args)
    else:
        evaluate(args.manifest, args.export, args.output)


if __name__ == "__main__":
    main()
