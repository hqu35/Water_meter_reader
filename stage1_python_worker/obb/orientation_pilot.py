"""Standalone pretrained text-line orientation pilot; never production inference."""

import csv
import importlib.metadata
import json
import math
import platform
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.io import read_json, sha256_file, write_json
from . import config
from .geometry import aligned_crop

MODEL_NAME = "PP-LCNet_x0_25_textline_ori"
FIELDS = [
    "example_id",
    "source_image_path",
    "crop_source",
    "predicted_orientation",
    "confidence",
    "original_crop_path",
    "corrected_crop_path",
    "status",
    "error",
]


def select_training_records(manifest):
    """Lexicographic ID ordering is explicit, stable and independent of file order."""
    records = sorted(
        (r for r in manifest["records"] if r["split"] == "train"),
        key=lambda record: record["id"],
    )
    if len(records) < 100 or len({r["id"] for r in records}) != len(records):
        raise ValueError("Expected at least 100 training records with unique IDs")
    return records[:100]


def parse_orientation(result):
    """Use the model's label, checking its ID rather than guessing a class mapping."""
    if len(result["label_names"]) != 1 or len(result["scores"]) != 1:
        raise ValueError("Expected one top-1 orientation prediction")
    label = str(result["label_names"][0])
    orientations = {"0_degree": 0, "180_degree": 180}
    if label not in orientations:
        raise ValueError(f"Unexpected orientation label: {label}")
    orientation = orientations[label]
    class_id = int(result["class_ids"][0])
    if class_id != (0 if orientation == 0 else 1):
        raise ValueError("Model label and class ID disagree")
    score = float(result["scores"][0])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid confidence")
    return orientation, score


def correct_crop(image, orientation):
    """A transpose by ROTATE_180 is exactly two axis reversals, without resampling."""
    if orientation == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if orientation == 0:
        return image.copy()
    raise ValueError("Only 0/180-degree corrections are supported")


def ground_truth_crop(data, record):
    image_path = data / "images/train" / record["image_file"]
    label_path = data / "labels/train" / (record["id"] + ".txt")
    if sha256_file(image_path) != record["image_sha256"]:
        raise ValueError("Source image checksum mismatch")
    if sha256_file(label_path) != record["label_sha256"]:
        raise ValueError("GT label checksum mismatch")
    tokens = label_path.read_text().split()
    if len(tokens) != 9 or tokens[0] != "0":
        raise ValueError("Expected one four-corner GT register label")
    corners = np.asarray(list(map(float, tokens[1:])), dtype=np.float64).reshape(4, 2)
    if not np.isfinite(corners).all() or (corners < 0).any() or (corners > 1).any():
        raise ValueError("Invalid normalized GT coordinates")
    with Image.open(image_path) as image:
        if image.size != (record["width"], record["height"]):
            raise ValueError("Image dimensions changed")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize EXIF orientation before using GT labels")
        pixels, geometry = aligned_crop(
            np.asarray(image.convert("RGB")),
            corners * image.size,
            config.CONFIG["crop_margin_fraction"],
        )
    return Image.fromarray(pixels), geometry


def render_sheets(output, rows):
    directory = output / "comparison"
    directory.mkdir()
    font = ImageFont.load_default(size=17)
    sheets = []
    for start in range(0, len(rows), 10):
        sheet = Image.new("RGB", (1440, 1560), "#f2f4f6")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            "GT OBB crop (left) | PaddleOCR correction (right) | Confidence is not calibrated accuracy",
            fill="black",
            font=font,
        )
        for index, row in enumerate(rows[start : start + 10]):
            y = 50 + index * 150
            display_score = (
                f"{row['confidence']:.4f}" if row["confidence"] != "" else "n/a"
            )
            description = (
                f"{row['example_id']}  |  predicted: {row['predicted_orientation']}°"
                f"  |  confidence: {display_score}  |  execution: {row['status']}"
            )
            draw.text((12, y), description, fill="black", font=font)
            if row["error"]:
                draw.text((12, y + 25), row["error"][:140], fill="#9d1111", font=font)
            for column, key in enumerate(["original_crop_path", "corrected_crop_path"]):
                if not row[key]:
                    continue
                with Image.open(output / row[key]) as image:
                    # Preview scaling only; original/corrected PNGs are unresized.
                    image = image.convert("RGB")
                    ratio = min(690 / image.width, 108 / image.height)
                    image = image.resize(
                        (
                            max(1, round(image.width * ratio)),
                            max(1, round(image.height * ratio)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    sheet.paste(image, (12 + column * 720, y + 35))
        path = directory / f"comparison_{start // 10 + 1:02d}.jpg"
        sheet.save(path, quality=95)
        sheets.append(str(path.relative_to(output)))
    return sheets


def run(root, output=None, crop_source="ground-truth"):
    """Process exactly the first 100 training identities; per-sample failures continue."""
    if crop_source != "ground-truth":
        raise ValueError(
            "This pilot explicitly supports GT labels only; no source fallback"
        )
    data = config.dataset_dir(root)
    manifest_path = config.metadata_dir(root) / "split_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    records = select_training_records(manifest)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = config.REPO / "outputs/pilots/paddle_orientation" / stamp
    output = Path(output).resolve()
    if output.is_relative_to(data.resolve()):
        raise ValueError("Pilot output cannot be inside the dataset")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ["original", "corrected"]:
        (output / folder).mkdir()
    input_hashes = {}
    for record in records:
        for path in [
            data / "images/train" / record["image_file"],
            data / "labels/train" / (record["id"] + ".txt"),
        ]:
            input_hashes[str(path)] = sha256_file(path) if path.is_file() else None
    write_json(
        output / "selection.json",
        {
            "crop_source": crop_source,
            "ordering": "Unicode lexicographic ascending example ID",
            "split": "train",
            "count": 100,
            "manifest_sha256": manifest_hash,
            "records": [
                {
                    key: r[key]
                    for key in ["id", "image_file", "image_sha256", "label_sha256"]
                }
                for r in records
            ],
        },
    )
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(output / "model_cache")
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    from paddleocr import TextLineOrientationClassification

    classifier = TextLineOrientationClassification(
        model_name=MODEL_NAME, device="cpu", enable_mkldnn=False, cpu_threads=2
    )
    rows, raw = [], []
    csv_path = output / "predictions.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            row = dict.fromkeys(FIELDS, "")
            row.update(
                example_id=record["id"],
                crop_source=crop_source,
                source_image_path=str(
                    (data / "images/train" / record["image_file"]).resolve()
                ),
                status="failed",
            )
            try:
                original, geometry = ground_truth_crop(data, record)
                original_path = output / "original" / (record["id"] + ".png")
                original.save(original_path)
                row["original_crop_path"] = str(original_path.relative_to(output))
                results = list(classifier.predict(str(original_path), batch_size=1))
                if len(results) != 1:
                    raise ValueError("Classifier did not return exactly one result")
                orientation, score = parse_orientation(results[0])
                row.update(predicted_orientation=orientation, confidence=score)
                corrected = correct_crop(original, orientation)
                corrected_path = output / "corrected" / (record["id"] + ".png")
                corrected.save(corrected_path)
                expected = np.asarray(original)
                if orientation == 180:
                    expected = expected[::-1, ::-1]
                with Image.open(corrected_path) as saved:
                    if not np.array_equal(np.asarray(saved), expected):
                        raise ValueError(
                            "Saved correction is not the exact expected pixel transform"
                        )
                raw.append(
                    {
                        "example_id": record["id"],
                        "result": results[0].json,
                        "crop_geometry": geometry,
                    }
                )
                row.update(
                    corrected_crop_path=str(corrected_path.relative_to(output)),
                    status="success",
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            print(f"PILOT {len(rows)}/100 {record['id']} {row['status']}", flush=True)
    write_json(output / "raw_predictions.json", raw)
    sheets = render_sheets(output, rows)
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("Dataset manifest changed during pilot")
    for filename, digest in input_hashes.items():
        path = Path(filename)
        after = sha256_file(path) if path.is_file() else None
        if after != digest:
            raise RuntimeError(f"Input changed during pilot: {filename}")
    successful = sum(row["status"] == "success" for row in rows)
    summary = {
        "model": MODEL_NAME,
        "crop_source": crop_source,
        "device": "cpu",
        "selected": 100,
        "successful": successful,
        "failed": 100 - successful,
        "predicted_180": sum(
            row["status"] == "success" and row["predicted_orientation"] == 180
            for row in rows
        ),
        "contact_sheets": sheets,
        "manifest_sha256": manifest_hash,
        "manual_orientation_ground_truth": "not available; accuracy not measured",
        "confidence_interpretation": "uncalibrated model score; not probability of correctness",
        "crop_margin_fraction": config.CONFIG["crop_margin_fraction"],
        "input_image_and_label_hashes_unchanged": True,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "paddleocr",
                "paddlex",
                "paddlepaddle",
                "numpy",
                "Pillow",
                "opencv-python",
            ]
        },
        "pilot_source_sha256": sha256_file(__file__),
        "geometry_source_sha256": sha256_file(Path(__file__).with_name("geometry.py")),
    }
    write_json(output / "summary.json", summary)
    (output / "environment-lock.txt").write_text(
        "\n".join(
            sorted(
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
    write_json(
        output / "model_checksums.json",
        {
            str(path.relative_to(output)): sha256_file(path)
            for path in (output / "model_cache").rglob("*")
            if path.is_file()
        },
    )
    from .orientation_review import write_review_interface

    write_review_interface(output)
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return summary
