"""Annotation contact sheets, explicit approval and the training review gate."""

import random
import time
from pathlib import Path
from zipfile import ZipFile

from .. import config
from ..utils.io import read_json, sha256_file, write_json
from .archive import decode_image_mask
from .validation import verify_dataset_integrity


def select_review_records(records: list, rng: random.Random) -> list:
    """Select the same 32 examples, including mask-area extremes in each split."""
    selected = []
    for split, count in [("train", 16), ("val", 8), ("test", 8)]:
        split_records = [record for record in records if record["split"] == split]
        by_area = sorted(
            split_records, key=lambda record: record["foreground_area_ratio"]
        )
        extreme_ids = {by_area[0]["id"], by_area[-1]["id"]}
        others = [record for record in split_records if record["id"] not in extreme_ids]
        selected.extend([by_area[0], by_area[-1]] + rng.sample(others, count - 2))
    return selected


def generate_review_sheets(
    archive_path: Path, records: list, rng: random.Random, metadata_root: Path
) -> list:
    """Render four original/mask/box sheets with the original layout and encoding."""
    import numpy as np
    from PIL import Image, ImageDraw

    selected = select_review_records(records, rng)
    destination = metadata_root / "annotations"
    destination.mkdir(parents=True, exist_ok=True)
    sheets = []
    with ZipFile(archive_path) as archive:
        for sheet_index in range(4):
            sheet = Image.new("RGB", (1536, 1680), "#f5f5f5")
            draw = ImageDraw.Draw(sheet)
            group = selected[sheet_index * 8 : (sheet_index + 1) * 8]
            for index, record in enumerate(group):
                image, foreground, _ = decode_image_mask(
                    archive, record["source_image"], record["source_mask"]
                )
                overlay_pixels = np.array(image).copy()
                overlay_pixels[foreground] = (
                    overlay_pixels[foreground] * 0.55 + np.array([50, 240, 90]) * 0.45
                ).astype("uint8")
                overlay = Image.fromarray(overlay_pixels)
                overlay_draw = ImageDraw.Draw(overlay)
                left, top, right, bottom = record["box_xyxy"]
                overlay_draw.rectangle(
                    (left, top, right - 1, bottom - 1), outline=(255, 40, 40), width=3
                )
                x = (index % 2) * 768
                y = (index // 2) * 420
                sheet.paste(image.resize((384, 384)), (x, y + 32))
                sheet.paste(overlay.resize((384, 384)), (x + 384, y + 32))
                title = (
                    record["split"]
                    + " / "
                    + record["id"]
                    + "     ORIGINAL | MASK + YOLO BOX"
                )
                draw.text((x + 7, y + 9), title, fill="#111111")

            path = destination / f"review_{sheet_index + 1:02d}.jpg"
            sheet.save(path, quality=94)
            sheets.append(
                {
                    "path": str(path.relative_to(metadata_root)),
                    "sha256": sha256_file(path),
                    "ids": [record["id"] for record in group],
                }
            )
    return sheets


def approve_annotations(run_root: Path, reviewer: str) -> None:
    """Record explicit visual approval only after verifying conversion outputs."""
    metadata_root = config.metadata_dir(run_root)
    report = read_json(metadata_root / "validation.json")
    assert report["passed"]
    assert report["review_image_count"] >= 30
    assert (
        sha256_file(metadata_root / "split_manifest.json") == report["manifest_sha256"]
    )
    for sheet in report["sheets"]:
        assert sha256_file(metadata_root / sheet["path"]) == sheet["sha256"]

    write_json(
        metadata_root / "annotation_review.json",
        {
            "approved": True,
            "reviewer": reviewer,
            "reviewed_image_count": report["review_image_count"],
            "manifest_sha256": report["manifest_sha256"],
            "reviewed_sheets": report["sheets"],
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def require_annotation_review(run_root: Path) -> dict:
    """Block training when approval, reviewed bytes or split membership changed."""
    metadata_root = config.metadata_dir(run_root)
    review = read_json(metadata_root / "annotation_review.json")
    manifest_path = metadata_root / "split_manifest.json"
    assert review["approved"] and review["reviewed_image_count"] >= 30
    assert sha256_file(manifest_path) == review["manifest_sha256"], (
        "Manifest changed after review"
    )
    for sheet in review["reviewed_sheets"]:
        assert sha256_file(metadata_root / sheet["path"]) == sheet["sha256"]
    manifest = read_json(manifest_path)
    verify_dataset_integrity(run_root, manifest["records"])
    return review
