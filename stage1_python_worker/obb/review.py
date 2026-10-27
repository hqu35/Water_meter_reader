"""Render mask/OBB/crop evidence and enforce a separate explicit human gate."""

import random
import time

import numpy as np
from PIL import Image, ImageDraw

from ..utils.io import read_json, sha256_file, write_json
from . import config
from .geometry import aligned_crop


def render_review(root, records):
    data = config.dataset_dir(root)
    rng = random.Random(config.CONFIG["seed"])
    selected = []
    # Include the largest padding cases as well as deterministic random examples.
    for split, count in [("train", 16), ("val", 8), ("test", 8)]:
        pool = [r for r in records if r["split"] == split]
        extremes = sorted(pool, key=lambda r: sum(r["padding_ltrb"]), reverse=True)[:2]
        others = [r for r in pool if r not in extremes]
        selected.extend(
            extremes + rng.sample(others, min(count - len(extremes), len(others)))
        )
    directory = config.annotation_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    sheets = []
    for start in range(0, len(selected), 8):
        sheet = Image.new("RGB", (1440, 8 * 270 + 48), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            "OBB REVIEW: original | foreground mask + OBB | aligned crop (180-degree polarity unresolved)",
            fill="black",
        )
        for row, record in enumerate(selected[start : start + 8]):
            image = Image.open(
                data / "images" / record["split"] / record["image_file"]
            ).convert("RGB")
            mask = (
                np.asarray(
                    Image.open(
                        data / "masks" / record["split"] / (record["id"] + ".png")
                    )
                )
                != 0
            )
            pixels = np.asarray(image).copy()
            pixels[mask] = (
                pixels[mask] * 0.55 + np.array([40, 240, 70]) * 0.45
            ).astype(np.uint8)
            overlay = Image.fromarray(pixels)
            polygon = [tuple(p) for p in record["corners_xy"]]
            painter = ImageDraw.Draw(overlay)
            painter.line(polygon + [polygon[0]], fill="red", width=3)
            for index, point in enumerate(polygon):
                painter.text(point, str(index), fill="yellow")
            crop, _ = aligned_crop(
                np.asarray(image),
                record["corners_xy"],
                config.CONFIG["crop_margin_fraction"],
            )
            panels = [image, overlay, Image.fromarray(crop)]
            y = 48 + row * 270
            draw.text(
                (12, y),
                f"{record['split']} / {record['id']}  padding LTRB={record['padding_ltrb']}",
                fill="black",
            )
            for column, panel in enumerate(panels):
                panel.thumbnail((460, 235))
                sheet.paste(panel, (column * 480 + 10, y + 22))
        name = f"obb_review_{start // 8 + 1:02d}.jpg"
        path = directory / name
        sheet.save(path, quality=95)
        sheets.append({"file": name, "sha256": sha256_file(path)})
    return sheets, [r["id"] for r in selected]


def review_evidence(root):
    from .data import validate_dataset

    metadata = config.metadata_dir(root)
    manifest = validate_dataset(root)
    validation = read_json(metadata / "validation.json")
    assert validation["task"] == "obb" and validation["passed"]
    assert validation["reviewed_image_count"] >= 30
    ids = validation["reviewed_ids"]
    assert len(ids) == len(set(ids)) == validation["reviewed_image_count"]
    assert set(ids).issubset({r["id"] for r in manifest["records"]})
    for sheet in validation["review_sheets"]:
        assert (
            sha256_file(config.annotation_dir(root) / sheet["file"]) == sheet["sha256"]
        ), "Review sheet changed"
    return {
        "task": "obb",
        "manifest_sha256": sha256_file(metadata / "split_manifest.json"),
        "validation_sha256": sha256_file(metadata / "validation.json"),
        "config_sha256": sha256_file(config.CONFIG_PATH),
        "reviewed_image_count": len(ids),
        "reviewed_sheets": validation["review_sheets"],
    }


def approve(root, reviewer, confirm_reviewed=False):
    if not reviewer.strip() or not confirm_reviewed:
        raise ValueError("Explicit human reviewer and --confirm-reviewed are required")
    evidence = review_evidence(root)
    path = config.metadata_dir(root) / "annotation_review.json"
    if path.exists():
        raise FileExistsError("Approval exists; refusing to overwrite review history")
    write_json(
        path,
        {
            **evidence,
            "approved": True,
            "reviewer": reviewer,
            "reviewer_type": "human",
            "attestation": "I visually reviewed all OBB sheets, masks and aligned crops.",
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def require_approval(root):
    path = config.metadata_dir(root) / "annotation_review.json"
    if not path.exists():
        raise RuntimeError(
            "OBB training blocked: explicit human visual approval is pending"
        )
    approval = read_json(path)
    assert approval.get("approved") is True and approval.get("reviewer_type") == "human"
    assert approval.get("reviewer", "").strip() and approval.get("attestation")
    for key, value in review_evidence(root).items():
        assert approval.get(key) == value, f"OBB approval stale or invalid: {key}"
