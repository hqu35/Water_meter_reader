"""Convert selected real source image/mask pairs into oriented rectangles."""

import hashlib
import io
from zipfile import ZipFile
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.validation import verify_directory_membership
from ..utils.io import read_json, sha256_file, write_json
from . import config
from .geometry import (
    mask_is_covered,
    normalized_corners,
    ordered_corners,
    rectangle_from_mask,
    required_padding,
)


def pixel_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def validate_record(data, record):
    """Verify bytes, alignment, label round-trip, winding and complete mask coverage."""
    split, identity = record["split"], record["id"]
    image = data / "images" / split / record["image_file"]
    mask_path = data / "masks" / split / (identity + ".png")
    label = data / "labels" / split / (identity + ".txt")
    for path, key in [
        (image, "image_sha256"),
        (mask_path, "mask_sha256"),
        (label, "label_sha256"),
    ]:
        assert sha256_file(path) == record[key], f"Changed file: {path}"
    assert pixel_hash(image) == record["pixel_sha256"], "Decoded pixels changed"
    with Image.open(image) as photo, Image.open(mask_path) as mask_image:
        assert photo.size == mask_image.size == (record["width"], record["height"])
        mask = np.asarray(mask_image)
    assert set(np.unique(mask)).issubset({0, 255}), "Unexpected mask values"
    tokens = label.read_text().split()
    assert len(tokens) == 9 and tokens[0] == "0", "Expected one YOLO OBB register"
    normalized = np.array(list(map(float, tokens[1:]))).reshape(4, 2)
    assert (
        np.isfinite(normalized).all()
        and (normalized >= 0).all()
        and (normalized <= 1).all()
    )
    corners = normalized * [record["width"], record["height"]]
    assert np.allclose(corners, ordered_corners(corners), atol=1e-4), (
        "Noncanonical corner order"
    )
    assert np.allclose(corners, record["corners_xy"], atol=1e-5), (
        "Label round-trip failed"
    )
    assert mask_is_covered(mask, corners), "Foreground extends outside rectangle"
    recomputed = rectangle_from_mask(mask)
    # OpenCV may choose another equivalent start/axis for exact square ties.
    from .geometry import rotated_iou

    assert rotated_iou(corners, recomputed) > 0.9999, (
        "Not the mask minimum-area rectangle"
    )
    assert record["official_split"] == ("test" if split == "test" else "train")


def validate_dataset(root):
    data = config.dataset_dir(root)
    manifest = read_json(config.metadata_dir(root) / "split_manifest.json")
    assert (
        manifest["task"] == "obb"
        and manifest["conversion_version"] == config.CONFIG["conversion_version"]
    )
    records = manifest["records"]
    selection_path = config.metadata_dir(root) / "source_selection.json"
    assert sha256_file(selection_path) == manifest["source_selection_sha256"]
    selected = read_json(selection_path)
    assert [(r["id"], r["split"]) for r in selected["records"]] == [
        (r["id"], r["split"]) for r in records
    ], "Selected identities changed"
    for source, record in zip(selected["records"], records):
        for key in ["source_image", "source_mask", "official_split"]:
            assert source[key] == record[key], "Source pairing changed"
        for key in ["image_sha256", "mask_sha256", "pixel_sha256"]:
            assert source[key] == record["source_" + key], "Source identity changed"
    assert manifest["dataset_sha256"] == config.CONFIG["dataset_sha256"]
    assert manifest["seed"] == config.CONFIG["seed"]
    counts = Counter(record["split"] for record in records)
    assert dict(counts) == {
        split: config.CONFIG[split + "_count"] for split in ["train", "val", "test"]
    }, "Unexpected selection size"
    ids = [(r["split"], r["id"]) for r in records]
    assert len(ids) == len(set(ids))
    source_pixels = [r["source_pixel_sha256"] for r in records]
    assert len(source_pixels) == len(set(source_pixels)), "Source image leakage"
    assert (
        manifest["selected_identities_sha256"]
        == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    )
    for record in records:
        validate_record(data, record)
    verify_directory_membership(data, records)
    return manifest


def prepare(root, archive_path=None, selection_path=None):
    """Rebuild from the original source ZIP and fixed, format-neutral identities."""
    archive_path = (
        Path(archive_path) if archive_path else config.source_archive_path(root)
    )
    selection_path = (
        Path(selection_path)
        if selection_path
        else config.source_dir(root) / "selection.json"
    )
    destination = config.dataset_dir(root)
    if any(
        (destination / name).exists()
        for name in ["images", "labels", "masks", "metadata/split_manifest.json"]
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing OBB dataset: {destination}"
        )
    original = read_json(selection_path)
    assert original["dataset_sha256"] == config.CONFIG["dataset_sha256"]
    assert original["seed"] == config.CONFIG["seed"]
    assert sha256_file(archive_path) == original["dataset_sha256"], (
        "Source ZIP checksum mismatch"
    )
    records = []
    with ZipFile(archive_path) as archive:
        for old in original["records"]:
            identity, split = old["id"], old["split"]
            image_bytes = archive.read(old["source_image"])
            mask_bytes = archive.read(old["source_mask"])
            assert hashlib.sha256(image_bytes).hexdigest() == old["image_sha256"]
            assert hashlib.sha256(mask_bytes).hexdigest() == old["mask_sha256"]
            with (
                Image.open(io.BytesIO(image_bytes)) as image,
                Image.open(io.BytesIO(mask_bytes)) as mask_image,
            ):
                assert image.size == mask_image.size == (old["width"], old["height"])
                pixels = np.asarray(image.convert("RGB"))
                assert (
                    hashlib.sha256(pixels.tobytes()).hexdigest() == old["pixel_sha256"]
                )
                mask = np.asarray(mask_image)
                assert set(np.unique(mask)).issubset({0, 255})
                corners = rectangle_from_mask(mask)
                padding = required_padding(corners, image.width, image.height)
                left, top, right, bottom = padding
                extension = {"JPEG": ".jpg", "PNG": ".png"}[image.format]
                image_file = identity + (".png" if any(padding) else extension)
                image_path = destination / "images" / split / image_file
                mask_path = destination / "masks" / split / (identity + ".png")
                label_path = destination / "labels" / split / (identity + ".txt")
                for path in [image_path, mask_path, label_path]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                if any(padding):
                    pixels = np.pad(
                        pixels, ((top, bottom), (left, right), (0, 0)), mode="edge"
                    )
                    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant")
                    Image.fromarray(pixels).save(image_path)
                    Image.fromarray(mask).save(mask_path)
                    corners = corners + [left, top]
                else:
                    image_path.write_bytes(image_bytes)
                    mask_path.write_bytes(mask_bytes)
                width, height = (
                    old["width"] + left + right,
                    old["height"] + top + bottom,
                )
                normalized = normalized_corners(corners, width, height)
                label_path.write_text(
                    "0 " + " ".join(f"{x:.10f}" for x in normalized.ravel()) + "\n"
                )
                record = {
                    "id": identity,
                    "split": split,
                    "official_split": old["official_split"],
                    "source_image": old["source_image"],
                    "source_mask": old["source_mask"],
                    "source_image_sha256": old["image_sha256"],
                    "source_mask_sha256": old["mask_sha256"],
                    "source_pixel_sha256": old["pixel_sha256"],
                    "source_size": [old["width"], old["height"]],
                    "image_file": image_file,
                    "width": width,
                    "height": height,
                    "padding_ltrb": padding,
                    "corners_xy": corners.tolist(),
                    "image_sha256": sha256_file(image_path),
                    "mask_sha256": sha256_file(mask_path),
                    "label_sha256": sha256_file(label_path),
                    "pixel_sha256": pixel_hash(image_path),
                }
                validate_record(destination, record)
                records.append(record)
    ids = [(r["split"], r["id"]) for r in records]
    manifest = {
        "task": "obb",
        "conversion_version": config.CONFIG["conversion_version"],
        "seed": original["seed"],
        "dataset_sha256": original["dataset_sha256"],
        "source_selection_sha256": sha256_file(selection_path),
        "selected_identities_sha256": hashlib.sha256(
            json.dumps(ids).encode()
        ).hexdigest(),
        "exclusions": original["exclusions"],
        "records": records,
    }
    metadata = config.metadata_dir(root)
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection_path, metadata / "source_selection.json")
    write_json(metadata / "split_manifest.json", manifest)
    config.write_yaml(root)
    validate_dataset(root)
    from .review import render_review

    sheets, reviewed_ids = render_review(root, records)
    report = {
        "task": "obb",
        "passed": True,
        "counts": dict(Counter(r["split"] for r in records)),
        "same_selected_identities_and_official_splits": True,
        "source_selection_sha256": manifest["source_selection_sha256"],
        "padded_images": sum(any(r["padding_ltrb"]) for r in records),
        "reviewed_ids": reviewed_ids,
        "reviewed_image_count": len(reviewed_ids),
        "review_sheets": sheets,
        "human_approval": "PENDING",
        "opencv_version": __import__("cv2").__version__,
    }
    write_json(metadata / "validation.json", report)
    print(json.dumps(report, indent=2))
