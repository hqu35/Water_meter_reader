"""Convert selected real source image/mask pairs into oriented rectangles."""

import io
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ..utils.io import read_json, sha256_file, write_json
from . import config
from .geometry import normalized_corners, rectangle_from_mask, required_padding

SPLITS = ("train", "val", "test")
# Ultralytics scans image directories recursively rather than using our manifest.
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".heif",
    ".jp2",
}


def check_splits(records):
    expected = {split: config.CONFIG[f"{split}_count"] for split in SPLITS}
    if Counter(r["split"] for r in records) != expected:
        raise ValueError(f"Expected frozen split sizes: {expected}")
    for key in ("id", "source_image", "source_mask"):
        if len({r[key] for r in records}) != len(records):
            raise ValueError(f"Repeated {key} in dataset splits")
    for record in records:
        identity = record["id"]
        if not identity or Path(identity).name != identity or identity in (".", ".."):
            raise ValueError(f"Invalid image ID: {identity}")
        official = "test" if record["split"] == "test" else "train"
        if record["official_split"] != official:
            raise ValueError(f"Incorrect official split: {identity}")


def validate_dataset(root, splits=SPLITS):
    """Check the manifest and only the image/label pairs consumed by this command."""
    data = config.dataset_dir(root)
    manifest = read_json(config.metadata_dir(root) / "split_manifest.json")
    if manifest["task"] != "obb":
        raise ValueError("Expected an OBB split manifest")
    records = manifest["records"]
    check_splits(records)
    for split in splits:
        split_records = [r for r in records if r["split"] == split]
        folder = data / "images" / split
        actual = {
            p.relative_to(folder).as_posix()
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        expected_images = {r["image_file"] for r in split_records}
        if actual != expected_images:
            missing = sorted(expected_images - actual)[:5]
            extra = sorted(actual - expected_images)[:5]
            raise ValueError(
                f"Image inventory mismatch in {folder}: missing={missing}; extra={extra}"
            )

        for record in split_records:
            identity = record["id"]
            image_file = Path(record["image_file"])
            if image_file.name != str(image_file) or image_file.stem != identity:
                raise ValueError(f"Image filename does not match ID: {identity}")
            size = (record["width"], record["height"])
            with Image.open(folder / image_file) as image:
                image.load()
                if image.size != size:
                    raise ValueError(f"Incorrect image dimensions: {image_file}")

            label = data / "labels" / split / (identity + ".txt")
            rows = label.read_text().strip().splitlines()
            tokens = rows[0].split() if len(rows) == 1 else []
            if len(tokens) != 9 or tokens[0] != "0":
                raise ValueError(f"Expected one class-0 OBB in {label}")
            corners = np.array([float(token) for token in tokens[1:]]).reshape(4, 2)
            if (
                not np.isfinite(corners).all()
                or (corners < 0).any()
                or (corners > 1).any()
            ):
                raise ValueError(f"Invalid normalized coordinates in {label}")
            edges = np.roll(corners, -1, axis=0) - corners
            turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
                edges[:, 0], -1
            )
            if not (turns > 0).all():
                raise ValueError(f"Invalid OBB corner order or area in {label}")
            # Held-out polygon diagnostics use these stored coordinates.
            expected = np.asarray(record["corners_xy"])
            if expected.shape != (4, 2) or not np.allclose(
                corners * size, expected, atol=1e-5
            ):
                raise ValueError(f"Label does not match split manifest: {label}")
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
    if (original["dataset_sha256"], original["seed"]) != (
        config.CONFIG["dataset_sha256"],
        config.CONFIG["seed"],
    ):
        raise ValueError("Selection does not match the configured archive and seed")
    check_splits(original["records"])
    # Reuse the frozen inventory's decoded identities for split deduplication.
    source_pixels = [r["pixel_sha256"] for r in original["records"]]
    if len(source_pixels) != len(set(source_pixels)):
        raise ValueError("Duplicate source pixels in the frozen selection")
    if sha256_file(archive_path) != original["dataset_sha256"]:
        raise ValueError("Source ZIP checksum mismatch")
    records = []
    with ZipFile(archive_path) as archive:
        for old in original["records"]:
            identity, split = old["id"], old["split"]
            official = old["official_split"]
            source = PurePosixPath(old["source_image"])
            mask_directory = {
                "det_img": "seg_label",
                f"{official}_img": f"{official}_seg_label",
            }
            folder = mask_directory.get(source.parent.name.lower())
            if (
                not folder
                or official not in source.parts[:-2]
                or old["source_mask"]
                != str(source.parent.parent / folder / source.name)
                or identity != f"{official}_{source.stem}"
            ):
                raise ValueError(f"Incorrect source image/mask pair: {identity}")
            image_bytes = archive.read(old["source_image"])
            mask_bytes = archive.read(old["source_mask"])
            with (
                Image.open(io.BytesIO(image_bytes)) as image,
                Image.open(io.BytesIO(mask_bytes)) as mask_image,
            ):
                if image.size != mask_image.size or image.size != (
                    old["width"],
                    old["height"],
                ):
                    raise ValueError(f"Image/mask dimensions do not match: {identity}")
                pixels = np.asarray(image.convert("RGB"))
                mask = np.asarray(mask_image)
                if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 255}):
                    raise ValueError(f"Expected a binary foreground mask: {identity}")
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
                    "source_pixel_sha256": old["pixel_sha256"],
                    "source_size": [old["width"], old["height"]],
                    "image_file": image_file,
                    "width": width,
                    "height": height,
                    "padding_ltrb": padding,
                    "corners_xy": corners.tolist(),
                }
                records.append(record)
    manifest = {
        "task": "obb",
        "conversion_version": config.CONFIG["conversion_version"],
        "seed": original["seed"],
        "dataset_sha256": original["dataset_sha256"],
        "records": records,
    }
    metadata = config.metadata_dir(root)
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection_path, metadata / "source_selection.json")
    write_json(metadata / "split_manifest.json", manifest)
    config.write_yaml(root)
    print(f"Prepared {len(records)} image/label pairs in {destination}")
