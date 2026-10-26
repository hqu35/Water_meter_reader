"""Deterministic real-image subsets and mask-to-YOLO annotation conversion."""

import random
from collections import Counter
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from .. import config
from ..utils.io import sha256_file, write_json
from .archive import decode_image_mask, discover_image_mask_pairs
from .download import verify_source_archive
from .review import generate_review_sheets
from .validation import validate_converted_box


def bbox_from_mask(foreground) -> Optional[list]:
    """Enclose all foreground pixels, retaining disconnected regions and edges."""
    import numpy as np

    y_indices, x_indices = np.nonzero(foreground)
    if len(x_indices) == 0:
        return None
    # Half-open bounds preserve both a one-pixel object and the far image edge.
    return [
        int(x_indices.min()),
        int(y_indices.min()),
        int(x_indices.max()) + 1,
        int(y_indices.max()) + 1,
    ]


def normalize_box(box: list, width: int, height: int) -> list:
    """Convert half-open pixel bounds to normalized YOLO center/width/height."""
    left, top, right, bottom = box
    assert 0 <= left < right <= width and 0 <= top < bottom <= height
    return [
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    ]


def export_example(
    archive: ZipFile,
    target: Path,
    *,
    image_name: str,
    mask_name: str,
    official_split: str,
    split: str,
    image,
    foreground,
    hashes: dict,
    box: list,
) -> dict:
    """Write original image/mask bytes and a checked class-0 YOLO label."""
    example_id = f"{official_split}_{Path(image_name).stem}"
    extension = ".jpg" if hashes["image_format"] == "JPEG" else ".png"
    image_file = example_id + extension
    image_path = target / "images" / split / image_file
    label_path = target / "labels" / split / (example_id + ".txt")
    mask_path = target / "masks" / split / (example_id + ".png")
    for path in [image_path, label_path, mask_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # The source's .png photographs can contain JPEG bytes: never recompress them.
    image_path.write_bytes(archive.read(image_name))
    mask_path.write_bytes(archive.read(mask_name))
    normalized_box = normalize_box(box, image.width, image.height)
    label = "0 " + " ".join(f"{value:.10f}" for value in normalized_box) + "\n"
    label_path.write_text(label)
    validate_converted_box(box, label, image.size, foreground)

    return {
        "id": example_id,
        "image_file": image_file,
        "split": split,
        "official_split": official_split,
        "source_image": image_name,
        "source_mask": mask_name,
        "width": image.width,
        "height": image.height,
        "box_xyxy": box,
        "yolo": normalized_box,
        "foreground_pixels": int(foreground.sum()),
        "foreground_area_ratio": float(foreground.mean()),
        "label_sha256": sha256_file(label_path),
        **hashes,
    }


def create_dataset_splits(archive: ZipFile, target: Path, rng: random.Random) -> tuple:
    """Select official test first, then disjoint validation/train examples."""
    settings = config.CONFIG
    pools = discover_image_mask_pairs(archive)
    records = []
    seen_pixels = {}
    exclusions = []
    split_sizes = [
        ("test", settings["test_count"]),
        ("train", settings["train_count"] + settings["val_count"]),
    ]
    for official_split, required_count in split_sizes:
        candidates = list(pools[official_split])
        rng.shuffle(candidates)
        accepted = 0
        for image_name, mask_name in candidates:
            image, foreground, hashes = decode_image_mask(
                archive, image_name, mask_name
            )
            box = bbox_from_mask(foreground)
            if box is None:
                exclusions.append(
                    {"image": image_name, "reason": "empty_foreground_mask"}
                )
                continue
            pixel_hash = hashes["pixel_sha256"]
            if pixel_hash in seen_pixels:
                exclusions.append(
                    {
                        "image": image_name,
                        "reason": "duplicate_decoded_image",
                        "duplicate_of": seen_pixels[pixel_hash],
                    }
                )
                continue

            seen_pixels[pixel_hash] = image_name
            if official_split == "test":
                split = "test"
            else:
                split = "val" if accepted < settings["val_count"] else "train"
            records.append(
                export_example(
                    archive,
                    target,
                    image_name=image_name,
                    mask_name=mask_name,
                    official_split=official_split,
                    split=split,
                    image=image,
                    foreground=foreground,
                    hashes=hashes,
                    box=box,
                )
            )
            accepted += 1
            if accepted == required_count:
                break
        assert accepted == required_count, (official_split, accepted, required_count)
    return records, exclusions


def prepare_dataset(run_root: Path) -> None:
    """Verify source, export deterministic splits, then save reviewable annotations."""
    archive_path = verify_source_archive(run_root)
    target = config.dataset_dir(run_root)
    metadata_root = config.metadata_dir(run_root)
    rng = random.Random(config.CONFIG["seed"])
    if (target / "images").exists():
        raise RuntimeError(
            "Use a fresh dataset directory; refuse to mix stale images into a new subset."
        )
    with ZipFile(archive_path) as archive:
        records, exclusions = create_dataset_splits(archive, target, rng)

    manifest_path = metadata_root / "split_manifest.json"
    write_json(
        manifest_path,
        {
            "seed": config.CONFIG["seed"],
            "dataset_sha256": config.CONFIG["dataset_sha256"],
            "records": records,
            "exclusions": exclusions,
        },
    )
    config.write_dataset_yaml(run_root)
    # Keep the SAME RNG after both shuffles: review IDs are part of the audit.
    sheets = generate_review_sheets(archive_path, records, rng, metadata_root)
    write_json(
        metadata_root / "validation.json",
        {
            "passed": True,
            "conversion": "union of all nonzero mask pixels; max+1 exclusive bounds; class 0 register",
            "mask_values": sorted(
                {value for record in records for value in record["mask_values"]}
            ),
            "counts": dict(Counter(record["split"] for record in records)),
            "manifest_sha256": sha256_file(manifest_path),
            "exact_decoded_image_deduplication": True,
            "all_selected_dimensions_aligned": True,
            "all_boxes_cover_foreground": True,
            "yolo_roundtrip_tolerance_pixels": 1e-5,
            "review_image_count": sum(len(sheet["ids"]) for sheet in sheets),
            "sheets": sheets,
            "excluded_count": len(exclusions),
        },
    )
    print(
        "PREPARATION_COMPLETE: 32 review images saved; training is still locked pending visual review."
    )
