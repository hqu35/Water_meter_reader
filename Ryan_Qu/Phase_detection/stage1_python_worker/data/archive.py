"""Read image/mask pairs inside the ZIP without executing dataset code."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from ..obb import config
from ..utils.io import sha256_file, write_json
from .download import verify_source_archive


def discover_image_mask_pairs(archive: ZipFile) -> dict:
    """Match exact parent/split/filename, never positional glob ordering."""
    member_names = set(archive.namelist())
    pairs = {"train": [], "test": []}
    missing_masks = []
    mask_directories = {
        "det_img": "seg_label",
        "train_img": "train_seg_label",
        "test_img": "test_seg_label",
    }
    for name in sorted(member_names):
        path = PurePosixPath(name)
        image_directory = path.parent.name.lower()
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        if image_directory not in mask_directories:
            continue

        parents = path.parts[:-2]
        if any("train" in part.lower() for part in parents):
            split = "train"
        elif any("test" in part.lower() for part in parents):
            split = "test"
        else:
            raise ValueError("Cannot determine official split: " + name)

        mask_directory = mask_directories[image_directory]
        mask_name = str(path.parent.parent / mask_directory / path.name)
        if mask_name not in member_names:
            missing_masks.append({"image": name, "expected_mask": mask_name})
            continue
        pairs[split].append((name, mask_name))

    if missing_masks:
        raise ValueError("Unpaired images: " + str(missing_masks[:10]))
    assert pairs["train"] and pairs["test"], "Detection image directories not found"
    return pairs


def decode_image_mask(archive: ZipFile, image_name: str, mask_name: str) -> tuple:
    """Validate dimensions/encoding and return RGB pixels, foreground and hashes."""
    import numpy as np
    from PIL import Image

    image_bytes = archive.read(image_name)
    mask_bytes = archive.read(mask_name)
    source_image = Image.open(io.BytesIO(image_bytes))
    image_format = source_image.format
    image = source_image.convert("RGB")
    mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
    assert mask.shape[:2] == (image.height, image.width), (
        "Image/mask dimension mismatch",
        image_name,
        mask.shape,
        image.size,
    )
    if mask.ndim == 3:
        assert np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(
            mask[..., 0], mask[..., 2]
        ), ("Unexpected color mask", mask_name)
        mask = mask[..., 0]

    mask_values = np.unique(mask).tolist()
    assert set(mask_values) <= {0, 1, 255}, (
        "Unexpected mask encoding",
        mask_name,
        mask_values,
    )
    hashes = {
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_format": image_format,
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "pixel_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
        "mask_values": mask_values,
    }
    return image, mask > 0, hashes


def inspect_dataset(run_root: Path) -> None:
    """Save actual split counts and a small decoded sample from the source ZIP."""
    from ..obb.geometry import rectangle_from_mask

    archive_path = verify_source_archive(run_root)
    with ZipFile(archive_path) as archive:
        pairs = discover_image_mask_pairs(archive)
        samples = []
        for split, split_pairs in pairs.items():
            for image_name, mask_name in split_pairs[:8]:
                image, foreground, hashes = decode_image_mask(
                    archive, image_name, mask_name
                )
                samples.append(
                    {
                        "split": split,
                        "image": image_name,
                        "mask": mask_name,
                        "size": image.size,
                        "image_format": hashes["image_format"],
                        "mask_values": hashes["mask_values"],
                        "foreground_pixels": int(foreground.sum()),
                        "corners_xy": rectangle_from_mask(foreground).tolist()
                        if foreground.any()
                        else None,
                    }
                )
        report = {
            "archive_sha256": sha256_file(archive_path),
            "archive_members": len(archive.namelist()),
            "official_counts": {split: len(items) for split, items in pairs.items()},
            "pairing": "exact split/parent and filename",
            "sample": samples,
            "top_level": sorted(
                {
                    PurePosixPath(name).parts[0]
                    for name in archive.namelist()
                    if PurePosixPath(name).parts
                }
            ),
        }
    write_json(config.metadata_dir(run_root) / "inspection.json", report)
    print(json.dumps(report, indent=2))
