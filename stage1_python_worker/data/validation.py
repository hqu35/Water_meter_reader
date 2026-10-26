"""Coordinate checks and integrity verification for the reviewed dataset."""

import hashlib
from pathlib import Path

from .. import config
from ..utils.io import sha256_file


def validate_converted_box(
    box: list, yolo_label: str, image_size: tuple, foreground
) -> None:
    """Check the serialized YOLO label recovers the box and covers every pixel."""
    import numpy as np

    width, height = image_size
    center_x, center_y, box_width, box_height = map(float, yolo_label.split()[1:])
    recovered = np.array(
        [
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ]
    )
    assert float(np.abs(recovered - np.array(box)).max()) < 1e-5
    left, top, right, bottom = box
    assert int(foreground[top:bottom, left:right].sum()) == int(foreground.sum())


def verify_dataset_integrity(run_root: Path, records: list) -> None:
    """Verify hashes, cross-split uniqueness and exact directory membership."""
    import numpy as np
    from PIL import Image

    target = config.dataset_dir(run_root)
    pixel_hashes = []
    for record in records:
        split = record["split"]
        image_path = target / "images" / split / record["image_file"]
        label_path = target / "labels" / split / (record["id"] + ".txt")
        mask_path = target / "masks" / split / (record["id"] + ".png")
        assert sha256_file(mask_path) == record["mask_sha256"], (
            "Mask changed after review"
        )
        assert sha256_file(label_path) == record["label_sha256"], (
            "Label changed after review"
        )
        pixels = np.array(Image.open(image_path).convert("RGB"))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        assert digest == record["pixel_sha256"], "Image changed after review"
        pixel_hashes.append(record["pixel_sha256"])

    assert len(pixel_hashes) == len(set(pixel_hashes)), "Image leakage across splits"
    verify_directory_membership(target, records)


def verify_directory_membership(dataset_root: Path, records: list) -> None:
    """Reject extra, missing or misplaced images, labels and masks."""
    for kind, extension in [("images", None), ("labels", ".txt"), ("masks", ".png")]:
        for split in ["train", "val", "test"]:
            split_records = [record for record in records if record["split"] == split]
            expected = {
                record["image_file"] if extension is None else record["id"] + extension
                for record in split_records
            }
            folder = dataset_root / kind / split
            actual = (
                {path.name for path in folder.iterdir() if path.is_file()}
                if folder.exists()
                else set()
            )
            assert actual == expected, (
                "Dataset directory differs from reviewed manifest",
                kind,
                split,
            )
