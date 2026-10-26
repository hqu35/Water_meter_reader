"""Tiny generated ZIP fixtures for conversion tests, never for model training."""

import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from stage1_python_worker import config
from stage1_python_worker.utils.io import sha256_file, write_json


def create_source_archive(path: Path) -> dict:
    """Include JPEG-in-PNG names, empty masks and cross-pool duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        for split, count in [("train", 40), ("test", 16)]:
            for index in range(count):
                color_id = index if split == "train" or index == 0 else index + 80
                image = Image.new(
                    "RGB", (24, 20), (color_id, color_id * 2, color_id * 3 % 256)
                )
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG" if index % 2 == 0 else "PNG")
                mask = Image.new("L", (24, 20), 0)
                if index != 1:
                    ImageDraw.Draw(mask).rectangle(
                        (2, 3, 4 + index % 10, 5 + index % 8), fill=255
                    )
                mask_bytes = io.BytesIO()
                mask.save(mask_bytes, format="PNG")
                base = f"Word-Wheel_Water_Meter_Dataset/detection/{split}"
                archive.writestr(
                    f"{base}/{split}_img/{split}{index}.png", image_bytes.getvalue()
                )
                archive.writestr(
                    f"{base}/{split}_seg_label/{split}{index}.png",
                    mask_bytes.getvalue(),
                )
    return {
        **config.CONFIG,
        "train_count": 16,
        "val_count": 8,
        "test_count": 8,
        "dataset_sha256": sha256_file(path),
        "dataset_bytes": path.stat().st_size,
    }


def create_reviewed_fixture(run_root: Path) -> Path:
    """Build three unique examples to exercise integrity gates, not to fit a model."""
    dataset_root = config.dataset_dir(run_root)
    metadata = config.metadata_dir(run_root)
    records = []
    for index, split in enumerate(["train", "val", "test"]):
        example_id = split + "_fixture"
        image_path = dataset_root / "images" / split / (example_id + ".png")
        label_path = dataset_root / "labels" / split / (example_id + ".txt")
        mask_path = dataset_root / "masks" / split / (example_id + ".png")
        for path in [image_path, label_path, mask_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (4, 4), (index * 40, 20, 80))
        image.save(image_path)
        Image.new("L", (4, 4), 255).save(mask_path)
        label_path.write_text("0 0.5 0.5 1 1\n")
        records.append(
            {
                "id": example_id,
                "split": split,
                "image_file": image_path.name,
                "label_sha256": sha256_file(label_path),
                "mask_sha256": sha256_file(mask_path),
                "pixel_sha256": hashlib.sha256(np.array(image).tobytes()).hexdigest(),
            }
        )
    write_json(metadata / "split_manifest.json", {"records": records})
    write_json(
        metadata / "annotation_review.json",
        {
            "approved": True,
            "reviewed_image_count": 32,
            "manifest_sha256": sha256_file(metadata / "split_manifest.json"),
            "reviewed_sheets": [],
        },
    )
    config.write_dataset_yaml(run_root)
    return dataset_root
