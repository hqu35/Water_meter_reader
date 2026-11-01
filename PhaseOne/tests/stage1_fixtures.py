"""Tiny generated ZIP fixtures for conversion tests, never for model training."""

import io
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from stage1_python_worker.obb import config
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


def create_source_fixture(directory):
    """Build source-only image identities, without invoking a label converter."""
    import random
    from stage1_python_worker.data.archive import (
        discover_image_mask_pairs,
        decode_image_mask,
    )

    archive_path = directory / "source.zip"
    settings = create_source_archive(archive_path)
    pools = {"train": [], "test": []}
    seen = set()
    with ZipFile(archive_path) as archive:
        for official, pairs in discover_image_mask_pairs(archive).items():
            for image_name, mask_name in pairs:
                image, mask, hashes = decode_image_mask(archive, image_name, mask_name)
                if not mask.any() or hashes["pixel_sha256"] in seen:
                    continue
                seen.add(hashes["pixel_sha256"])
                pools[official].append(
                    {
                        "id": official + "_" + Path(image_name).stem,
                        "official_split": official,
                        "source_image": image_name,
                        "source_mask": mask_name,
                        "width": image.width,
                        "height": image.height,
                        **{
                            key: hashes[key]
                            for key in ["image_sha256", "mask_sha256", "pixel_sha256"]
                        },
                    }
                )
    rng = random.Random(settings["seed"])
    for pool in pools.values():
        rng.shuffle(pool)
    records = []
    for split, selected in [
        ("train", pools["train"][8:24]),
        ("val", pools["train"][:8]),
        ("test", pools["test"][:8]),
    ]:
        records.extend({**record, "split": split} for record in selected)
    selection = directory / "selection.json"
    write_json(
        selection,
        {
            "seed": settings["seed"],
            "dataset_sha256": settings["dataset_sha256"],
            "records": records,
            "exclusions": [],
        },
    )
    return settings, archive_path, selection
