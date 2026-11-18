"""Prepare a selected set of Dryad image/mask pairs for YOLO OBB."""

import io
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ..utils.io import read_json, write_json
from . import config
from .geometry import normalized_corners, rectangle_from_mask, required_padding

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = set(
    "avif bmp dng heic heif jp2 jpeg jpg mpo png tif tiff webp".split()
)


def prepare(root, archive_path=None, selection_path=None, *, replace=False):
    """Convert pairs into a fresh tree before replacing any generated files."""
    archive_path = Path(archive_path or config.source_archive_path(root))
    selection_path = Path(selection_path or config.source_dir(root) / "selection.json")
    destination = config.dataset_dir(root)
    generated = (
        "images",
        "masks",
        "labels",
        "data.yaml",
        "metadata/source_selection.json",
        "metadata/split_manifest.json",
    )
    for name in generated:
        output = destination / name
        if (output.exists() or output.is_symlink()) and not replace:
            raise FileExistsError(
                f"{output} exists; use prepare --replace to regenerate"
            )
        if any(
            source.absolute().is_relative_to(output.absolute())
            or source.resolve().is_relative_to(output.resolve())
            for source in (archive_path, selection_path)
        ):
            raise ValueError(
                f"Source ZIP or selection is inside generated output: {output}"
            )

    selection = read_json(selection_path)["records"]
    if {row["split"] for row in selection} != set(SPLITS):
        raise ValueError("Selection must contain train, val and test splits")
    for key in ("id", "source_image"):
        if len({row[key] for row in selection}) != len(selection):
            raise ValueError(f"Repeated {key} in selection")
    # The existing inventory may identify duplicate photos under different filenames.
    pixels = [row["pixel_sha256"] for row in selection if row.get("pixel_sha256")]
    if len(pixels) != len(set(pixels)):
        raise ValueError("Duplicate source photos in selection")

    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(dir=destination.parent, prefix=".obb-prepare-") as temp:
        staged = Path(temp)
        for split in SPLITS:
            for kind in ("images", "masks", "labels"):
                (staged / kind / split).mkdir(parents=True)
        with ZipFile(archive_path) as archive:
            for row in selection:
                identity, split = row["id"], row["split"]
                if (
                    not identity
                    or Path(identity).name != identity
                    or identity in (".", "..")
                ):
                    raise ValueError(f"Invalid image ID: {identity}")

                # Check the archive paths, not the historical manifest's split assertion.
                official = "test" if split == "test" else "train"
                photo = PurePosixPath(row["source_image"])
                mask_folder = {
                    "det_img": "seg_label",
                    f"{official}_img": f"{official}_seg_label",
                }
                mask_dir = mask_folder.get(photo.parent.name)
                if (
                    official not in photo.parts[:-2]
                    or not mask_dir
                    or row["source_mask"]
                    != str(photo.parent.parent / mask_dir / photo.name)
                ):
                    raise ValueError(
                        f"Incorrect source split or image/mask pair: {photo}"
                    )

                image_bytes = archive.read(row["source_image"])
                mask_bytes = archive.read(row["source_mask"])
                with (
                    Image.open(io.BytesIO(image_bytes)) as image,
                    Image.open(io.BytesIO(mask_bytes)) as mask_image,
                ):
                    extension = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format)
                    if not extension:
                        raise ValueError(
                            f"Unsupported photo format for {identity}: {image.format}"
                        )
                    if image.size != mask_image.size:
                        raise ValueError(
                            f"Image/mask dimensions do not match: {identity}"
                        )
                    image.load()
                    mask = np.asarray(mask_image)
                    if mask.ndim != 2 or not np.isin(mask, (0, 255)).all():
                        raise ValueError(
                            f"Expected a binary foreground mask: {identity}"
                        )

                    corners = rectangle_from_mask(mask)
                    padding = required_padding(corners, image.width, image.height)
                    left, top, right, bottom = padding
                    width = image.width + left + right
                    height = image.height + top + bottom
                    corners += [left, top]
                    normalized = normalized_corners(corners, width, height)

                    image_file = identity + (".png" if any(padding) else extension)
                    image_path = staged / "images" / split / image_file
                    mask_path = staged / "masks" / split / (identity + ".png")
                    if any(padding):
                        pixels = np.asarray(image.convert("RGB"))
                        padded = np.pad(
                            pixels, ((top, bottom), (left, right), (0, 0)), mode="edge"
                        )
                        Image.fromarray(padded).save(image_path)
                        Image.fromarray(
                            np.pad(mask, ((top, bottom), (left, right)))
                        ).save(mask_path)
                    else:
                        image_path.write_bytes(image_bytes)
                        mask_path.write_bytes(mask_bytes)

                (staged / "labels" / split / (identity + ".txt")).write_text(
                    "0 "
                    + " ".join(f"{value:.10f}" for value in normalized.ravel())
                    + "\n"
                )
                record = {
                    "id": identity,
                    "split": split,
                    "official_split": official,
                    "source_image": row["source_image"],
                    "source_mask": row["source_mask"],
                    "image_file": image_file,
                    "width": width,
                    "height": height,
                    "padding_ltrb": padding,
                    "corners_xy": corners.tolist(),
                }
                if row.get("pixel_sha256"):
                    record["source_pixel_sha256"] = row["pixel_sha256"]
                records.append(record)

        metadata = staged / "metadata"
        write_json(
            metadata / "split_manifest.json", {"task": "obb", "records": records}
        )
        shutil.copyfile(selection_path, metadata / "source_selection.json")
        for name in generated:
            output = destination / name
            if output.is_dir() and not output.is_symlink():
                shutil.rmtree(output)
            else:
                output.unlink(missing_ok=True)
            if name != "data.yaml":
                output.parent.mkdir(parents=True, exist_ok=True)
                (staged / name).replace(output)
        config.write_yaml(root)

    print(f"Prepared {len(records)} image/label pairs in {destination}")


def validate_dataset(root, splits=SPLITS):
    """Check that YOLO will see the intended images and their label files."""
    data = config.dataset_dir(root)
    manifest = read_json(config.metadata_dir(root) / "split_manifest.json")
    for split in splits:
        folder = data / "images" / split
        files = {r["image_file"] for r in manifest["records"] if r["split"] == split}
        images = {
            p.relative_to(folder).as_posix()
            for p in folder.rglob("*")
            if p.is_file() and p.suffix[1:].lower() in IMAGE_SUFFIXES
        }
        if not files or images != files:
            raise ValueError(
                f"Image inventory mismatch in {folder}: "
                f"missing={sorted(files - images)[:5]}; extra={sorted(images - files)[:5]}"
            )
        for image in files:
            label = data / "labels" / split / Path(image).with_suffix(".txt")
            if not label.is_file() or not label.stat().st_size:
                raise ValueError(f"Missing or empty label: {label}")
    return manifest
