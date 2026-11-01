"""Exact image, label and mask membership checks for the oriented dataset."""

from pathlib import Path


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
