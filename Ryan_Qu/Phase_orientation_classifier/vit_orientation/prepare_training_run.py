"""Freeze corrected Dryad + Roboflow v3 inputs, deduplicate, and group-split."""

import argparse
import collections
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from PIL import Image, ImageOps
from .data import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    revision = root / "outputs/orientation_label_review/revisions/review_2468f6812375"
    candidates = list(csv.DictReader((revision / "dryad_training_labels.csv").open()))
    for r in candidates:
        r.update(
            original_split="local_review_pool",
            original_filename=Path(r["path"]).name,
            source_version="human_review_2468f6812375",
        )
    archive = root / "outputs/datasets/roboflow-orientation-v3.zip"
    folder = root / "outputs/datasets/roboflow_v3"
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha != "cf19c204c15453b7adaba795cc251352cbf32ceae2f1bd99c0cfaa1d0502e4fb":
        raise ValueError("Roboflow archive checksum changed")
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = Path(name).parts
            if (
                len(parts) != 3
                or parts[0] not in ("train", "valid", "test")
                or parts[1] not in ("Correct", "upside_down")
            ):
                raise ValueError("Unrecognized archive label/path: " + name)
            match = re.fullmatch(
                r"dbg_(\d+)_rotated_with_obb(?:_flipped)?_png\.rf\.[^.]+\.(?:jpg|jpeg|png)",
                parts[2],
            )
            if not match:
                raise ValueError("Cannot recover source group: " + name)
            dest = folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = z.read(name)
            if dest.exists():
                if dest.read_bytes() != content:
                    raise ValueError("Previously extracted content changed")
            else:
                dest.write_bytes(content)
            candidates.append(
                {
                    "path": str(dest.resolve()),
                    "label": str(int(parts[1] == "upside_down")),
                    "source": "roboflow",
                    "group_id": "roboflow:dbg_" + match.group(1),
                    "label_evidence": "roboflow_v3_folder_class",
                    "original_split": parts[0],
                    "original_filename": parts[2],
                    "source_version": "meter-orientation-classification-ugjff/3",
                }
            )
    pixel_groups = collections.defaultdict(list)
    for r in candidates:
        with Image.open(r["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            key = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        r["_decoded_hash"] = key
        pixel_groups[key].append(r)
    conflict_groups = set()
    for entries in pixel_groups.values():
        if len({r["label"] for r in entries}) > 1:
            conflict_groups.update(r["group_id"] for r in entries)
    quarantined = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in candidates
        if r["group_id"] in conflict_groups
    ]
    kept = []
    seen = {}
    duplicates = []
    for r in candidates:
        if r["group_id"] in conflict_groups:
            continue
        key = r.pop("_decoded_hash")
        if key in seen:
            prev = seen[key]
            if prev["group_id"] != r["group_id"]:
                raise ValueError(
                    "Identical image has different source IDs; resolve grouping before training"
                )
            duplicates.append({"removed": r["path"], "retained": prev["path"]})
            continue
        seen[key] = r
        kept.append(r)
    csv_path = a.output / "combined_labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    manifest = prepare(csv_path, a.output / "split_manifest.json", 42)
    source_by_group = {}
    for row in kept:
        source_by_group.setdefault(row["group_id"], row["source"])
    summary = {
        "seed": 42,
        "candidate_images": dict(collections.Counter(r["source"] for r in candidates)),
        "retained_images": dict(collections.Counter(r["source"] for r in kept)),
        "distinct_source_groups": dict(collections.Counter(source_by_group.values())),
        "exact_pixel_duplicates_removed": len(duplicates),
        "conflicting_label_groups_excluded": sorted(conflict_groups),
        "conflicting_group_images_excluded": len(quarantined),
        "splits": {
            split: {
                "images": sum(r["split"] == split for r in manifest["records"]),
                "virtual_orientation_samples": 2
                * sum(r["split"] == split for r in manifest["records"]),
                "source_counts": dict(
                    collections.Counter(
                        r["source"] for r in manifest["records"] if r["split"] == split
                    )
                ),
            }
            for split in ("train", "val", "test")
        },
        "limitations": [
            "Roboflow v3 images were already stretched to 640x640 and photometrically augmented before publication. Original pristine images are unavailable in this export.",
            "All filename-derived variants of each Roboflow dbg ID are held in one split. Original Roboflow splits are metadata only.",
            "No new stochastic augmentation is applied to validation or test.",
        ],
    }
    (a.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.output / "deduplication.json").write_text(
        json.dumps(duplicates, indent=2) + "\n"
    )
    (a.output / "conflicting_labels_excluded.json").write_text(
        json.dumps(quarantined, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
