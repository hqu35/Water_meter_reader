"""Verified full-image manifests, grouped splits, and on-the-fly flip pairs."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare(input_csv, output, seed=42):
    with Path(input_csv).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    required = {"path", "label", "source", "group_id", "label_evidence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Input CSV must contain {sorted(required)} and at least one row"
        )
    groups, hashes, sources = defaultdict(list), {}, set()
    for row in rows:
        if row["label"] not in ("0", "1") or not all(row[k].strip() for k in required):
            raise ValueError(
                "Each full image requires a verified binary label, source, group and evidence"
            )
        p = Path(row["path"]).expanduser().resolve(strict=True)
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image: {p} and {hashes[digest]}")
        hashes[digest] = str(p)
        row.update(path=str(p), sha256=digest, label=int(row["label"]))
        groups[row["group_id"]].append(row)
        sources.add(row["source"])
    if not {"dryad", "roboflow"}.issubset(sources):
        raise ValueError("Both dryad and roboflow verified sources are required.")
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    if len(keys) < 3:
        raise ValueError("At least three independent source groups are required")
    total = sum(len(groups[k]) for k in keys)
    targets = {"train": 0.70 * total, "val": 0.15 * total, "test": 0.15 * total}
    counts = dict.fromkeys(targets, 0)
    assignments = {}
    for key in keys:
        split = max(targets, key=lambda s: (targets[s] - counts[s]) / targets[s])
        counts[split] += len(groups[key])
        assignments[key] = split
    for group, entries in groups.items():
        for row in entries:
            row["split"] = assignments[group]
    manifest = {
        "schema": "orientation-full-image-splits-v1",
        "seed": seed,
        "rotation": "virtual_180_label_inversion",
        "records": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(manifest):
    if manifest.get("schema") != "orientation-full-image-splits-v1":
        raise ValueError("Unrecognized split manifest")
    groups, hashes = {}, set()
    for row in manifest["records"]:
        if (
            row.get("split") not in ("train", "val", "test")
            or row.get("label") not in (0, 1)
            or not row.get("label_evidence")
            or not row.get("group_id")
        ):
            raise ValueError("Invalid split, label, group, or evidence")
        group = row["group_id"]
        split = row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("Source group crosses split boundaries")
        groups[group] = split
        digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        if digest != row["sha256"] or digest in hashes:
            raise ValueError("Missing, modified, or duplicated image")
        hashes.add(digest)
    if not {"dryad", "roboflow"}.issubset({r["source"] for r in manifest["records"]}):
        raise ValueError("Both sources are required")


def load_dataset(manifest, split, model, training=False):
    from PIL import Image, ImageOps
    from torch.utils.data import Dataset
    from torchvision.transforms import v2
    from .model import letterbox, preprocessing

    normalize = preprocessing(model)
    jitter = v2.ColorJitter(brightness=0.15, contrast=0.15) if training else None

    class Paired(Dataset):
        def __init__(self):
            self.records = [r for r in manifest["records"] if r["split"] == split]

        def __len__(self):
            return len(self.records) * 2

        def __getitem__(self, index):
            row = self.records[index // 2]
            flipped = index % 2
            with Image.open(row["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if flipped:
                    image = image.transpose(Image.Transpose.ROTATE_180)
                image = letterbox(image)
                if jitter:
                    image = jitter(image)
                return normalize(image), row["label"] ^ flipped

    return Paired()
