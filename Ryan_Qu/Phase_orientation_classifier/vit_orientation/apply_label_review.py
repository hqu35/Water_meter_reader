"""Validate a review export and save an immutable corrected-label revision."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def apply(review_path, directory):
    review_path = Path(review_path)
    directory = Path(directory)
    raw = review_path.read_bytes()
    export = json.loads(raw)
    baseline = json.loads((directory / "restored_labels.json").read_text())
    records = sorted(baseline["records"], key=lambda r: r["index"])
    fingerprint = hashlib.sha256()
    for row in records:
        identity = {
            k: row.get(k)
            for k in [
                "id",
                "index",
                "restored_label",
                "status",
                "excluded_by_previous_review",
                "image_sha256",
            ]
        }
        fingerprint.update(json.dumps(identity, sort_keys=True).encode())
    if (
        export.get("schema") != "watermeter-restored-orientation-review-v1"
        or export.get("dataset_fingerprint") != fingerprint.hexdigest()
        or export.get("label_stage") != baseline["label_stage"]
    ):
        raise ValueError("Review schema, fingerprint, or image stage mismatch")
    incoming = export.get("records")
    if (
        not isinstance(incoming, list)
        or export.get("total_images") != len(records)
        or len(incoming) != len(records)
    ):
        raise ValueError("Review must contain every baseline record")
    by_id = {r["id"]: r for r in records}
    seen = set()
    corrected = []
    changes = []
    for row in incoming:
        rid = row.get("id")
        original = by_id.get(rid)
        if original is None or rid in seen:
            raise ValueError("Unknown or repeated image ID")
        seen.add(rid)
        if (
            row.get("image_sha256") != original["image_sha256"]
            or row.get("restored_label") != original["restored_label"]
            or row.get("previously_excluded") != original["excluded_by_previous_review"]
        ):
            raise ValueError(f"Image identity or baseline label differs: {rid}")
        label = row.get("label")
        reviewed = row.get("reviewed")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("Invalid label")
        if type(reviewed) is not bool or row.get("excluded") != (label is None):
            raise ValueError("Invalid review flags")
        if label != original["restored_label"] and not reviewed:
            raise ValueError("Changed label without a human review flag")
        image = directory / original["image_path"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != original["image_sha256"]:
            raise ValueError(f"Image content changed: {rid}")
        corrected.append(
            dict(
                original,
                label=label,
                human_reviewed_this_round=reviewed,
                label_evidence="human_review_export"
                if reviewed
                else "historical_restoration_retained",
                excluded=label is None,
            )
        )
        if label != original["restored_label"]:
            changes.append(
                {
                    "id": rid,
                    "previous_label": original["restored_label"],
                    "corrected_label": label,
                }
            )
    corrected.sort(key=lambda r: r["index"])
    digest = hashlib.sha256(raw).hexdigest()
    revision = directory / "revisions" / ("review_" + digest[:12])
    payload = {
        "schema": "watermeter-corrected-orientation-labels-v1",
        "label_stage": baseline["label_stage"],
        "review_export_sha256": digest,
        "baseline_fingerprint": fingerprint.hexdigest(),
        "records": corrected,
    }
    summary = {
        "images": len(corrected),
        "training_eligible": sum(r["label"] is not None for r in corrected),
        "human_reviewed_this_round": sum(
            r["human_reviewed_this_round"] for r in corrected
        ),
        "changed_labels": len(changes),
        "label_counts": dict(collections.Counter(str(r["label"]) for r in corrected)),
        "changes": changes,
        "review_export_sha256": digest,
    }
    if revision.exists():
        if (revision / "review_original.json").read_bytes() != raw:
            raise ValueError("Revision collision")
        print(json.dumps(summary, indent=2))
        print(revision)
        return revision
    revision.mkdir(parents=True)
    (revision / "review_original.json").write_bytes(raw)
    (revision / "corrected_labels.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (revision / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (revision / "dryad_training_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "source", "group_id", "label_evidence"]
        )
        writer.writeheader()
        for row in corrected:
            if row["label"] is None:
                continue
            writer.writerow(
                {
                    "path": str((directory / row["image_path"]).resolve()),
                    "label": row["label"],
                    "source": "dryad",
                    "group_id": "dryad:" + row["id"],
                    "label_evidence": row["label_evidence"],
                }
            )
    print(json.dumps(summary, indent=2))
    print("Revision:", revision)
    return revision


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review")
    p.add_argument("directory")
    a = p.parse_args()
    apply(a.review, a.directory)
