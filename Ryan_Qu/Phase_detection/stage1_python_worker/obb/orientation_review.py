"""Build a portable, offline review page from an existing orientation pilot."""

import base64
import csv
import hashlib
import io
import json
import mimetypes
from pathlib import Path


def _image_url(path, content):
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")


def write_review_interface(pilot_dir, destination=None):
    """Embed images and immutable predictions; manual labels stay browser-local."""
    pilot_dir = Path(pilot_dir)
    destination = Path(destination) if destination else pilot_dir / "index.html"
    predictions = pilot_dir / "predictions.csv"
    selection_path = pilot_dir / "selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    summary = json.loads((pilot_dir / "summary.json").read_text())
    prediction_bytes = predictions.read_bytes()
    with io.TextIOWrapper(io.BytesIO(prediction_bytes), newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = [row["example_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Review data have duplicate sample IDs")
    if identities != [record["id"] for record in selection["records"]]:
        raise ValueError("Prediction IDs differ from selected IDs")
    records = []
    for row in rows:
        record = dict(row)
        record["predicted_orientation"] = (
            int(row["predicted_orientation"]) if row["predicted_orientation"] else None
        )
        record["confidence"] = float(row["confidence"]) if row["confidence"] else None
        for field in ("original_crop_path", "corrected_crop_path"):
            crop_path = (pilot_dir / row[field]).resolve() if row[field] else None
            if crop_path and not crop_path.is_relative_to(pilot_dir.resolve()):
                raise ValueError("Crop path escapes the pilot directory")
            crop_bytes = (
                crop_path.read_bytes() if crop_path and crop_path.is_file() else None
            )
            record[field.replace("_path", "_image")] = (
                _image_url(crop_path, crop_bytes) if crop_bytes is not None else None
            )
            record[field.replace("_path", "_sha256")] = (
                hashlib.sha256(crop_bytes).hexdigest()
                if crop_bytes is not None
                else None
            )
        source_path = Path(row["source_image_path"])
        record["source_image"] = (
            _image_url(source_path, source_path.read_bytes())
            if source_path.is_file()
            else None
        )
        records.append(record)
    data = {
        "schema_version": 1,
        "pilot_id": pilot_dir.name,
        "model": summary["model"],
        "crop_source": selection["crop_source"],
        "manifest_sha256": selection["manifest_sha256"],
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "fingerprint": hashlib.sha256(
            prediction_bytes
            + selection_bytes
            + json.dumps(
                [
                    [r["original_crop_sha256"], r["corrected_crop_sha256"]]
                    for r in records
                ]
            ).encode()
        ).hexdigest(),
        "samples": records,
    }
    template = Path(__file__).with_name("orientation_review.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    destination.write_text(template.replace('"__PILOT_DATA__"', payload))
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    args = parser.parse_args()
    print(write_review_interface(args.pilot_dir))
