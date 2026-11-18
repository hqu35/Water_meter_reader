"""Build a portable ten-column human review of unseen orientation predictions."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source = json.loads((args.directory / "predictions.json").read_text())
    items = []
    fingerprint = hashlib.sha256()
    for record in source["records"]:
        item = {
            "id": record["id"],
            "index": record["index"],
            "predicted_label": record.get("predicted_label"),
            "confidence": record.get("confidence"),
            "status": "success" if record["status"] == "success" else "failed",
            "initially_excluded": False,
            "image_sha256": record.get("rectified_sha256"),
            "original_sha256": record["original_sha256"],
        }
        if record["status"] == "success":
            path = args.directory / record["rectified_image_path"]
            image_bytes = path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != record["rectified_sha256"]:
                raise ValueError(f"Review image checksum mismatch: {path}")
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                im.thumbnail((900, 900))
                buffer = io.BytesIO()
                im.save(buffer, format="WEBP", quality=87, method=4)
            item["preview"] = (
                "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()
            )
        fingerprint.update(
            json.dumps(
                {k: v for k, v in item.items() if k != "preview"}, sort_keys=True
            ).encode()
        )
        items.append(item)
    template = Path(__file__).with_name("label_review.html").read_text()
    replacements = [
        (
            "Water Meter Orientation Label Review",
            "Unseen Water Meter Orientation — 100-Image Pilot",
        ),
        (
            "Full geometrically rectified images, <b>before</b> the old classifier's 180° correction. Judge the visible digit reading direction.",
            "Previously unused image IDs: Stage I geometric rectification → trained ViT-Small. Displayed images have <b>not</b> been flipped by the classifier.",
        ),
        (
            "0 = Upright · 1 = Upside down · Each row contains 10 images. Click a picture to enlarge it.",
            "0 = Normal · 1 = Upside down · 10 images per row. Click to enlarge. Select the actual visible orientation to confirm or correct the prediction.",
        ),
        ("Upright", "Normal"),
        ("upright", "normal"),
        ("restored_label", "predicted_label"),
        ("Restored:", "Prediction:"),
        ("Restored: ", "Prediction: "),
        ("restored-orientation-review-v1", "unseen-orientation-review-v1"),
        (
            "watermeter_orientation_label_review.json",
            "watermeter_unseen100_orientation_review.json",
        ),
        ("excluded_by_previous_review", "initially_excluded"),
        (
            "This image was previously rejected for crop or information quality. Restore it to the labeled set?",
            "Restore this excluded image to the labeled set?",
        ),
        ("Previously rejected; excluded from training.", "Excluded from this review."),
        (
            "restored.textContent='Prediction: '+labelName(item.predicted_label)",
            "restored.textContent='Prediction: '+labelName(item.predicted_label)+(item.confidence==null?'':' · score '+(item.confidence*100).toFixed(1)+'%')",
        ),
        (
            "image_sha256:item.image_sha256,predicted_label:item.predicted_label",
            "image_sha256:item.image_sha256,original_sha256:item.original_sha256,confidence:item.confidence,predicted_label:item.predicted_label",
        ),
    ]
    for old, new in replacements:
        template = template.replace(old, new)
    # The viewer uses the same immutable prediction as its card.
    template = template.replace("' · Restored: '", "' · Prediction: '")
    data = {
        "schema": "watermeter-unseen-orientation-dataset-v1",
        "fingerprint": fingerprint.hexdigest(),
        "label_stage": "full_stage1_geometrically_rectified_image_before_classifier_correction",
        "config": source["config"],
        "images": items,
    }
    assert len({r["id"] for r in items}) == len(items) == 100
    template = template.replace(
        "__DATASET__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    (args.directory / "index.html").write_text(template)
    (args.directory / "review_dataset.json").write_text(
        json.dumps(
            {k: v for k, v in data.items() if k != "images"}
            | {
                "images": [
                    {k: v for k, v in i.items() if k != "preview"} for i in items
                ]
            },
            indent=2,
        )
        + "\n"
    )
    print("Built", args.directory / "index.html", "with", len(items), "fixed samples")


if __name__ == "__main__":
    main()
