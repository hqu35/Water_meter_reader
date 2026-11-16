"""Build a portable 10-column HTML review of restored full-image orientation labels."""

import argparse
import base64
import collections
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    fingerprint = hashlib.sha256()
    files = list((args.directory / "records").glob("*.json"))
    if len(files) != 2000:
        raise ValueError(f"Expected all 2,000 records, found {len(files)}")
    for f in files:
        records.append(json.loads(f.read_text()))
    records.sort(key=lambda r: r["index"])
    assert len({r["id"] for r in records}) == 2000
    images = []
    for record in records:
        item = {
            k: record.get(k)
            for k in [
                "id",
                "index",
                "restored_label",
                "status",
                "excluded_by_previous_review",
                "image_sha256",
            ]
        }
        if record["status"] != "failed":
            path = args.directory / record["image_path"]
            image_bytes = path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != record["image_sha256"]:
                raise ValueError(f"Review image checksum mismatch: {path}")
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                im.thumbnail((640, 640))
                b = io.BytesIO()
                im.save(b, format="WEBP", quality=82, method=4)
            item["preview"] = (
                "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()
            )
        fingerprint.update(
            json.dumps(
                {k: item[k] for k in item if k != "preview"}, sort_keys=True
            ).encode()
        )
        images.append(item)
    data = {
        "schema": "watermeter-restored-orientation-dataset-v1",
        "fingerprint": fingerprint.hexdigest(),
        "label_stage": "geometrically_rectified_full_image_before_legacy_orientation_classifier",
        "images": images,
    }
    template = Path(__file__).with_name("label_review.html").read_text()
    assert template.count("__DATASET__") == 1
    html = template.replace(
        "__DATASET__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    pending_html = args.directory / "index.building.html"
    pending_html.write_text(html)
    pending_html.replace(args.directory / "index.html")
    (args.directory / "restored_labels.json").write_text(
        json.dumps(
            {
                "schema": "watermeter-restored-orientation-v1",
                "label_stage": data["label_stage"],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "count": len(records),
                "status": dict(collections.Counter(r["status"] for r in records)),
                "restored_labels": dict(
                    collections.Counter(str(r.get("restored_label")) for r in records)
                ),
                "html_bytes": len(html.encode()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
