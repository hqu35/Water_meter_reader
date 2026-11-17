"""Draw manual digit quadrilaterals on unchanged-size annotation source images."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.annotations.read_text())["annotations"]
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    overlays = []
    colors = [
        (30, 240, 30),
        (30, 80, 255),
        (255, 200, 30),
        (255, 40, 220),
        (0, 230, 255),
        (255, 140, 80),
    ]
    for record in records:
        source = (args.source_root / record["source_image"]).resolve()
        assert source.is_relative_to(args.source_root.resolve())
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == record["source_image_sha256"]
        )
        assert record["status"] == "complete" and record["validation"]["valid"]
        assert record["digit_point_order"] == ["TL", "BL", "BR", "TR"]
        image = cv2.imread(str(source))
        assert (
            image is not None
            and list(image.shape[1::-1]) == record["original_image_size_wh"]
        )
        if record["orientation_correction"] == "rotate_180":
            image = cv2.rotate(image, cv2.ROTATE_180)
        else:
            assert record["orientation_correction"] == "identity"
        # Subpixel fixed-point drawing preserves the manually selected corner positions.
        overlay = image.copy()
        for index, corners in enumerate(record["digit_quadrilaterals"]):
            color = colors[index % len(colors)]
            points = np.rint(np.asarray(corners) * 256).astype(np.int32)
            cv2.polylines(
                overlay,
                [points.reshape(-1, 1, 2)],
                True,
                color,
                1,
                cv2.LINE_AA,
                shift=8,
            )
            for point in points:
                cv2.circle(overlay, tuple(point), 256, color, -1, cv2.LINE_AA, shift=8)
        overlays.append((record["image_id"], overlay))
    args.output.mkdir(parents=True)
    sheet = Image.new("RGB", (1000, len(overlays) * 180), "#222222")
    draw = ImageDraw.Draw(sheet)
    for row, (image_id, overlay) in enumerate(overlays):
        path = args.output / f"{image_id}_digit_boxes.png"
        if not cv2.imwrite(str(path), overlay):
            raise OSError(f"Could not write overlay: {path}")
        preview = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        scale = min(960 / preview.width, 145 / preview.height)
        preview = preview.resize(
            (round(preview.width * scale), round(preview.height * scale)),
            Image.Resampling.NEAREST,
        )
        draw.text((20, row * 180 + 5), image_id, fill="white")
        sheet.paste(preview, (20, row * 180 + 28))
    sheet.save(args.output / "overview.png")
    print(f"Saved {len(overlays)} full-image overlays and overview to {args.output}")


if __name__ == "__main__":
    main()
