"""Render optional mask, OBB and crop previews."""

import random

import numpy as np
from PIL import Image, ImageDraw

from ..utils.io import read_json
from . import config
from .geometry import aligned_crop


def render_preview(root, directory):
    records = read_json(config.metadata_dir(root) / "split_manifest.json")["records"]
    data = config.dataset_dir(root)
    rng = random.Random(config.CONFIG["seed"])
    selected = []
    # Include the largest padding cases as well as deterministic random examples.
    for split, count in [("train", 16), ("val", 8), ("test", 8)]:
        pool = [r for r in records if r["split"] == split]
        extremes = sorted(pool, key=lambda r: sum(r["padding_ltrb"]), reverse=True)[:2]
        others = [r for r in pool if r not in extremes]
        selected.extend(
            extremes + rng.sample(others, min(count - len(extremes), len(others)))
        )
    directory.mkdir(parents=True)
    for start in range(0, len(selected), 8):
        sheet = Image.new("RGB", (1440, 8 * 270 + 48), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            "OBB REVIEW: original | foreground mask + OBB | aligned crop (180-degree polarity unresolved)",
            fill="black",
        )
        for row, record in enumerate(selected[start : start + 8]):
            image = Image.open(
                data / "images" / record["split"] / record["image_file"]
            ).convert("RGB")
            mask = (
                np.asarray(
                    Image.open(
                        data / "masks" / record["split"] / (record["id"] + ".png")
                    )
                )
                != 0
            )
            pixels = np.asarray(image).copy()
            pixels[mask] = (
                pixels[mask] * 0.55 + np.array([40, 240, 70]) * 0.45
            ).astype(np.uint8)
            overlay = Image.fromarray(pixels)
            polygon = [tuple(p) for p in record["corners_xy"]]
            painter = ImageDraw.Draw(overlay)
            painter.line(polygon + [polygon[0]], fill="red", width=3)
            for index, point in enumerate(polygon):
                painter.text(point, str(index), fill="yellow")
            crop, _ = aligned_crop(
                np.asarray(image),
                record["corners_xy"],
                config.CONFIG["crop_margin_fraction"],
            )
            panels = [image, overlay, Image.fromarray(crop)]
            y = 48 + row * 270
            draw.text(
                (12, y),
                f"{record['split']} / {record['id']}  padding LTRB={record['padding_ltrb']}",
                fill="black",
            )
            for column, panel in enumerate(panels):
                panel.thumbnail((460, 235))
                sheet.paste(panel, (column * 480 + 10, y + 22))
        name = f"obb_review_{start // 8 + 1:02d}.jpg"
        path = directory / name
        sheet.save(path, quality=95)
    print(f"Saved previews for {len(selected)} images in {directory}")
