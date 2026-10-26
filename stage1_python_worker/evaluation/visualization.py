"""Render held-out failures and seeded random predictions with ground-truth boxes."""

import random
from pathlib import Path

from .. import config


def save_prediction_sheets(records: list, image_root: Path, artifacts: Path) -> None:
    """Keep the original two sheets: eight worst-IoU and eight random examples."""
    from PIL import Image, ImageDraw

    selected = sorted(records, key=lambda record: record["best_iou"])[:8]
    selected += random.Random(config.CONFIG["seed"]).sample(records, 8)
    for sheet_index in range(2):
        sheet = Image.new("RGB", (1600, 1800), "white")
        draw = ImageDraw.Draw(sheet)
        group = selected[sheet_index * 8 : (sheet_index + 1) * 8]
        for index, record in enumerate(group):
            image = Image.open(image_root / record["image_file"]).convert("RGB")
            image_draw = ImageDraw.Draw(image)
            image_draw.rectangle(record["gt_box"], outline="#28e43e", width=3)
            for box in record["boxes"]:
                image_draw.rectangle(box, outline="#ff4030", width=3)
            x = (index % 2) * 800
            y = (index // 2) * 450
            sheet.paste(image.resize((416, 416)), (x + 180, y + 28))
            title = (
                record["id"] + f" / IoU {record['best_iou']:.3f} / GREEN GT; RED PRED"
            )
            draw.text((x + 15, y + 7), title, fill="black")
        sheet.save(
            artifacts / f"test_predictions_{sheet_index + 1:02d}.jpg", quality=93
        )
