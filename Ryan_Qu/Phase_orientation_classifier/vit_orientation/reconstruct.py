"""Reconstruct full rectified images and propose crop-to-frame polarity matches.

The polarity match is an unverified proposal, never a training label. Uses the
original Dryad segmentation mask and the existing Stage I geometry functions.
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1].parent
# Stage I is a sibling project used by this replay script.
sys.path.insert(0, str(PROJECT / "Phase_detection"))
from stage1_python_worker.obb.geometry import rectangle_from_mask, aligned_crop  # noqa: E402
from stage1_python_worker.obb.integrated import rectify_full_image  # noqa: E402


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dryad-train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "images").mkdir(exist_ok=True)
    records = a.output / "proposals.jsonl"
    done = (
        {json.loads(line)["id"] for line in records.read_text().splitlines()}
        if records.exists()
        else set()
    )
    rows = [
        r
        for r in manifest["records"]
        if not r["user_excluded"] and r["original_id"] not in done
    ]
    if a.limit is not None:
        rows = rows[: a.limit]
    for index, row in enumerate(rows, 1):
        id = row["original_id"]
        record = {
            "id": id,
            "status": "review_required",
            "label_source": "reviewed_register_crop",
            "reviewed_crop_orientation": row["orientation_label"],
            "source_mapping_method": row["source_mapping_method"],
        }
        try:
            source = a.dryad_train / "train_img" / f"{id}.png"
            maskpath = a.dryad_train / "train_seg_label" / f"{id}.png"
            crop_path = a.manifest.parent / row["packaged_image_path"]
            image = cv2.imread(str(source))
            mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE)
            reference = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or reference is None:
                raise ValueError("Missing original, mask or reviewed crop")
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError("Original and mask dimensions differ")
            corners = rectangle_from_mask(mask)
            full, corners2, matrix, angle = rectify_full_image(image, corners)
            crop, _ = aligned_crop(full, corners2, 0.03)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (reference.shape[1], reference.shape[0]))

            def similarity(x):
                return float(
                    cv2.matchTemplate(x, reference, cv2.TM_CCOEFF_NORMED)[0, 0]
                )

            score_0 = similarity(crop)
            score_180 = similarity(cv2.rotate(crop, cv2.ROTATE_180))
            if not np.isfinite([score_0, score_180]).all():
                raise ValueError("Undefined matching scores")
            flip = score_180 > score_0
            if flip:
                full = cv2.rotate(full, cv2.ROTATE_180)
            path = a.output / "images" / f"{id}.png"
            if path.exists():
                raise FileExistsError("Output exists without journal entry")
            if not cv2.imwrite(str(path), full):
                raise IOError("Image write failed")
            record.update(
                path=str(path.resolve()),
                original_path=str(source.resolve()),
                mask_path=str(maskpath.resolve()),
                reviewed_crop_path=str(crop_path.resolve()),
                rotation_angle_degrees=angle,
                affine_matrix=matrix.tolist(),
                additional_180=flip,
                score_0=score_0,
                score_180=score_180,
                score_margin=abs(score_0 - score_180),
                proposed_label=0 if row["orientation_label"] == "upright" else 1,
            )
        except Exception as e:
            record.update(status="failed", error=f"{type(e).__name__}: {e}")
        with records.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        if index % 50 == 0 or index == len(rows):
            print(
                f"{index}/{len(rows)} completed; latest: {id} {record['status']}",
                flush=True,
            )


if __name__ == "__main__":
    run()
