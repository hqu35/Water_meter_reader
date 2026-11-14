"""Independent Stage I experiment: OBB -> full image rotation -> orientation -> crop."""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .geometry import aligned_crop, ordered_corners
from .inference import detections_from_result
from ..utils.io import sha256_file

MODEL_ID = "meter-orientation-classification-ugjff/3"


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ np.asarray(matrix, dtype=np.float64)[:, :2].T + matrix[:, 2]


def rectify_full_image(image, corners):
    """Rotate the whole image on an expanded canvas and carry OBB points forward.

    Pixel coordinates refer to pixel centers; image bounds extend half a pixel
    beyond the centers. Positive OpenCV angles map the long axis to the right.
    """
    q = ordered_corners(corners)
    u = q[1] - q[0]
    angle = math.degrees(math.atan2(u[1], u[0]))
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(tuple(q.mean(axis=0)), angle, 1.0)
    bounds = np.array(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ]
    )
    rotated_bounds = transform_points(bounds, matrix)
    low, high = rotated_bounds.min(axis=0), rotated_bounds.max(axis=0)
    size = np.ceil(high - low - 1e-9).astype(int)
    matrix[:, 2] += -0.5 - low
    assert np.linalg.det(matrix[:, :2]) > 0
    result = cv2.warpAffine(
        image,
        matrix,
        tuple(size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    transformed = transform_points(q, matrix)
    direction = transformed[1] - transformed[0]
    assert direction[0] > 0 and abs(direction[1]) < 1e-6 * np.linalg.norm(direction)
    moved_bounds = transform_points(bounds, matrix)
    assert np.all(moved_bounds >= -0.50001) and np.all(
        moved_bounds <= size - 0.5 + 1e-5
    )
    return result, transformed, matrix, angle


def rotate_180(image, corners):
    height, width = image.shape[:2]
    matrix = np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0]])
    return cv2.rotate(image, cv2.ROTATE_180), transform_points(corners, matrix), matrix


def classify_full_image(image_path, key):
    """Return actual hosted response, with no implicit fallback or secret logging."""
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=key)
    client.configure(
        InferenceConfiguration(
            api_key_transport="header", max_batch_size=1, max_concurrent_requests=1
        )
    )
    response = client.infer(str(image_path), model_id=MODEL_ID)
    if not isinstance(response, dict):
        raise ValueError("Invalid classifier response")
    raw_label = response.get("top")
    confidence = response.get("confidence")
    if not isinstance(raw_label, str) or not isinstance(confidence, (int, float)):
        raise ValueError("Missing classifier label or confidence")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid classifier confidence")
    normalized = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
    return raw_label, normalized, confidence, response


def save_image(path, image):
    Image.fromarray(image).save(path)


def preview(image, corners, path):
    view = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(view)
    q = [tuple(map(float, p)) for p in corners]
    draw.line(q + [q[0]], fill="red", width=max(2, round(min(view.size) / 200)))
    view.save(path)


def run_one(
    model, image_path, weights, output, key, confidence=0.25, margin=0.03, device="cpu"
):
    """Perform a single detector call and return inspectable state even on failure."""
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_image": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "checkpoint": str(weights.resolve()),
        "checkpoint_sha256": sha256_file(weights),
        "orientation_model": MODEL_ID,
        "status": "started",
        "error": None,
        "detections": [],
        "crop_path": None,
    }
    try:
        with Image.open(image_path) as im:
            if im.getexif().get(274, 1) != 1:
                raise ValueError("Normalize EXIF orientation before inference")
            image = np.asarray(im.convert("RGB"))
        result = model.predict(
            image[:, :, ::-1].copy(),
            device=device,
            imgsz=640,
            conf=confidence,
            iou=0.7,
            verbose=False,
        )[0]
        report["detections"] = sorted(
            detections_from_result(result), key=lambda d: d["confidence"], reverse=True
        )
        if not report["detections"]:
            report["status"] = "no_detection"
        else:
            q = np.asarray(report["detections"][0]["corners_xy"])
            report["selected_detection"] = report["detections"][0]
            preview(image, q, output / "original_obb.png")
            rectified, q1, affine, angle = rectify_full_image(image, q)
            save_image(output / "rectified_full.png", rectified)
            preview(rectified, q1, output / "rectified_obb.png")
            report.update(
                rotation_angle_degrees=angle,
                affine_matrix=affine.tolist(),
                expanded_canvas_wh=[rectified.shape[1], rectified.shape[0]],
                rectified_obb=q1.tolist(),
            )
            try:
                raw, label, score, response = classify_full_image(
                    output / "rectified_full.png", key
                )
            except Exception as exc:
                # External SDK exceptions may contain credentials; never persist exception text.
                report.update(status="orientation_unresolved", error=type(exc).__name__)
            else:
                if key in json.dumps(response):
                    raise ValueError("Unsafe classifier response")
                report["orientation"] = {
                    "raw_class": raw,
                    "normalized_class": label,
                    "confidence": score,
                    "raw_response": response,
                }
                if label not in ("correct", "upside_down"):
                    report["status"] = "orientation_unresolved"
                    (output / "result.json").write_text(json.dumps(report, indent=2))
                    return report
                if label == "upside_down":
                    final, q2, flip = rotate_180(rectified, q1)
                    report["rotation_180_applied"] = True
                    report["orientation_affine_matrix"] = flip.tolist()
                else:
                    final, q2 = rectified, q1
                    report["rotation_180_applied"] = False
                    report["orientation_affine_matrix"] = [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                report["final_obb"] = q2.tolist()
                save_image(output / "corrected_full.png", final)
                preview(final, q2, output / "corrected_obb.png")
                crop, crop_geometry = aligned_crop(final, q2, margin)
                if crop.shape[1] < crop.shape[0]:
                    raise ValueError("Crop is not horizontal")
                save_image(output / "register_crop.png", crop)
                report.update(
                    status="success",
                    crop_path=str((output / "register_crop.png").resolve()),
                    crop_geometry=crop_geometry,
                )
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    (output / "result.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--input-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        parser.error("ROBOFLOW_API_KEY environment variable is required")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    images = (
        [args.image]
        if args.image
        else sorted(
            p
            for p in args.input_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    )
    if not images or len({p.stem for p in images}) != len(images):
        parser.error("Input must contain uniquely named images")
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="obb")
    if model.task != "obb":
        raise ValueError("Expected OBB model")
    args.output.mkdir(parents=True)
    for path in images:
        report = run_one(
            model,
            path,
            args.weights,
            args.output / path.stem,
            key,
            args.confidence,
            args.margin,
            args.device,
        )
        print(path.name, report["status"], flush=True)


if __name__ == "__main__":
    main()
