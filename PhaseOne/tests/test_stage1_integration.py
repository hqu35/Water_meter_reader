"""Geometry and orientation invariants for the independent Stage I integration."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from unittest.mock import patch

import cv2
import numpy as np

from stage1_python_worker.obb.geometry import aligned_crop
from stage1_python_worker.obb.integrated import (
    rectify_full_image,
    rotate_180,
    transform_points,
    run_one,
)


class FullImageGeometryTests(unittest.TestCase):
    def test_angles_and_180_preserve_image_and_register(self):
        for angle in (0, 45, 89, 90, 92, 135, 179):
            with self.subTest(angle=angle):
                image = np.full((360, 420, 3), 255, np.uint8)
                rectangle = cv2.boxPoints(((210, 180), (150, 50), angle))
                # Asymmetric markers reveal left-right reflection and clipped content.
                for pt, color in zip(
                    rectangle, ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0))
                ):
                    cv2.circle(image, tuple(np.round(pt).astype(int)), 6, color, -1)
                cv2.circle(image, (12, 12), 5, (0, 0, 0), -1)
                cv2.circle(image, (407, 347), 5, (0, 0, 0), -1)
                rotated, points, matrix, _ = rectify_full_image(image, rectangle)
                self.assertGreater(np.linalg.det(matrix[:, :2]), 0.99)
                edge = points[1] - points[0]
                self.assertLess(abs(edge[1]), 1e-5)
                np.testing.assert_allclose(
                    transform_points(rectangle, matrix).mean(axis=0),
                    points.mean(axis=0),
                    atol=1e-6,
                )
                final, final_points, flip = rotate_180(rotated, points)
                np.testing.assert_allclose(final_points, transform_points(points, flip))
                np.testing.assert_array_equal(final, rotated[::-1, ::-1])
                crop, _ = aligned_crop(final, final_points, 0.03)
                self.assertGreater(crop.shape[1], crop.shape[0])
                for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)):
                    self.assertGreater(
                        np.count_nonzero(
                            np.linalg.norm(crop.astype(int) - color, axis=2) < 70
                        ),
                        2,
                    )
                for pt in ((12, 12), (407, 347)):
                    x, y = np.round(transform_points([pt], matrix)[0]).astype(int)
                    self.assertLess(np.mean(rotated[y, x]), 100)

    def test_double_rotation_round_trip(self):
        image = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        points = np.array([[2.0, 2.0], [10.0, 2.0], [10.0, 5.0], [2.0, 5.0]])
        flipped, p, _ = rotate_180(image, points)
        original, recovered, _ = rotate_180(flipped, p)
        np.testing.assert_array_equal(original, image)
        np.testing.assert_allclose(recovered, points)


class PipelineControlTests(unittest.TestCase):
    def test_unknown_orientation_preserves_raw_and_never_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.png"
            weights = root / "best.pt"
            Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
            weights.write_bytes(b"weights")
            model = SimpleNamespace(predict=lambda *a, **k: [object()])
            boxes = [
                {
                    "corners_xy": [[10, 20], [80, 20], [80, 40], [10, 40]],
                    "confidence": 0.8,
                    "class_id": 0,
                }
            ]
            with (
                patch(
                    "stage1_python_worker.obb.integrated.detections_from_result",
                    return_value=boxes,
                ),
                patch(
                    "stage1_python_worker.obb.integrated.classify_full_image",
                    return_value=(
                        "Unlabeled",
                        "unlabeled",
                        0.8,
                        {"top": "Unlabeled", "confidence": 0.8},
                    ),
                ),
            ):
                report = run_one(model, path, weights, root / "result", "secret")
            self.assertEqual(report["status"], "orientation_unresolved")
            self.assertEqual(report["orientation"]["raw_class"], "Unlabeled")
            self.assertFalse((root / "result/register_crop.png").exists())

    def test_no_detections_does_not_call_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.png"
            weights = root / "best.pt"
            Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
            weights.write_bytes(b"weights")
            model = SimpleNamespace(predict=lambda *a, **k: [object()])
            with (
                patch(
                    "stage1_python_worker.obb.integrated.detections_from_result",
                    return_value=[],
                ),
                patch(
                    "stage1_python_worker.obb.integrated.classify_full_image"
                ) as classify,
            ):
                report = run_one(model, path, weights, root / "result", "secret")
            self.assertEqual(report["status"], "no_detection")
            classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
