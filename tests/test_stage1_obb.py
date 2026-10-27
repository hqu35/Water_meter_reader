"""OBB geometry and workflow gates; generated patterns are tests, not training data."""

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from stage1_python_worker.obb import config
from stage1_python_worker.obb.geometry import (
    aligned_crop,
    mask_is_covered,
    normalized_corners,
    ordered_corners,
    rectangle_from_mask,
    required_padding,
    rotated_iou,
)
from stage1_python_worker.obb.evaluation import match_detections
from stage1_python_worker.obb.review import approve, require_approval
from stage1_python_worker.obb.training import train, training_arguments
from stage1_python_worker.utils.io import write_json


class OBBGeometryTests(unittest.TestCase):
    def test_axis_aligned_uses_pixel_edges(self):
        mask = np.zeros((30, 50), np.uint8)
        mask[7:16, 3:34] = 255
        q = rectangle_from_mask(mask)
        np.testing.assert_allclose(q, [[3, 7], [34, 7], [34, 16], [3, 16]], atol=1e-5)
        self.assertTrue(mask_is_covered(mask, q))

    def test_empty_and_single_pixel(self):
        with self.assertRaises(ValueError):
            rectangle_from_mask(np.zeros((10, 10)))
        mask = np.zeros((10, 10))
        mask[2, 3] = 1
        q = rectangle_from_mask(mask)
        self.assertAlmostEqual(cv2.contourArea(q.astype(np.float32)), 1)
        self.assertTrue(mask_is_covered(mask, q))

    def test_all_corner_permutations_same_winding_and_axis(self):
        q = cv2.boxPoints(((100, 80), (90, 24), 32))
        expected = ordered_corners(q)
        for permutation in itertools.permutations(q):
            actual = ordered_corners(permutation)
            np.testing.assert_allclose(actual, expected)
            u, v = actual[1] - actual[0], actual[2] - actual[1]
            self.assertGreater(u[0] * v[1] - u[1] * v[0], 0)
            self.assertGreaterEqual(
                np.linalg.norm(actual[1] - actual[0]),
                np.linalg.norm(actual[2] - actual[1]) - 1e-4,
            )

    def test_rotations_disconnected_pixels_and_boundary_padding(self):
        for angle in [-89, -60, -45, -1, 0, 20, 45, 60, 89, 90]:
            mask = np.zeros((140, 180), np.uint8)
            corners = cv2.boxPoints(((90, 70), (110, 25), angle)).astype(np.int32)
            cv2.fillConvexPoly(mask, corners, 255)
            mask[0, 0] = 255  # Do not silently filter disconnected foreground.
            q = rectangle_from_mask(mask)
            self.assertTrue(mask_is_covered(mask, q))
            left, top, right, bottom = required_padding(q, 180, 140)
            moved = q + [left, top]
            normal = normalized_corners(moved, 180 + left + right, 140 + top + bottom)
            self.assertTrue((normal >= 0).all() and (normal <= 1).all())
            text = " ".join(f"{v:.10f}" for v in normal.ravel())
            recovered = np.array(list(map(float, text.split()))).reshape(4, 2) * [
                180 + left + right,
                140 + top + bottom,
            ]
            np.testing.assert_allclose(moved, recovered, atol=1e-6)

    def test_normalization_rejects_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            normalized_corners([[-2, 0], [20, 0], [20, 10], [-2, 10]], 30, 30)
        with self.assertRaises(ValueError):
            ordered_corners([[0, 0], [3, 0], [2, 1], [0, 1]])

    def test_crop_exact_axis_pixel_values_no_transposition(self):
        image = np.arange(30 * 50, dtype=np.float32).reshape(30, 50)
        q = [[3, 7], [34, 7], [34, 16], [3, 16]]
        crop, report = aligned_crop(image, q, 0)
        np.testing.assert_allclose(crop, image[7:16, 3:34])
        self.assertEqual((report["width"], report["height"]), (31, 9))

    def test_rotated_crop_preserves_digit_order_and_both_edge_digits(self):
        # Asymmetric digit panel with colored corner/edge markers. Render and
        # rectify at many angles, including vertical and inverted directions.
        panel = Image.new("RGB", (160, 40), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((2, 12), "1234567890", fill="black")
        draw.rectangle((0, 0, 8, 8), fill=(255, 0, 0))
        draw.rectangle((151, 0, 159, 8), fill=(0, 255, 0))
        draw.rectangle((151, 31, 159, 39), fill=(0, 0, 255))
        draw.rectangle((0, 31, 8, 39), fill=(255, 255, 0))
        pixels = np.asarray(panel)
        for angle in [-89, -60, -30, 0, 30, 60, 89, 90, 120, 179]:
            theta = np.deg2rad(angle)
            u = np.array([np.cos(theta), np.sin(theta)])
            v = np.array([-np.sin(theta), np.cos(theta)])
            q = np.array([[-80, -20], [80, -20], [80, 20], [-80, 20]]) @ np.array(
                [u, v]
            ) + [150, 150]
            source = np.array([[0, 0], [160, 0], [160, 40], [0, 40]], np.float32) - 0.5
            matrix = cv2.getPerspectiveTransform(source, (q - 0.5).astype(np.float32))
            canvas = cv2.warpPerspective(
                pixels, matrix, (300, 300), borderValue=(180, 180, 180)
            )
            actual, _ = aligned_crop(canvas, q, 0)
            expected = (
                pixels
                if np.dot(ordered_corners(q)[1] - ordered_corners(q)[0], u) > 0
                else pixels[::-1, ::-1]
            )
            expected = cv2.resize(expected, (actual.shape[1], actual.shape[0]))
            # Compare interior and four corner colors; reflections, transpose,
            # lost edge digits, and arbitrary 90-degree changes fail this check.
            self.assertLess(
                np.abs(
                    actual.astype(float)[2:-2, 2:-2]
                    - expected.astype(float)[2:-2, 2:-2]
                ).mean(),
                15,
            )
            for y, x in [(4, 4), (4, -5), (-5, -5), (-5, 4)]:
                np.testing.assert_allclose(actual[y, x], expected[y, x], atol=35)

    def test_default_margin_retains_foreground_cells(self):
        q = cv2.boxPoints(((80, 70), (100, 30), 25))
        crop, report = aligned_crop(np.zeros((160, 160, 3), np.uint8), q)
        matrix = np.asarray(report["source_edge_to_crop_edge"])
        mapped = cv2.perspectiveTransform(q.astype(np.float32)[None], matrix)[0]
        self.assertTrue((mapped > 0).all())
        self.assertTrue((mapped < [crop.shape[1], crop.shape[0]]).all())
        self.assertTrue(report["orientation_ambiguous_180"])

    def test_iou_is_polygon_not_enclosing_box(self):
        a = cv2.boxPoints(((70, 70), (100, 10), 45))
        b = cv2.boxPoints(((70, 70), (100, 10), -45))
        self.assertAlmostEqual(rotated_iou(a, a), 1, places=5)
        self.assertLess(rotated_iou(a, b), 0.1)
        diagnostic = match_detections(
            [
                {"corners_xy": a, "confidence": 0.9},
                {"corners_xy": a, "confidence": 0.8},
            ],
            a,
        )
        self.assertEqual(
            (diagnostic["tp"], diagnostic["fp"], diagnostic["fn"]), (1, 1, 0)
        )
        self.assertEqual(match_detections([], a)["fn"], 1)

    def test_nearly_coincident_rectangles_have_stable_polygon_iou(self):
        a = np.array(
            [
                [172.217499, 0.225463853],
                [391.721466, 319.503967],
                [310.575592, 375.291748],
                [91.0716324, 56.0132446],
            ]
        )
        b = a.copy()
        b[[1, 2], 0] -= 0.00003
        b[3, 0] -= 0.000015
        self.assertGreater(rotated_iou(a, b), 0.99999)

    def test_polygon_iou_against_independent_opencv_intersections(self):
        rng = np.random.default_rng(539)
        for _ in range(100):
            a = cv2.boxPoints(
                (
                    tuple(rng.uniform(50, 100, 2)),
                    tuple(rng.uniform(15, 70, 2)),
                    float(rng.uniform(-80, 80)),
                )
            )
            b = cv2.boxPoints(
                (
                    tuple(rng.uniform(50, 100, 2)),
                    tuple(rng.uniform(15, 70, 2)),
                    float(rng.uniform(-80, 80)),
                )
            )
            area, _ = cv2.intersectConvexConvex(a, b)
            expected = area / (cv2.contourArea(a) + cv2.contourArea(b) - area)
            self.assertAlmostEqual(rotated_iou(a, b), expected, places=5)


class OBBWorkflowTests(unittest.TestCase):
    def test_train_blocks_before_gpu_or_model_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("stage1_python_worker.obb.training.verify_gpu") as gpu:
                with self.assertRaisesRegex(RuntimeError, "human visual approval"):
                    train(Path(directory))
                gpu.assert_not_called()

    def test_nonhuman_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                config.metadata_dir(root) / "annotation_review.json",
                {
                    "approved": True,
                    "reviewer": "old automated reviewer",
                    "task": "detect",
                },
            )
            with self.assertRaises(AssertionError):
                require_approval(root)
            with self.assertRaises(ValueError):
                approve(root, "human", False)

    def test_training_configuration_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = training_arguments(Path(directory))
            expected = {
                "task": "obb",
                "epochs": 30,
                "imgsz": 640,
                "batch": 32,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "seed": 20261022,
                "warmup_epochs": 3,
                "warmup_bias_lr": 0,
                "patience": 10,
                "device": 0,
                "amp": False,
                "deterministic": True,
            }
            for key, value in expected.items():
                self.assertEqual(arguments[key], value, key)
            for key in [
                "mosaic",
                "mixup",
                "copy_paste",
                "fliplr",
                "flipud",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "hsv_h",
                "hsv_s",
                "hsv_v",
            ]:
                self.assertEqual(arguments[key], 0)
            self.assertEqual(config.CONFIG["model"], "yolo11n-obb.pt")


class OBBIntegrationTests(unittest.TestCase):
    def test_source_conversion_and_moved_review_gate_detect_tampering(self):
        import contextlib
        import io
        from stage1_fixtures import create_source_fixture
        from stage1_python_worker.obb.data import prepare, validate_dataset
        from stage1_python_worker.utils.io import read_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, archive, selection = create_source_fixture(root)
            run = root / "run"
            with (
                patch.dict(config.CONFIG, settings),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                prepare(run, archive, selection)
                manifest = validate_dataset(run)
                original = read_json(selection)
                self.assertEqual(
                    [(r["split"], r["id"]) for r in manifest["records"]],
                    [(r["split"], r["id"]) for r in original["records"]],
                )
                sheets = sorted(config.annotation_dir(run).glob("*.jpg"))
                self.assertEqual(len(sheets), 4)
                self.assertFalse((config.metadata_dir(run) / "annotations").exists())
                # Approval applies only to isolated test fixtures, never real data.
                approve(run, "unit-test fixture reviewer", True)
                require_approval(run)
                sheets[0].write_bytes(sheets[0].read_bytes() + b"changed")
                with self.assertRaisesRegex(AssertionError, "Review sheet changed"):
                    require_approval(run)
                with self.assertRaises(FileExistsError):
                    prepare(run, archive, selection)

    def test_obb_output_and_crop_interface_without_network_or_model_training(self):
        import sys
        from types import SimpleNamespace
        from stage1_python_worker.obb.inference import predict

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        result = SimpleNamespace(
            obb=SimpleNamespace(
                xyxyxyxy=Tensor([[[3, 7], [34, 7], [34, 16], [3, 16]]]),
                conf=Tensor([0.95]),
                cls=Tensor([0]),
            )
        )
        model = SimpleNamespace(task="obb", predict=lambda *a, **kw: [result])
        fake = SimpleNamespace(__version__="8.4.0", YOLO=lambda *a, **kw: model)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, ultralytics=fake),
        ):
            root = Path(directory)
            image = root / "real-interface-fixture.png"
            Image.new("RGB", (50, 30), "white").save(image)
            weights = root / "fixture.pt"
            weights.write_bytes(b"interface stub, never trained")
            report = predict(weights, image, root / "prediction")
            self.assertEqual(len(report["detections"][0]["corners_xy"]), 4)
            self.assertAlmostEqual(report["detections"][0]["confidence"], 0.95)
            self.assertTrue(report["orientation_ambiguous_180"])
            with Image.open(root / "prediction/register_crop.png") as crop:
                self.assertGreater(crop.width, crop.height)
            with self.assertRaises(FileExistsError):
                predict(weights, image, root / "prediction")

    def test_colab_train_pending_approval_never_launches_remote_job(self):
        from stage1_python_worker.obb.colab import dispatch

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("stage1_python_worker.obb.colab.run_colab") as remote,
        ):
            with self.assertRaisesRegex(RuntimeError, "human visual approval"):
                dispatch("train", Path(directory), True)
            remote.assert_not_called()

    def test_colab_sync_and_remote_evaluation_target_separate_root(self):
        from stage1_python_worker.obb.colab import dispatch, sync

        with patch("stage1_python_worker.obb.colab.run_colab") as remote:
            sync()
            uploads = [
                call.args for call in remote.call_args_list if call.args[0] == "upload"
            ]
            self.assertTrue(
                any(args[-1].endswith("/configs/stage1_obb.json") for args in uploads)
            )
            self.assertTrue(
                all(
                    "/content/cs539_stage1_obb/project/" in args[-1] for args in uploads
                )
            )
            for call in remote.call_args_list:
                if call.args[0] == "exec":
                    compile(call.kwargs["stdin"], "<remote payload>", "exec")
            remote.reset_mock()
            dispatch("evaluate", Path("/unused"), True)
            payload = remote.call_args.kwargs["stdin"]
            self.assertIn(
                "'obb', '--root', '/content/cs539_stage1_obb', 'evaluate'", payload
            )
            self.assertIn(".open('x')", payload)
            compile(payload, "<remote evaluation payload>", "exec")


if __name__ == "__main__":
    unittest.main()
