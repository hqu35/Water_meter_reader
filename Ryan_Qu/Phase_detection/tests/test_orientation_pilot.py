"""Pilot contract checks without pretrained-model downloads or dataset writes."""

import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from stage1_python_worker.obb import config
from stage1_python_worker.obb.orientation_pilot import (
    correct_crop,
    parse_orientation,
    run,
    select_training_records,
)
from stage1_python_worker.utils.io import write_json


class OrientationPilotTests(unittest.TestCase):
    def test_selection_is_training_only_and_lexicographic(self):
        records = [
            {"id": f"train_{i:03}", "split": "train"} for i in reversed(range(110))
        ]
        records.append({"id": "000_before_everything", "split": "test"})
        selected = select_training_records({"records": records})
        self.assertEqual(
            [r["id"] for r in selected], [f"train_{i:03}" for i in range(100)]
        )

    def test_orientation_parser_rejects_unexpected_predictions(self):
        self.assertEqual(
            parse_orientation(
                {"label_names": ["180_degree"], "class_ids": [1], "scores": [0.8]}
            ),
            (180, 0.8),
        )
        for label, class_id, score in [
            ("90_degree", 1, 0.8),
            ("0_degree", 1, 0.8),
            ("0_degree", 0, float("nan")),
        ]:
            with self.assertRaises(ValueError):
                parse_orientation(
                    {"label_names": [label], "class_ids": [class_id], "scores": [score]}
                )

    def test_correction_is_exact_rotation_or_identity(self):
        pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        image = Image.fromarray(pixels)
        np.testing.assert_array_equal(
            np.asarray(correct_crop(image, 180)), pixels[::-1, ::-1]
        )
        np.testing.assert_array_equal(np.asarray(correct_crop(image, 0)), pixels)

    def test_sample_failures_continue_and_outputs_never_overwrite(self):
        class Classifier:
            def __init__(self, **kwargs):
                pass

            def predict(self, path, batch_size):
                index = int(Path(path).stem.split("_")[-1])
                if index in [5, 99]:
                    raise RuntimeError("deliberate isolated fixture failure")
                class_id = index % 2
                label = "180_degree" if class_id else "0_degree"

                class Result(dict):
                    @property
                    def json(self):
                        return dict(self)

                return [Result(label_names=[label], class_ids=[class_id], scores=[0.9])]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {
                    "id": f"train_{i:03}",
                    "split": "train",
                    "image_file": f"image_{i}.png",
                    "image_sha256": "fixture",
                    "label_sha256": "fixture",
                }
                for i in range(100)
            ]
            manifest = config.metadata_dir(root) / "split_manifest.json"
            write_json(manifest, {"records": records})
            before = manifest.read_bytes()
            fake = SimpleNamespace(TextLineOrientationClassification=Classifier)
            with (
                patch.dict(sys.modules, paddleocr=fake),
                patch.dict(os.environ, {}, clear=False),
                patch(
                    "stage1_python_worker.obb.orientation_pilot.ground_truth_crop",
                    return_value=(Image.new("RGB", (40, 8), "white"), {}),
                ),
                patch("importlib.metadata.version", return_value="test"),
                patch(
                    "stage1_python_worker.obb.orientation_review.write_review_interface"
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                output = root / "pilot"
                summary = run(root, output)
                self.assertEqual((summary["successful"], summary["failed"]), (98, 2))
                with (output / "predictions.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 100)
                self.assertEqual(len(summary["contact_sheets"]), 10)
                self.assertEqual(sum(row["status"] == "failed" for row in rows), 2)
                self.assertEqual(manifest.read_bytes(), before)
                with self.assertRaises(FileExistsError):
                    run(root, output)
                with self.assertRaisesRegex(ValueError, "no source fallback"):
                    run(root, root / "other", "inference")


if __name__ == "__main__":
    unittest.main()
