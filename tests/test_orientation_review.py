"""Portable review generation; model predictions remain immutable inputs."""

import csv
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from stage1_python_worker.obb.orientation_review import write_review_interface
from stage1_python_worker.utils.io import write_json


class OrientationReviewTests(unittest.TestCase):
    def fixture(self, root):
        photo = root / "source.png"
        Image.new("RGB", (8, 4), "white").save(photo)
        rows = []
        for index in range(3):
            name = f"example_{index}"
            for folder in ["original", "corrected"]:
                (root / folder).mkdir(exist_ok=True)
                Image.new("RGB", (8, 4), (index, 20, 80)).save(
                    root / folder / (name + ".png")
                )
            rows.append(
                {
                    "example_id": name,
                    "source_image_path": str(photo),
                    "crop_source": "ground-truth",
                    "predicted_orientation": "0",
                    "confidence": ".7",
                    "status": "success",
                    "error": "",
                    "original_crop_path": "original/" + name + ".png",
                    "corrected_crop_path": "corrected/" + name + ".png",
                }
            )
        self.write_rows(root, rows)
        write_json(
            root / "selection.json",
            {
                "crop_source": "ground-truth",
                "manifest_sha256": "fixture",
                "records": [{"id": r["example_id"]} for r in rows],
            },
        )
        write_json(root / "summary.json", {"model": "PP-LCNet_x0_25_textline_ori"})
        return rows

    def write_rows(self, root, rows):
        with (root / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def embedded(self, path):
        match = re.search(
            r'<script type="application/json" id="pilot-data">(.*?)</script>',
            path.read_text(),
            re.S,
        )
        return json.loads(match.group(1))

    def test_self_contained_images_and_unchanged_input_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            before = (root / "predictions.csv").read_bytes()
            path = write_review_interface(root)
            data = self.embedded(path)
            self.assertEqual(len(data["samples"]), 3)
            self.assertEqual(
                data["predictions_sha256"], hashlib.sha256(before).hexdigest()
            )
            for sample in data["samples"]:
                for key in [
                    "original_crop_image",
                    "corrected_crop_image",
                    "source_image",
                ]:
                    self.assertTrue(sample[key].startswith("data:image/png;base64,"))
            self.assertEqual((root / "predictions.csv").read_bytes(), before)
            self.assertIn("export-json", path.read_text())
            self.assertIn("export-csv", path.read_text())
            self.assertIn("export-html", path.read_text())

    def test_duplicate_ids_and_crop_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.fixture(root)
            rows[1]["example_id"] = rows[0]["example_id"]
            self.write_rows(root, rows)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_review_interface(root)
            rows = self.fixture(root)
            rows[0]["original_crop_path"] = "../outside.png"
            self.write_rows(root, rows)
            with self.assertRaisesRegex(ValueError, "escapes"):
                write_review_interface(root)

    def test_embedded_ids_are_escaped_and_image_changes_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.fixture(root)
            before = self.embedded(write_review_interface(root))["fingerprint"]
            Image.new("RGB", (8, 4), "red").save(root / rows[0]["original_crop_path"])
            after = self.embedded(write_review_interface(root))["fingerprint"]
            self.assertNotEqual(before, after)
            hostile = "example_</script><script>not_executed</script>"
            rows[0]["example_id"] = hostile
            self.write_rows(root, rows)
            selection = json.loads((root / "selection.json").read_text())
            selection["records"][0]["id"] = hostile
            write_json(root / "selection.json", selection)
            output = write_review_interface(root)
            self.assertNotIn(hostile, output.read_text())
            self.assertEqual(self.embedded(output)["samples"][0]["example_id"], hostile)
