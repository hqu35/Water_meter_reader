"""Sampling, task mapping, and annotation accounting for orientation review."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from stage1_python_worker.obb.orientation_evaluation import (
    evaluate,
    export_tasks,
    select_images,
)


class OrientationEvaluationTests(unittest.TestCase):
    def test_sampling_reproducible_and_duplicate_pixels_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [("a", 1), ("b", 2), ("c", 3), ("duplicate", 3)]:
                Image.fromarray(np.full((3, 3, 3), value, np.uint8)).save(
                    root / f"{name}.png"
                )
            manifest = root / "frozen.json"
            manifest.write_text(json.dumps({"records": []}))
            a = select_images(root, 3, 42, manifest, "test", "train")
            b = select_images(root, 3, 42, manifest, "test", "train")
            self.assertEqual(a, b)
            self.assertEqual(len({r["source_pixel_sha256"] for r in a}), 3)
            with self.assertRaises(ValueError):
                select_images(root, 4, 42, manifest, "test", "train")

    def test_task_urls_and_evaluation_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "orientation_eval"
            root.mkdir()
            records = [
                {
                    "sample_id": "s1",
                    "status": "success",
                    "normalized_prediction": "correct",
                    "prediction": "Correct",
                    "applied_transform": "identity",
                    "source": "dryad",
                    "official_split": "train",
                },
                {
                    "sample_id": "s2",
                    "status": "success",
                    "normalized_prediction": "upside_down",
                    "prediction": "upside_down",
                    "applied_transform": "rotate_180",
                    "source": "dryad",
                    "official_split": "train",
                },
                {
                    "sample_id": "s3",
                    "status": "success",
                    "normalized_prediction": "correct",
                    "prediction": "Correct",
                    "applied_transform": "identity",
                    "source": "dryad",
                    "official_split": "train",
                },
                {"sample_id": "s4", "status": "failed"},
            ]
            manifest = {"samples": records}
            (root / "manifest.json").write_text(json.dumps(manifest))
            export_tasks(root, manifest, "")
            tasks = json.loads((root / "tasks.json").read_text())
            self.assertEqual(len(tasks), 3)
            self.assertEqual(
                tasks[0]["data"]["image_reference"],
                "/data/local-files/?d=orientation_eval/images/reference/s1.png",
            )

            def annotated(task, verdict):
                return {
                    **task,
                    "annotations": [
                        {
                            "result": [
                                {
                                    "from_name": "prediction_verdict",
                                    "value": {"choices": [verdict]},
                                }
                            ]
                        }
                    ],
                }

            exported = [
                annotated(tasks[0], "Correct"),
                annotated(tasks[1], "Incorrect"),
                tasks[2],
            ]
            path = root / "export.json"
            path.write_text(json.dumps(exported))
            evaluate(root / "manifest.json", path, root / "evaluation")
            summary = json.loads((root / "evaluation/summary.json").read_text())
            self.assertEqual(
                (summary["total_sampled"], summary["processed"], summary["annotated"]),
                (4, 3, 2),
            )
            self.assertEqual(
                (summary["correct"], summary["incorrect"], summary["accuracy"]),
                (1, 1, 0.5),
            )
            self.assertEqual(
                (summary["unannotated"], summary["failed_or_pending"]), (1, 1)
            )

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "orientation_evaluation/label_studio.xml"
        ).is_file(),
        "Optional local Label Studio configuration is not available",
    )
    def test_label_config_is_well_formed(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "orientation_evaluation/label_studio.xml"
        )
        root = ElementTree.parse(config).getroot()
        self.assertEqual(root.tag, "View")
        self.assertEqual(
            [c.attrib["value"] for c in root.findall(".//Choice")],
            ["Correct", "Incorrect"],
        )


if __name__ == "__main__":
    unittest.main()
