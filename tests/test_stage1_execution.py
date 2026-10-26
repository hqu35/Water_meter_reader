"""Stage I CLI, training contracts, evaluation and checkpoint tests."""

import contextlib
import io
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from stage1_fixtures import create_reviewed_fixture

from stage1_python_worker import cli, config
from stage1_python_worker.evaluation.evaluator import (
    evaluation_arguments,
    match_prediction,
    operating_point_summary,
)
from stage1_python_worker.training import checkpoints, trainer
from stage1_python_worker.training.trainer import training_arguments
from stage1_python_worker.utils.io import read_json, sha256_file, write_json
from stage1_python_worker.utils.packaging import package_results

FIXTURES = Path(__file__).parent / "fixtures"


def normalized_arguments(arguments, run_root):
    return {
        name: value.replace(str(run_root), "<run-root>")
        if isinstance(value, str)
        else value
        for name, value in arguments.items()
    }


class TrainingContractTests(unittest.TestCase):
    def test_all_training_arguments_match_pre_refactor_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                normalized_arguments(training_arguments(root), root),
                read_json(FIXTURES / "training_arguments.json"),
            )

    def test_all_evaluation_arguments_match_pre_refactor_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                normalized_arguments(evaluation_arguments(root), root),
                read_json(FIXTURES / "evaluation_arguments.json"),
            )

    def test_training_checks_review_before_gpu_or_model_creation(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(trainer, "verify_gpu") as verify_gpu,
            patch.dict(os.environ),
        ):
            with self.assertRaises(FileNotFoundError):
                trainer.train_model(Path(temporary))
            verify_gpu.assert_not_called()

    def test_training_passes_exact_arguments_to_yolo(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict(os.environ),
        ):
            root = Path(temporary)
            artifacts = config.artifacts_dir(root)
            artifacts.mkdir()
            pretrained = root / "pretrained.pt"
            pretrained.write_bytes(b"fixture checkpoint")
            model = Mock()
            model.ckpt_path = pretrained
            model.trainer.save_dir = artifacts / "runs/register_yolo11n"
            model.trainer.best = model.trainer.save_dir / "weights/best.pt"
            model.trainer.last = model.trainer.save_dir / "weights/last.pt"
            model.train.return_value = SimpleNamespace(results_dict={"fixture": 1})
            factory = Mock(return_value=model)
            library = SimpleNamespace(__version__="8.4.0", YOLO=factory)
            with (
                patch.dict(sys.modules, {"ultralytics": library}),
                patch.object(trainer, "require_annotation_review"),
                patch.object(trainer, "verify_gpu"),
                patch.object(trainer, "normalize_checkpoints") as normalize,
                patch.object(
                    trainer.subprocess,
                    "check_output",
                    return_value="fixture environment",
                ),
            ):
                trainer.train_model(root)
            factory.assert_called_once_with("yolo11n.pt")
            self.assertEqual(
                normalized_arguments(model.train.call_args.kwargs, root),
                read_json(FIXTURES / "training_arguments.json"),
            )
            normalize.assert_called_once_with(root)
            report = read_json(artifacts / "training_complete.json")
            self.assertEqual(report["best"], str(model.trainer.best))
            source_files = read_json(artifacts / "source_checksums.json")
            self.assertIn("stage1_python_worker/training/trainer.py", source_files)


class EvaluationTests(unittest.TestCase):
    def test_matching_keeps_iou_threshold_and_empty_prediction_behavior(self):
        record = {
            "id": "example",
            "image_file": "example.jpg",
            "box_xyxy": [0, 0, 10, 10],
        }

        def tensor(values):
            array = np.array(values)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: array))

        prediction = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=tensor([[0, 0, 5, 10], [20, 20, 30, 30]]), conf=tensor([0.9, 0.8])
            )
        )
        match = match_prediction(record, prediction)
        self.assertEqual(match["best_iou"], 0.5)
        self.assertTrue(match["matched"])
        empty = SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([]), conf=tensor([])))
        miss = match_prediction(record, empty)
        self.assertEqual(miss["best_iou"], 0)
        self.assertFalse(miss["matched"])
        self.assertEqual(
            operating_point_summary([match, miss]),
            {
                "confidence": 0.25,
                "iou": 0.5,
                "images": 2,
                "localized": 1,
                "image_localization_recall": 0.5,
                "missed": 1,
                "total_predictions": 2,
            },
        )


class CliTests(unittest.TestCase):
    def test_every_existing_phase_dispatches_without_running_the_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase in cli.PHASES:
                if phase == "approve":
                    continue
                handler = Mock()
                with (
                    self.subTest(phase=phase),
                    patch.dict(cli.PHASES, {phase: handler}),
                ):
                    cli.main(["--root", str(root), phase])
                handler.assert_called_once_with(root)

    def test_approval_still_requires_a_reviewer(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(AssertionError):
                cli.main(["--root", temporary, "approve"])
            with patch.object(cli, "approve_annotations") as approve:
                cli.main(["--root", temporary, "approve", "--reviewer", "Reviewer"])
            approve.assert_called_once_with(Path(temporary), "Reviewer")

    def test_package_entry_point_lists_all_worker_phases(self):
        completed = subprocess.run(
            [sys.executable, "-m", "stage1_python_worker", "--help"],
            cwd=config.REPO,
            text=True,
            capture_output=True,
            check=True,
        )
        for phase in cli.PHASES:
            self.assertIn(phase, completed.stdout)


class PackagingTests(unittest.TestCase):
    def test_bundle_paths_and_checksums_stay_compatible(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            create_reviewed_fixture(root)
            artifacts = config.artifacts_dir(root)
            write_json(artifacts / "heldout_metrics.json", {"fixture": True})
            weights = artifacts / "runs/register_yolo11n/weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"fixture checkpoint")
            package_results(root)
            report = read_json(root / "results_bundle.json")
            self.assertEqual(report["sha256"], sha256_file(root / "stage1_results.zip"))
            with zipfile.ZipFile(root / "stage1_results.zip") as archive:
                names = archive.namelist()
                self.assertIn("artifacts/runs/register_yolo11n/weights/best.pt", names)
                self.assertIn("artifacts/dataset_metadata/split_manifest.json", names)
                self.assertFalse(any(name.startswith("dataset/") for name in names))
            for name, expected in read_json(
                artifacts / "artifact_checksums.json"
            ).items():
                self.assertEqual(sha256_file(artifacts / name), expected)


class TensorFixture:
    """Only exercises metadata-control flow; this is not a real PyTorch tensor."""

    def __init__(self, values):
        self.values = np.array(values)

    def detach(self):
        return self

    def clone(self):
        return TensorFixture(self.values.copy())


class ModelFixture:
    def __init__(self):
        self.names = {0: "item"}
        self.weights = {"weight": TensorFixture([1, 2, 3])}

    def state_dict(self):
        return self.weights


class CheckpointTests(unittest.TestCase):
    def test_normalization_changes_only_names_with_serialization_stub(self):
        def load(path, **kwargs):
            with Path(path).open("rb") as source:
                return pickle.load(source)

        def save(value, path):
            with Path(path).open("wb") as destination:
                pickle.dump(value, destination)

        torch_stub = SimpleNamespace(
            load=load,
            save=save,
            equal=lambda left, right: np.array_equal(left.values, right.values),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save(
                {
                    "model": ModelFixture(),
                    "ema": ModelFixture(),
                    "epoch": 27,
                    "optimizer": {"fixture": 123},
                },
                path,
            )
            with patch.dict(sys.modules, {"torch": torch_stub}):
                report = checkpoints.normalize_checkpoint(path)
            restored = load(path)
            self.assertEqual(restored["model"].names, {0: "register"})
            self.assertEqual(restored["ema"].names, {0: "register"})
            np.testing.assert_array_equal(
                restored["model"].state_dict()["weight"].values, [1, 2, 3]
            )
            self.assertEqual(restored["optimizer"], {"fixture": 123})
            self.assertEqual(restored["epoch"], 27)
            self.assertTrue(report["all_model_tensors_identical"])


class InferenceTests(unittest.TestCase):
    def run_prediction(self, root, boxes, scores):
        from PIL import Image

        from stage1_python_worker.inference import predict_register

        source = root / "photo.png"
        weights = root / "weights.pt"
        Image.new("RGB", (8, 9), "red").save(source)
        weights.write_bytes(b"fixture weights")
        prediction = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=SimpleNamespace(cpu=lambda: np.array(boxes)),
                conf=SimpleNamespace(cpu=lambda: np.array(scores)),
            ),
            save=lambda filename: Image.new("RGB", (8, 9), "red").save(filename),
        )
        model = Mock()
        model.predict.return_value = [prediction]
        library = SimpleNamespace(YOLO=Mock(return_value=model))
        with (
            patch.dict(sys.modules, {"ultralytics": library}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            record = predict_register(weights, source, root / "prediction")
        self.assertEqual(
            model.predict.call_args.kwargs,
            {
                "source": str(source),
                "conf": 0.25,
                "imgsz": 640,
                "device": "cpu",
                "verbose": False,
            },
        )
        self.assertEqual(read_json(root / "prediction/prediction.json"), record)
        return record

    def test_highest_confidence_box_is_rounded_clipped_and_cropped(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.run_prediction(
                root, [[0, 0, 2, 2], [2.2, 3.8, 9.1, 10.5]], [0.3, 0.9]
            )
            self.assertTrue(record["register_found"])
            self.assertEqual(record["selected_box"], [2.2, 3.8, 9.1, 10.5])
            self.assertEqual(record["crop_xyxy"], [2, 3, 8, 9])
            with Image.open(root / "prediction/register_crop.png") as crop:
                self.assertEqual(crop.size, (6, 6))

    def test_no_detection_is_reported_without_creating_a_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.run_prediction(root, [], [])
            self.assertFalse(record["register_found"])
            self.assertEqual(record["boxes_xyxy"], [])
            self.assertFalse((root / "prediction/register_crop.png").exists())

    def test_prediction_is_dispatched_through_the_same_cli(self):
        with patch("stage1_python_worker.inference.predict_register") as predict:
            cli.main(["predict", "--weights", "model.pt", "--image", "photo.jpg"])
        predict.assert_called_once_with(
            Path("model.pt"),
            Path("photo.jpg"),
            Path("outputs/stage1/prediction"),
            "cpu",
            0.25,
        )
