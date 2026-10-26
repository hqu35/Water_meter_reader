"""Stage I data conversion, deterministic selection and approval gates."""

import contextlib
import io
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from stage1_fixtures import create_reviewed_fixture, create_source_archive

from stage1_python_worker import config
from stage1_python_worker.data.archive import (
    decode_image_mask,
    discover_image_mask_pairs,
)
from stage1_python_worker.data.preparation import (
    bbox_from_mask,
    create_dataset_splits,
    normalize_box,
    prepare_dataset,
)
from stage1_python_worker.data.review import (
    approve_annotations,
    require_annotation_review,
)
from stage1_python_worker.evaluation.evaluator import prediction_batches
from stage1_python_worker.utils.io import read_json, sha256_file, write_json

FIXTURES = Path(__file__).parent / "fixtures"


class ConversionTests(unittest.TestCase):
    def test_one_pixel_and_far_edge(self):
        foreground = np.zeros((7, 11), dtype=bool)
        foreground[6, 10] = True
        box = bbox_from_mask(foreground)
        self.assertEqual(box, [10, 6, 11, 7])
        center_x, center_y, width, height = normalize_box(box, 11, 7)
        np.testing.assert_allclose(
            [
                (center_x - width / 2) * 11,
                (center_y - height / 2) * 7,
                (center_x + width / 2) * 11,
                (center_y + height / 2) * 7,
            ],
            [10, 6, 11, 7],
        )

    def test_disconnected_foreground_is_not_discarded(self):
        foreground = np.zeros((20, 30), dtype=bool)
        foreground[3:5, 4:10] = True
        foreground[18, 28] = True
        self.assertEqual(bbox_from_mask(foreground), [4, 3, 29, 19])

    def test_empty_is_explicit(self):
        self.assertIsNone(bbox_from_mask(np.zeros((3, 4), dtype=bool)))

    def test_full_image_is_valid(self):
        self.assertEqual(normalize_box([0, 0, 512, 512], 512, 512), [0.5, 0.5, 1, 1])

    def test_invalid_box_fails(self):
        with self.assertRaises(AssertionError):
            normalize_box([0, 0, 513, 512], 512, 512)

    def test_pairing_uses_full_parent_not_stem(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split in ["train", "test"]:
                    for directory in ["det_img", "seg_label"]:
                        archive.writestr(
                            f"detection/{split}/{directory}/same.png", b"fixture"
                        )
            with zipfile.ZipFile(archive_path) as archive:
                pairs = discover_image_mask_pairs(archive)
            self.assertIn("/train/", pairs["train"][0][1])
            self.assertIn("/test/", pairs["test"][0][1])

    def test_missing_pair_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("detection/train/det_img/a.png", b"fixture")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaises(ValueError):
                    discover_image_mask_pairs(archive)

    def test_actual_dryad_archive_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split in ["train", "test"]:
                    for suffix in ["img", "seg_label"]:
                        archive.writestr(
                            f"Word-Wheel_Water_Meter_Dataset/detection/{split}/{split}_{suffix}/{split}0.png",
                            b"fixture",
                        )
            with zipfile.ZipFile(archive_path) as archive:
                pairs = discover_image_mask_pairs(archive)
            self.assertEqual(len(pairs["train"]), 1)
            self.assertIn("test_seg_label", pairs["test"][0][1])

    def test_dimension_mismatch_fails(self):
        def png_bytes(size):
            buffer = io.BytesIO()
            Image.new("L", size).save(buffer, format="PNG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("image.png", png_bytes((4, 4)))
                archive.writestr("mask.png", png_bytes((3, 4)))
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaises(AssertionError):
                    decode_image_mask(archive, "image.png", "mask.png")

    def test_train_gate_cannot_approve_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                require_annotation_review(Path(temporary))

    def test_manifest_mutation_revokes_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_reviewed_fixture(root)
            (config.metadata_dir(root) / "split_manifest.json").write_text(
                '{"modified":true}'
            )
            with self.assertRaises(AssertionError):
                require_annotation_review(root)

    def test_prediction_calls_are_bounded_and_keep_manifest_order(self):
        class Model:
            def __init__(self):
                self.batch_sizes = []

            def predict(self, source, **kwargs):
                self.batch_sizes.append(len(source))
                return iter(SimpleNamespace(path=path) for path in source)

        model = Model()
        records = [{"image_file": f"{index}.jpg"} for index in [5, 2, 9, 1, 7]]
        pairs = list(prediction_batches(model, records, Path("/images"), batch_size=2))
        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual(
            [record["image_file"] for record, _ in pairs],
            [record["image_file"] for record in records],
        )

    def test_prediction_reordering_is_rejected(self):
        class Model:
            def predict(self, source, **kwargs):
                return iter(SimpleNamespace(path=path) for path in reversed(source))

        records = [{"image_file": "a.jpg"}, {"image_file": "b.jpg"}]
        with self.assertRaises(AssertionError):
            list(prediction_batches(Model(), records, Path("/images")))


class PreparationTests(unittest.TestCase):
    def test_seeded_splits_and_review_ids_match_pre_refactor_selection(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            source = root / "source.zip"
            settings = create_source_archive(source)
            manifests = []
            for name in ["first", "second"]:
                run_root = root / name
                destination = config.source_archive_path(run_root)
                destination.parent.mkdir(parents=True)
                shutil.copy2(source, destination)
                with patch.object(config, "CONFIG", settings):
                    prepare_dataset(run_root)
                metadata = config.metadata_dir(run_root)
                manifest = read_json(metadata / "split_manifest.json")
                manifests.append(manifest)
                self.assertEqual(len(manifest["records"]), 32)
                validation = read_json(metadata / "validation.json")
                selection = {
                    "records": [
                        {
                            key: record[key]
                            for key in ["id", "split", "box_xyxy", "yolo"]
                        }
                        for record in manifest["records"]
                    ],
                    "exclusions": manifest["exclusions"],
                    "review_ids": [sheet["ids"] for sheet in validation["sheets"]],
                }
                self.assertEqual(
                    selection, read_json(FIXTURES / "subset_selection.json")
                )
                with self.assertRaises(FileNotFoundError):
                    require_annotation_review(run_root)
                approve_annotations(run_root, "Fixture-only approval test")
                self.assertTrue(require_annotation_review(run_root)["approved"])
            self.assertEqual(manifests[0], manifests[1])

    def test_duplicate_and_empty_examples_are_replaced_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            settings = create_source_archive(source)
            settings.update(train_count=1, val_count=1, test_count=1)
            # Sorted traversal exposes test0 first, then the identical train0.
            ordered_rng = SimpleNamespace(shuffle=lambda candidates: None)
            with (
                patch.object(config, "CONFIG", settings),
                zipfile.ZipFile(source) as archive,
            ):
                records, exclusions = create_dataset_splits(
                    archive, root / "subset", ordered_rng
                )
            self.assertEqual(
                [record["id"] for record in records],
                ["test_test0", "train_train10", "train_train11"],
            )
            self.assertEqual(
                [record["split"] for record in records], ["test", "val", "train"]
            )
            self.assertEqual(
                [item["reason"] for item in exclusions],
                ["duplicate_decoded_image", "empty_foreground_mask"],
            )
            self.assertEqual(len({record["pixel_sha256"] for record in records}), 3)

    def test_edited_review_sheet_revokes_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            create_reviewed_fixture(run_root)
            metadata = config.metadata_dir(run_root)
            sheet = metadata / "annotations/review.jpg"
            sheet.parent.mkdir()
            sheet.write_bytes(b"reviewed sheet")
            approval = read_json(metadata / "annotation_review.json")
            approval["reviewed_sheets"] = [
                {"path": "annotations/review.jpg", "sha256": sha256_file(sheet)}
            ]
            write_json(metadata / "annotation_review.json", approval)
            sheet.write_bytes(b"edited sheet")
            with self.assertRaises(AssertionError):
                require_annotation_review(run_root)


class DatasetLayoutTests(unittest.TestCase):
    def test_local_dataset_is_separate_from_outputs(self):
        self.assertEqual(
            config.dataset_dir(config.REPO / "outputs/stage1"),
            config.REPO / "dataset/stageOne",
        )

    def test_relocated_dataset_passes_review_and_mask_mutation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = create_reviewed_fixture(root)
            self.assertTrue(require_annotation_review(root)["approved"])
            (dataset_root / "masks/train/train_fixture.png").write_bytes(b"changed")
            with self.assertRaisesRegex(AssertionError, "Mask changed"):
                require_annotation_review(root)

    def test_extra_images_cannot_silently_join_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = create_reviewed_fixture(root)
            (dataset_root / "images/train/unreviewed.png").write_bytes(b"not reviewed")
            with self.assertRaisesRegex(AssertionError, "directory differs"):
                require_annotation_review(root)
