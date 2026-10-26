"""Colab transport tests use temporary local directories, never a live runtime."""

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

import yaml
from stage1_fixtures import create_reviewed_fixture

from stage1_python_worker import cli, colab, config
from stage1_python_worker.data.review import require_annotation_review
from stage1_python_worker.utils.io import extract_zip_safely, read_json, sha256_file


class ColabTests(unittest.TestCase):
    def fake_cli(self, *arguments, stdin=None):
        if arguments[0] == "exec":
            exec(compile(stdin, "<colab-test>", "exec"), {})
        elif arguments[0] in ["upload", "download"]:
            shutil.copy2(arguments[-2], arguments[-1])
        else:
            raise AssertionError(arguments)

    def test_subset_transfer_keeps_data_layout_and_excludes_raw_and_weights(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            repo = Path(temporary) / "project"
            remote = Path(temporary) / "remote"
            remote.mkdir()
            root = repo / "outputs/stage1"
            with (
                patch.object(config, "REPO", repo),
                patch.object(colab, "REMOTE", str(remote)),
                patch.object(colab, "run_colab", self.fake_cli),
            ):
                dataset_root = create_reviewed_fixture(root)
                source = dataset_root / "raw/source.zip"
                source.parent.mkdir()
                source.write_bytes(b"must not upload")
                cli.main(["--root", str(root), "colab", "push-prepared"])
                self.assertTrue(require_annotation_review(remote)["approved"])
                definition = yaml.safe_load(
                    (remote / "dataset/stageOne/data.yaml").read_text()
                )
                self.assertEqual(
                    definition["path"], str((remote / "dataset/stageOne").resolve())
                )
                with ZipFile(dataset_root / "transfers/prepared_subset.zip") as archive:
                    self.assertTrue(
                        all(
                            name.startswith("dataset/stageOne/")
                            for name in archive.namelist()
                        )
                    )
                    self.assertFalse(
                        any(
                            "/raw/" in name
                            or "/transfers/" in name
                            or name.endswith(".pt")
                            for name in archive.namelist()
                        )
                    )

    def test_synced_package_runs_independently_without_any_scripts_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            remote = Path(temporary) / "remote"
            with (
                patch.object(colab, "REMOTE", str(remote)),
                patch.object(colab, "run_colab", self.fake_cli),
            ):
                colab.sync_source()
            project = remote / "project"
            self.assertFalse((project / "scripts").exists())
            self.assertTrue((project / "stage1_python_worker/__main__.py").is_file())
            completed = subprocess.run(
                [sys.executable, "-m", "stage1_python_worker", "--help"],
                cwd=project,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("normalize-checkpoints", completed.stdout)
            self.assertIn("predict", completed.stdout)
            self.assertIn("colab", completed.stdout)

    def test_source_inventory_uploads_each_module_once(self):
        calls = []
        with patch.object(
            colab, "run_colab", lambda *args, **kwargs: calls.append((args, kwargs))
        ):
            colab.sync_source()
        uploads = [Path(args[-2]).resolve() for args, _ in calls if args[0] == "upload"]
        expected = {path.resolve() for path in config.source_files()}
        self.assertEqual(set(uploads), expected)
        self.assertEqual(len(uploads), len(expected))
        self.assertFalse(any("scripts" in path.parts for path in uploads))

    def test_remote_phase_launches_the_same_package_in_the_project_directory(self):
        for background in [False, True]:
            with (
                self.subTest(background=background),
                tempfile.TemporaryDirectory() as temporary,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                payloads = []
                with (
                    patch.object(colab, "REMOTE", temporary),
                    patch.object(
                        colab,
                        "run_colab",
                        lambda *args, stdin=None: payloads.append(stdin),
                    ),
                ):
                    colab.run_phase("train", None, background)
                subprocess_stub = SimpleNamespace(
                    run=Mock(),
                    Popen=Mock(return_value=SimpleNamespace(pid=123)),
                    STDOUT=-2,
                )
                with patch.dict(sys.modules, {"subprocess": subprocess_stub}):
                    exec(compile(payloads[0], "<remote-phase>", "exec"), {})
                called = subprocess_stub.Popen if background else subprocess_stub.run
                self.assertEqual(
                    called.call_args.args[0],
                    [
                        sys.executable,
                        "-m",
                        "stage1_python_worker",
                        "--root",
                        temporary,
                        "train",
                    ],
                )
                self.assertEqual(called.call_args.kwargs["cwd"], temporary + "/project")

    def test_remote_approval_requires_reviewer(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(colab, "dispatch") as dispatch,
        ):
            with self.assertRaises(SystemExit):
                cli.main(["--root", temporary, "colab", "approve"])
            dispatch.assert_not_called()

    def test_corrupt_upload_chunk_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "subset.part000").write_bytes(b"changed")
            with self.assertRaisesRegex(AssertionError, "chunk hash mismatch"):
                colab.restore_uploaded_subset(
                    root, [{"name": "subset.part000", "sha256": "wrong"}], "wrong"
                )
            self.assertFalse((root / "dataset").exists())

    def test_shared_zip_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bad.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("../escaped.txt", "outside")
            with ZipFile(source) as archive:
                with self.assertRaisesRegex(AssertionError, "Unsafe archive member"):
                    extract_zip_safely(archive, root / "target")
            self.assertFalse((root / "escaped.txt").exists())

    def test_fetch_verifies_the_existing_bundle_format_without_creating_a_dataset(self):
        from stage1_python_worker.utils.io import write_json
        from stage1_python_worker.utils.packaging import package_results

        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            remote = root / "remote"
            local = root / "local"
            create_reviewed_fixture(remote)
            artifacts = config.artifacts_dir(remote)
            write_json(artifacts / "heldout_metrics.json", {"fixture": True})
            weights = artifacts / "runs/register_yolo11n/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fixture trained checkpoint")
            package_results(remote)
            with (
                patch.object(colab, "REMOTE", str(remote)),
                patch.object(colab, "run_colab", self.fake_cli),
            ):
                cli.main(["--root", str(local), "colab", "fetch"])
            self.assertEqual(
                (
                    local / "artifacts/runs/register_yolo11n/weights/best.pt"
                ).read_bytes(),
                weights.read_bytes(),
            )
            self.assertEqual(
                read_json(local / "results_bundle.json")["sha256"],
                sha256_file(local / "stage1_results.zip"),
            )
            self.assertFalse((local / "dataset").exists())
