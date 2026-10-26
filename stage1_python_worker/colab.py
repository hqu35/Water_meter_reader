"""Transport and launch the same worker package in the existing Colab session."""

import shutil
import subprocess
import textwrap
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from . import config
from .data.review import require_annotation_review
from .utils.io import extract_zip_safely, read_json, sha256_file

REMOTE = "/content/cs539_stage1"
SESSION = config.CONFIG["session"]
CHUNK_BYTES = 32 * 1024 * 1024
TRANSPORT_COMMANDS = ("status", "sync", "push-prepared", "fetch")


def run_colab(*arguments, stdin=None):
    """Use the existing named session; never provision or restart a runtime."""
    binary = shutil.which("colab") or str(Path.home() / ".local/bin/colab")
    return subprocess.run(
        [binary, *arguments], input=stdin, text=True, check=True, cwd=config.REPO
    )


def sync_source() -> None:
    """Upload the worker package, experiment config and requirements once."""
    files = config.source_files()
    directories = sorted({str(path.relative_to(config.REPO).parent) for path in files})
    code = textwrap.dedent(f"""\
        from pathlib import Path
        project = Path({(REMOTE + "/project")!r})
        for directory in {directories!r}:
            (project / directory).mkdir(parents=True, exist_ok=True)
        """)
    run_colab("exec", "--session", SESSION, stdin=code)
    for path in files:
        remote_path = REMOTE + "/project/" + str(path.relative_to(config.REPO))
        run_colab("upload", "--session", SESSION, str(path), remote_path)


def build_subset_bundle(dataset_root: Path, bundle: Path) -> None:
    """Bundle approved data; exclude source archives and experiment outputs."""
    with ZipFile(bundle, "w", ZIP_STORED) as archive:
        for folder in ["images", "labels", "masks", "metadata"]:
            for path in sorted((dataset_root / folder).rglob("*")):
                if path.is_file():
                    member = config.DATASET_RELATIVE_PATH / path.relative_to(
                        dataset_root
                    )
                    archive.write(path, str(member))
        archive.write(
            dataset_root / "data.yaml", str(config.DATASET_RELATIVE_PATH / "data.yaml")
        )


def restore_uploaded_subset(run_root: Path, parts: list, expected_sha256: str) -> None:
    """Run remotely: join verified chunks and restore the reviewed dataset."""
    bundle = run_root / "prepared_subset.zip"
    with bundle.open("wb") as destination:
        for part in parts:
            path = run_root / part["name"]
            assert sha256_file(path) == part["sha256"], "Upload chunk hash mismatch"
            destination.write(path.read_bytes())
    assert sha256_file(bundle) == expected_sha256, "Uploaded subset hash mismatch"
    with ZipFile(bundle) as archive:
        extract_zip_safely(archive, run_root)
    config.write_dataset_yaml(run_root)
    require_annotation_review(run_root)
    print("Verified prepared subset transferred; raw archive was not uploaded.")


def push_prepared(run_root: Path) -> None:
    """Upload the approved subset in bounded chunks, using the same worker code."""
    require_annotation_review(run_root)
    dataset_root = config.dataset_dir(run_root)
    transfers = dataset_root / "transfers"
    transfers.mkdir(exist_ok=True)
    bundle = transfers / "prepared_subset.zip"
    build_subset_bundle(dataset_root, bundle)
    parts = []
    with bundle.open("rb") as source:
        while True:
            content = source.read(CHUNK_BYTES)
            if not content:
                break
            path = transfers / f"subset.part{len(parts):03d}"
            path.write_bytes(content)
            parts.append({"name": path.name, "sha256": sha256_file(path)})
            run_colab(
                "upload", "--session", SESSION, str(path), REMOTE + "/" + path.name
            )

    # The payload calls a package function instead of embedding another converter.
    code = textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        sys.path.insert(0, {(REMOTE + "/project")!r})
        from {config.PACKAGE_NAME}.colab import restore_uploaded_subset
        restore_uploaded_subset(Path({REMOTE!r}), {parts!r}, {sha256_file(bundle)!r})
        """)
    run_colab("exec", "--session", SESSION, "--timeout", "60", stdin=code)


def fetch_results(run_root: Path) -> None:
    """Retrieve results with the same shared hash and safe-extraction utilities."""
    report_path = run_root / "results_bundle.json"
    run_colab(
        "download",
        "--session",
        SESSION,
        REMOTE + "/results_bundle.json",
        str(report_path),
    )
    report = read_json(report_path)
    bundle = run_root / "stage1_results.zip"
    run_colab(
        "download", "--session", SESSION, REMOTE + "/stage1_results.zip", str(bundle)
    )
    assert sha256_file(bundle) == report["sha256"], "Retrieved bundle hash mismatch"
    with ZipFile(bundle) as archive:
        extract_zip_safely(archive, run_root)
    artifacts = config.artifacts_dir(run_root)
    for filename, expected_hash in read_json(
        artifacts / "artifact_checksums.json"
    ).items():
        assert sha256_file(artifacts / filename) == expected_hash, filename
    print("Verified local checkpoints, metrics and figures:", artifacts)


def run_phase(phase: str, reviewer, background: bool) -> None:
    """Run the package on Colab's Python; the Mac never performs GPU training."""
    arguments = ["-m", config.PACKAGE_NAME, "--root", REMOTE, phase]
    if reviewer:
        arguments.extend(["--reviewer", reviewer])
    project = REMOTE + "/project"
    code = textwrap.dedent(f"""\
        import subprocess
        import sys
        from pathlib import Path
        arguments = [sys.executable, *{arguments!r}]
        """)
    if background:
        log_path = REMOTE + "/" + phase + ".log"
        code += textwrap.dedent(f"""\
            with Path({log_path!r}).open('w') as log:
                process = subprocess.Popen(
                    arguments, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, cwd={project!r},
                )
            print('Started PID', process.pid, 'log:', {log_path!r})
            """)
    else:
        code += f"subprocess.run(arguments, check=True, cwd={project!r})\n"
    run_colab("exec", "--session", SESSION, "--timeout", "60", stdin=code)


def dispatch(command: str, run_root: Path, reviewer, background: bool) -> None:
    """Handle transport commands or forward a worker phase to the runtime."""
    if command == "status":
        run_colab("status", "--session", SESSION)
    elif command == "sync":
        sync_source()
    elif command == "push-prepared":
        push_prepared(run_root)
    elif command == "fetch":
        fetch_results(run_root)
    else:
        run_phase(command, reviewer, background)
