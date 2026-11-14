"""OBB transport using cs539-stage1; separate remote root, no GPU provisioning."""

import shutil
import subprocess
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from .. import config as repository
from ..utils.io import extract_zip_safely, read_json, sha256_file, write_json
from . import config
from .review import require_approval

SESSION = config.CONFIG["session"]
CHUNK_BYTES = 32 * 1024 * 1024


def run_colab(*arguments, stdin=None):
    """Use only the existing named Colab session."""
    binary = shutil.which("colab") or str(Path.home() / ".local/bin/colab")
    return subprocess.run(
        [binary, *arguments], input=stdin, text=True, check=True, cwd=repository.REPO
    )


REMOTE = "/content/cs539_stage1_obb"
COMMANDS = (
    "status",
    "sync",
    "push-prepared",
    "gpu",
    "train",
    "evaluate",
    "package",
    "fetch",
)


def sync():
    files = repository.source_files()
    directories = sorted({str(p.relative_to(repository.REPO).parent) for p in files})
    run_colab(
        "exec",
        "--session",
        SESSION,
        stdin=textwrap.dedent(f"""\
        from pathlib import Path
        for directory in {directories!r}:
            (Path({(REMOTE + "/project")!r}) / directory).mkdir(parents=True, exist_ok=True)
        """),
    )
    for path in files:
        run_colab(
            "upload",
            "--session",
            SESSION,
            str(path),
            REMOTE + "/project/" + str(path.relative_to(repository.REPO)),
        )


def restore(root, parts, checksum):
    destination = config.dataset_dir(root)
    if destination.exists():
        raise FileExistsError("Remote OBB dataset exists; refusing to overwrite")
    bundle = root / "prepared_obb_subset.zip"
    with bundle.open("xb") as output:
        for part in parts:
            path = root / part["name"]
            assert sha256_file(path) == part["sha256"], "Chunk checksum mismatch"
            output.write(path.read_bytes())
    assert sha256_file(bundle) == checksum, "Subset checksum mismatch"
    with ZipFile(bundle) as archive:
        extract_zip_safely(archive, root)
    config.write_yaml(root)
    require_approval(root)


def push(root):
    require_approval(root)
    data = config.dataset_dir(root)
    directory = root / "transfers"
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "prepared_obb_subset.zip"
    if bundle.exists():
        raise FileExistsError(
            "Transfer exists; preserve it and use a fresh transfer directory"
        )
    with ZipFile(bundle, "x", ZIP_STORED) as archive:
        for kind in ["images", "labels", "masks", "metadata"]:
            for path in sorted((data / kind).rglob("*")):
                if path.is_file():
                    archive.write(
                        path, str(Path("dataset/stageOneOBB") / path.relative_to(data))
                    )
        for path in sorted(config.annotation_dir(root).glob("*.jpg")):
            archive.write(path, "artifacts/annotation/" + path.name)
    run_colab(
        "exec",
        "--session",
        SESSION,
        stdin=f"from pathlib import Path\nPath({REMOTE!r}).mkdir(parents=True, exist_ok=True)\n",
    )
    parts = []
    with bundle.open("rb") as stream:
        while True:
            content = stream.read(CHUNK_BYTES)
            if not content:
                break
            path = directory / f"obb_subset.part{len(parts):03d}"
            path.write_bytes(content)
            parts.append({"name": path.name, "sha256": sha256_file(path)})
            run_colab(
                "upload", "--session", SESSION, str(path), REMOTE + "/" + path.name
            )
    run_colab(
        "exec",
        "--session",
        SESSION,
        "--timeout",
        "60",
        stdin=textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        sys.path.insert(0, {(REMOTE + "/project")!r})
        from stage1_python_worker.obb.colab import restore
        restore(Path({REMOTE!r}), {parts!r}, {sha256_file(bundle)!r})
        """),
    )


def package(root):
    artifacts = root / "artifacts"
    assert (artifacts / "heldout_obb_metrics.json").exists(), "Evaluate OBB first"
    bundle = root / "stage1_obb_results.zip"
    if bundle.exists():
        raise FileExistsError("Preserving existing OBB result bundle")
    shutil.copytree(config.metadata_dir(root), artifacts / "dataset_metadata")
    checksums = {
        str(p.relative_to(artifacts)): sha256_file(p)
        for p in artifacts.rglob("*")
        if p.is_file()
    }
    write_json(artifacts / "artifact_checksums.json", checksums)
    with ZipFile(bundle, "x", ZIP_DEFLATED) as archive:
        for path in artifacts.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(root)))
    write_json(root / "obb_results_bundle.json", {"sha256": sha256_file(bundle)})


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "artifacts").exists() or (root / "stage1_obb_results.zip").exists():
        raise FileExistsError("Fetch needs an output root without existing results")
    report = root / "obb_results_bundle.json"
    bundle = root / "stage1_obb_results.zip"
    for remote, local in [
        ("obb_results_bundle.json", report),
        ("stage1_obb_results.zip", bundle),
    ]:
        run_colab("download", "--session", SESSION, REMOTE + "/" + remote, str(local))
    assert sha256_file(bundle) == read_json(report)["sha256"]
    with ZipFile(bundle) as archive:
        extract_zip_safely(archive, root)
    for relative, checksum in read_json(
        root / "artifacts/artifact_checksums.json"
    ).items():
        assert sha256_file(root / "artifacts" / relative) == checksum


def dispatch(phase, root, background=False):
    if phase == "status":
        run_colab("status", "--session", SESSION)
    elif phase == "sync":
        sync()
    elif phase == "push-prepared":
        push(root)
    elif phase == "fetch":
        fetch(root)
    else:
        if phase == "train":
            require_approval(root)  # Local gate as well as the remote worker gate.
        arguments = ["-m", "stage1_python_worker", "obb", "--root", REMOTE, phase]
        code = f"import subprocess, sys\nfrom pathlib import Path\nargs = [sys.executable, *{arguments!r}]\n"
        if background:
            code += textwrap.dedent(f"""\
                with Path({(REMOTE + "/" + phase + ".log")!r}).open('x') as log:
                    process = subprocess.Popen(args, cwd={(REMOTE + "/project")!r},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                print('Started PID', process.pid)
                """)
        else:
            code += f"subprocess.run(args, cwd={(REMOTE + '/project')!r}, check=True)\n"
        run_colab("exec", "--session", SESSION, "--timeout", "60", stdin=code)
