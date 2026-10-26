"""Package experiment artifacts and audit metadata, excluding image datasets."""

import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .. import config
from .io import sha256_file, write_json


def package_results(run_root: Path) -> None:
    """Save the original ZIP layout and checksum reports for local retrieval."""
    artifacts = config.artifacts_dir(run_root)
    assert (artifacts / "heldout_metrics.json").exists()
    shutil.copytree(
        config.metadata_dir(run_root),
        artifacts / "dataset_metadata",
        dirs_exist_ok=True,
    )
    checksums = {
        str(path.relative_to(artifacts)): sha256_file(path)
        for path in artifacts.rglob("*")
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    write_json(artifacts / "artifact_checksums.json", checksums)
    bundle = run_root / "stage1_results.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED, compresslevel=4) as archive:
        for path in artifacts.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(run_root)))
    report_path = run_root / "results_bundle.json"
    write_json(
        report_path,
        {
            "path": str(bundle),
            "bytes": bundle.stat().st_size,
            "sha256": sha256_file(bundle),
        },
    )
    print(report_path.read_text())
