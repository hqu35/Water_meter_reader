"""Download the frozen Dryad archive and enforce its published checksum."""

from pathlib import Path

from .. import config
from ..utils.io import sha256_file, write_json


def verify_source_archive(run_root: Path) -> Path:
    """Fail before consuming any archive whose bytes differ from the source."""
    archive = config.source_archive_path(run_root)
    assert sha256_file(archive) == config.CONFIG["dataset_sha256"], (
        "Archive checksum mismatch"
    )
    return archive


def download_dataset(run_root: Path) -> None:
    """Reuse an existing archive or stream the original public Dryad endpoint."""
    import requests

    archive = config.source_archive_path(run_root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    settings = config.CONFIG
    if not archive.exists() or archive.stat().st_size != settings["dataset_bytes"]:
        partial = archive.with_suffix(".partial")
        url = "https://datadryad.org/downloads/file_stream/" + str(
            settings["dataset_file_id"]
        )
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as destination:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    destination.write(chunk)
        partial.replace(archive)

    verify_source_archive(run_root)
    write_json(
        config.metadata_dir(run_root) / "download.json",
        {
            "doi": settings["dataset_doi"],
            "version": settings["dataset_version_id"],
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "checksum_verified": True,
        },
    )
