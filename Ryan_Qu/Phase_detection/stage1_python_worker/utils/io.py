"""JSON serialization and streaming file checksums."""

import hashlib
import json
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile

PathLike = Union[str, Path]


def sha256_file(path: PathLike) -> str:
    """Hash a file without loading a dataset archive into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: PathLike) -> Any:
    """Read an existing JSON document."""
    return json.loads(Path(path).read_text())


def write_json(path: PathLike, value: Any, *, sort_keys: bool = True) -> None:
    """Keep the pipeline's original indented, sorted JSON artifact format."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")


def extract_zip_safely(archive: ZipFile, destination: Path) -> None:
    """Reject paths escaping the destination before extracting any ZIP member."""
    for name in archive.namelist():
        assert (destination / name).resolve().is_relative_to(destination.resolve()), (
            "Unsafe archive member"
        )
    archive.extractall(destination)
