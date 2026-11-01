"""Repository identity and the source inventory uploaded to Colab."""

from pathlib import Path
from .utils.io import sha256_file

PACKAGE_NAME = Path(__file__).resolve().parent.name
REPO = Path(__file__).resolve().parents[1]


def artifacts_dir(root):
    return Path(root) / "artifacts"


def source_files():
    return [
        REPO / "configs/stage1_obb.json",
        REPO / "requirements-stage1.txt",
        REPO / "requirements-stage1-obb.txt",
        *sorted((REPO / PACKAGE_NAME).rglob("*.py")),
        *sorted((REPO / PACKAGE_NAME).rglob("*.html")),
    ]


def source_checksums():
    return {str(path.relative_to(REPO)): sha256_file(path) for path in source_files()}
