"""Experiment configuration and the shared local/Colab path convention."""

from pathlib import Path

from .utils.io import read_json, sha256_file

PACKAGE_NAME = Path(__file__).resolve().parent.name
REPO = Path(__file__).resolve().parents[1]
CONFIG = read_json(REPO / "configs/stage1.json")
DATASET_RELATIVE_PATH = Path("dataset/stageOne")


def default_run_root() -> Path:
    """Return the local experiment-output directory, separate from data."""
    return REPO / "outputs/stage1"


def dataset_dir(run_root: Path) -> Path:
    """Use repo-level data locally and run-local data for a custom/Colab root."""
    run_root = Path(run_root)
    if run_root.resolve() == default_run_root().resolve():
        return REPO / DATASET_RELATIVE_PATH
    return run_root / DATASET_RELATIVE_PATH


def metadata_dir(run_root: Path) -> Path:
    return dataset_dir(run_root) / "metadata"


def artifacts_dir(run_root: Path) -> Path:
    return run_root / "artifacts"


def source_archive_path(run_root: Path) -> Path:
    return dataset_dir(run_root) / "raw/Word-Wheel_Water_Meter_Dataset.zip"


def write_dataset_yaml(run_root: Path) -> Path:
    """Refresh the machine-specific YOLO root; split paths stay relative."""
    import yaml

    target = dataset_dir(run_root)
    destination = target / "data.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    definition = {
        "path": str(target.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "register"},
    }
    destination.write_text(yaml.safe_dump(definition, sort_keys=False))
    return destination


def source_files() -> list:
    """List everything needed to run the modular pipeline on Colab."""
    return [
        REPO / "configs/stage1.json",
        REPO / "requirements-stage1.txt",
        *sorted((REPO / PACKAGE_NAME).rglob("*.py")),
    ]


def source_checksums() -> dict:
    """Track the entry point, config and all executable package modules."""
    return {str(path.relative_to(REPO)): sha256_file(path) for path in source_files()}
