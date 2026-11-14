"""OBB data, source inventory, review and result paths."""

from pathlib import Path
from ..config import REPO
from ..utils.io import read_json

CONFIG_PATH = REPO / "configs/stage1_obb.json"
CONFIG = read_json(CONFIG_PATH)


def default_root():
    return REPO / "outputs/stage1_obb"


def dataset_dir(root):
    if Path(root).resolve() == default_root().resolve():
        return REPO / "dataset/stageOneOBB"
    return Path(root) / "dataset/stageOneOBB"


def metadata_dir(root):
    return dataset_dir(root) / "metadata"


def write_yaml(root):
    import yaml

    data = dataset_dir(root)
    path = data / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(data.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "register"},
            },
            sort_keys=False,
        )
    )
    return path


def annotation_dir(root):
    if Path(root).resolve() == default_root().resolve():
        return REPO / "artifacts/annotation"
    return Path(root) / "artifacts/annotation"


def source_dir(root):
    if Path(root).resolve() == default_root().resolve():
        return REPO / "dataset/wordWheelSource"
    return Path(root) / "dataset/wordWheelSource"


def source_archive_path(root):
    return source_dir(root) / "Word-Wheel_Water_Meter_Dataset.zip"
