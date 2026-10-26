"""Normalize single-class display metadata without changing learned tensors."""

from pathlib import Path

from .. import config
from ..utils.io import read_json, sha256_file, write_json


def normalize_checkpoint(path: Path) -> dict:
    """Round-trip and compare model/EMA tensors before replacing one checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    original_states = {}
    for slot in ["model", "ema"]:
        model = checkpoint.get(slot)
        if model is None:
            continue
        assert len(model.names) == 1, "Unexpected class count"
        original_states[slot] = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        model.names = {0: "register"}

    before_sha256 = sha256_file(path)
    temporary_path = path.with_suffix(".metadata.tmp")
    torch.save(checkpoint, temporary_path)
    restored = torch.load(temporary_path, map_location="cpu", weights_only=False)
    for slot, state in original_states.items():
        assert restored[slot].names == {0: "register"}
        for name, tensor in state.items():
            assert torch.equal(tensor, restored[slot].state_dict()[name]), (
                "Tensor changed",
                path,
                name,
            )
    temporary_path.replace(path)
    return {
        "file": str(path),
        "before_sha256": before_sha256,
        "after_sha256": sha256_file(path),
        "all_model_tensors_identical": True,
        "names": {0: "register"},
    }


def normalize_checkpoints(run_root: Path) -> None:
    """Replace the framework's generic 'item' name, preserving optimizer/weights."""
    artifacts = config.artifacts_dir(run_root)
    completed = read_json(artifacts / "training_complete.json")
    weights_dir = Path(completed["save_dir"]) / "weights"
    reports = [normalize_checkpoint(path) for path in sorted(weights_dir.glob("*.pt"))]
    write_json(
        artifacts / "checkpoint_class_names.json",
        {
            "reason": "Normalize framework single_cls display name item to register; no retraining or tensor changes",
            "source_sha256": sha256_file(__file__),
            "checkpoints": reports,
        },
    )
    print("CHECKPOINT_CLASS_NAMES_VERIFIED", len(reports), flush=True)
