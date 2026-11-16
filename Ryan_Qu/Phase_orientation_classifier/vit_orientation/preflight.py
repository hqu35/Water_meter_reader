"""Run one real-data MPS backward pass without starting a training run."""

import argparse
import json
import platform
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import load_dataset, verify_manifest
from .model import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    manifest = json.loads(a.manifest.read_text())
    verify_manifest(manifest)
    model = build(True).to("mps")
    model.train()
    ds = load_dataset(manifest, "train", model, training=True)
    x, y = next(iter(DataLoader(ds, batch_size=16, num_workers=0)))
    x, y = x.to("mps"), y.to("mps")
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    norm = torch.stack(
        [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
    ).norm()
    torch.mps.synchronize()
    if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():
        raise RuntimeError("Non-finite forward/backward result")
    val = load_dataset(manifest, "val", model)
    first = val[0][0]
    assert torch.equal(first, val[0][0])
    result = {
        "mps_available": True,
        "architecture": platform.machine(),
        "real_batch_size": len(y),
        "input_shape": list(x.shape),
        "logit_shape": list(logits.shape),
        "loss": loss.item(),
        "gradient_norm": norm.item(),
        "deterministic_validation_preprocessing": True,
        "optimizer_steps": 0,
        "training_started": False,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
