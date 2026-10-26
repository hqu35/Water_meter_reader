"""The repository's single command interface for local and Colab execution."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from . import config
from .config import default_run_root
from .data.archive import inspect_dataset
from .data.download import download_dataset
from .data.preparation import prepare_dataset
from .data.review import approve_annotations
from .evaluation.evaluator import evaluate_model
from .training.checkpoints import normalize_checkpoints
from .training.trainer import train_model, verify_gpu
from .utils.packaging import package_results

PHASES = {
    "gpu": verify_gpu,
    "download": download_dataset,
    "inspect": inspect_dataset,
    "prepare": prepare_dataset,
    "approve": approve_annotations,
    "train": train_model,
    "normalize-checkpoints": normalize_checkpoints,
    "evaluate": evaluate_model,
    "package": package_results,
}


def build_parser() -> argparse.ArgumentParser:
    """List worker phases, single-image prediction and remote transport together."""
    from .colab import TRANSPORT_COMMANDS

    parser = argparse.ArgumentParser(
        prog=f"python -m {config.PACKAGE_NAME}",
        description="CS539 Stage I register localization",
    )
    parser.add_argument("--root", type=Path, default=default_run_root())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in PHASES:
        phase_parser = commands.add_parser(name)
        if name == "approve":
            phase_parser.add_argument("--reviewer")

    prediction = commands.add_parser(
        "predict", help="Locate and crop one real register"
    )
    prediction.add_argument("--weights", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument(
        "--output", type=Path, default=Path("outputs/stage1/prediction")
    )
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--confidence", type=float, default=0.25)

    remote = commands.add_parser(
        "colab", help="Control the existing named Colab session"
    )
    remote.add_argument("phase", choices=[*TRANSPORT_COMMANDS, *PHASES])
    remote.add_argument("--reviewer")
    remote.add_argument("--background", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch each operation once; there are no parallel script implementations."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "predict":
        from .inference import predict_register

        predict_register(
            arguments.weights,
            arguments.image,
            arguments.output,
            arguments.device,
            arguments.confidence,
        )
        return

    arguments.root.mkdir(parents=True, exist_ok=True)
    if arguments.command == "colab":
        from .colab import dispatch

        if arguments.phase == "approve" and not arguments.reviewer:
            parser.error(
                "--reviewer is required after visual inspection of all review sheets"
            )
        dispatch(
            arguments.phase, arguments.root, arguments.reviewer, arguments.background
        )
    elif arguments.command == "approve":
        assert arguments.reviewer, (
            "Explicit reviewer and completed visual inspection required"
        )
        approve_annotations(arguments.root, arguments.reviewer)
    else:
        PHASES[arguments.command](arguments.root)
