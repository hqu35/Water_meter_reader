"""Command parsing for the OBB-only Stage I worker."""

import argparse
from pathlib import Path
from . import config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stage1_python_worker")
    parser.add_argument("--root", type=Path, default=config.default_root())
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", type=Path)
    prepare.add_argument("--selection", type=Path)
    commands.add_parser("validate")
    commands.add_parser("download")
    commands.add_parser("inspect")
    approve = commands.add_parser("approve")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--confirm-reviewed", action="store_true", required=True)
    commands.add_parser("train")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--weights", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--confidence", type=float, default=0.25)
    commands.add_parser("package")
    commands.add_parser("gpu")
    remote = commands.add_parser("colab")
    from .colab import COMMANDS

    remote.add_argument("phase", choices=COMMANDS)
    remote.add_argument("--background", action="store_true")
    pilot = commands.add_parser(
        "orientation-pilot", help="Independent 100-sample pretrained PaddleOCR pilot"
    )
    pilot.add_argument("--crop-source", choices=["ground-truth"], required=True)
    pilot.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "orientation-pilot":
        from .orientation_pilot import run

        run(args.root, args.output, args.crop_source)
    elif args.command == "colab":
        from .colab import dispatch

        dispatch(args.phase, args.root, args.background)
    elif args.command == "package":
        from .colab import package

        package(args.root)
    elif args.command == "gpu":
        from ..training.trainer import verify_gpu

        verify_gpu(args.root)
    elif args.command == "download":
        from ..data.download import download_dataset

        download_dataset(args.root)
    elif args.command == "inspect":
        from ..data.archive import inspect_dataset

        inspect_dataset(args.root)
    elif args.command == "prepare":
        from .data import prepare

        prepare(args.root, args.archive, args.selection)
    elif args.command == "validate":
        from .data import validate_dataset

        manifest = validate_dataset(args.root)
        print("OBB integrity passed:", len(manifest["records"]), "real images")
    elif args.command == "approve":
        from .review import approve

        approve(args.root, args.reviewer, args.confirm_reviewed)
    elif args.command == "train":
        from .training import train

        train(args.root)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(args.root, args.weights)
    elif args.command == "predict":
        from .inference import predict

        predict(args.weights, args.image, args.output, args.device, args.confidence)
