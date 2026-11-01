"""The single OBB-only Stage I entry point."""

import sys
from .obb.cli import main as obb_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "obb":
        arguments = arguments[1:]
    obb_main(arguments)
