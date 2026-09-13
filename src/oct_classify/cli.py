"""Torch-free command-family dispatcher."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCT classification utilities")
    groups = parser.add_subparsers(dest="group", required=True)
    groups.add_parser("data", add_help=False, help="Audit, manifest, and split datasets.")
    groups.add_parser("train", add_help=False, help="Train and evaluate models.")
    args, remaining = parser.parse_known_args(argv)

    if args.group == "data":
        from oct_classify.data.cli import main as data_main

        data_main(remaining)
    else:
        from oct_classify.training.cli import main as training_main

        training_main(remaining)
