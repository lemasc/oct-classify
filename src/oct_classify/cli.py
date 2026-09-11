from __future__ import annotations

import argparse
import json
from pathlib import Path

from oct_classify.data.audit import audit_records
from oct_classify.data.config import load_dataset_specs
from oct_classify.data.manifest import write_jsonl
from oct_classify.data.sources import get_source
from oct_classify.data.splits import validate_supplied_splits


def _records_for_spec(spec_path: Path):
    for spec in load_dataset_specs(spec_path):
        if not spec.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {spec.root}")
        yield spec, list(get_source(spec.source).records(spec))


def _audit(args: argparse.Namespace) -> None:
    for spec, records in _records_for_spec(args.config):
        report = audit_records(spec.root, records).to_dict()
        report["dataset"] = spec.name
        print(json.dumps(report, indent=2, sort_keys=True))


def _manifest(args: argparse.Namespace) -> None:
    for spec, records in _records_for_spec(args.config):
        path = args.output / f"{spec.name}.jsonl"
        write_jsonl(path, records)
        print(f"Wrote {len(records)} records to {path}")


def _validate_splits(args: argparse.Namespace) -> None:
    for spec, records in _records_for_spec(args.config):
        report = validate_supplied_splits(records)
        print(
            json.dumps(
                {
                    "dataset": spec.name,
                    "duplicate_paths": report.duplicate_paths,
                    "groups_in_multiple_splits": report.groups_in_multiple_splits,
                    "records_without_group": report.records_without_group,
                    "is_leakage_safe": report.is_leakage_safe,
                },
                indent=2,
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OCT dataset preparation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler, help_text in (
        ("audit", _audit, "Inspect all configured dataset images."),
        ("manifest", _manifest, "Write canonical JSONL manifests."),
        ("validate-splits", _validate_splits, "Check supplied splits for group leakage."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
        if command == "manifest":
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/manifests"))
        command_parser.set_defaults(handler=handler)

    args = parser.parse_args()
    args.handler(args)
