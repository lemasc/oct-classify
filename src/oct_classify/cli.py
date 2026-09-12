from __future__ import annotations

import argparse
import json
from pathlib import Path

from oct_classify.data.audit import (
    audit_cross_source_records,
    audit_records,
    load_perceptual_hash_cache,
    write_perceptual_hash_cache,
)
from oct_classify.data.config import load_dataset_specs
from oct_classify.data.manifest import write_jsonl
from oct_classify.data.sources import get_source
from oct_classify.data.splits import validate_supplied_splits


def _records_for_spec(spec_path: Path):
    for spec in load_dataset_specs(spec_path):
        if not spec.enabled:
            continue
        if not spec.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {spec.root}")
        yield spec, list(get_source(spec.source).records(spec))


def _audit(args: argparse.Namespace) -> None:
    reports: list[dict[str, object]] = []
    failures: list[str] = []
    record_sets = list(_records_for_spec(args.config))
    computed_perceptual_hashes = {}
    perceptual_hash_cache = (
        load_perceptual_hash_cache(args.perceptual_hash_cache)
        if not args.no_perceptual_hashes
        else None
    )
    hash_timing_log = None
    if args.hash_timing_log is not None:
        args.hash_timing_log.parent.mkdir(parents=True, exist_ok=True)
        hash_timing_log = args.hash_timing_log.open("w", encoding="utf-8", buffering=1)
        hash_timing_log.write("source\tpath\tphash_seconds\n")
    try:
        for spec, records in record_sets:
            audit_report = audit_records(
                spec.root,
                records,
                perceptual_hashes=not args.no_perceptual_hashes,
                max_hash_distance=args.max_hash_distance,
                hash_timing_log=hash_timing_log,
                computed_perceptual_hashes=computed_perceptual_hashes,
                perceptual_hash_cache=perceptual_hash_cache,
            )
            report = audit_report.to_dict()
            report["dataset"] = spec.name
            output_path = args.output / f"{spec.name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            reports.append(report)
            failures.extend(
                f"{spec.name}: {failure}" for failure in audit_report.integrity_failures
            )
            print(
                json.dumps(
                    {
                        "dataset": spec.name,
                        "image_count": audit_report.image_count,
                        "invalid_images": len(audit_report.invalid_images),
                        "unmanifested_images": len(audit_report.unmanifested_images),
                        "exact_duplicate_clusters": len(audit_report.exact_duplicates),
                        "near_duplicate_clusters": len(audit_report.near_duplicates),
                    },
                    sort_keys=True,
                )
            )
        cross_source_report = audit_cross_source_records(
            ((spec.root, records) for spec, records in record_sets),
            perceptual_hashes=not args.no_perceptual_hashes,
            max_hash_distance=args.max_hash_distance,
            hash_timing_log=hash_timing_log,
            computed_perceptual_hashes=computed_perceptual_hashes,
            perceptual_hash_cache=perceptual_hash_cache,
        )
        cross_source = cross_source_report.to_dict()
        cross_source_path = args.output / "cross-source.json"
        cross_source_path.write_text(
            json.dumps(cross_source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "dataset": "cross-source",
                    "image_count": cross_source_report.image_count,
                    "invalid_images": len(cross_source_report.invalid_images),
                    "exact_duplicate_clusters": len(cross_source_report.exact_duplicates),
                    "near_duplicate_clusters": len(cross_source_report.near_duplicates),
                },
                sort_keys=True,
            )
        )
    finally:
        if hash_timing_log is not None:
            hash_timing_log.close()

    if perceptual_hash_cache is not None:
        write_perceptual_hash_cache(args.perceptual_hash_cache, perceptual_hash_cache)

    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cross_source": cross_source,
                "datasets": reports,
                "integrity_failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote audit reports to {args.output}")
    if failures:
        raise SystemExit("Audit integrity failures: " + "; ".join(failures))


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
        if command == "audit":
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/audits"))
            command_parser.add_argument("--no-perceptual-hashes", action="store_true")
            command_parser.add_argument("--max-hash-distance", type=int, default=5)
            command_parser.add_argument("--hash-timing-log", type=Path)
            command_parser.add_argument(
                "--perceptual-hash-cache",
                type=Path,
                default=Path("artifacts/audits/phash-cache.json"),
            )
        if command == "manifest":
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/manifests"))
        command_parser.set_defaults(handler=handler)

    args = parser.parse_args()
    args.handler(args)
