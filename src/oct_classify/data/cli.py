"""Dataset preparation commands that deliberately do not import torch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from oct_classify.data.audit import audit_cross_source_records, audit_records
from oct_classify.data.config import load_dataset_specs
from oct_classify.data.manifest import apply_processing_decisions, read_jsonl, write_jsonl
from oct_classify.data.sources import get_source
from oct_classify.data.splits import (
    DerivedSplit,
    apply_derived_splits,
    create_derived_splits,
    file_sha256,
    validate_derived_splits,
    validate_supplied_splits,
    write_split_definition,
)


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
    computed_images = {}
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
                computed_images=computed_images,
                workers=args.workers,
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
            computed_images=computed_images,
            workers=args.workers,
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

    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"cross_source": cross_source, "datasets": reports, "integrity_failures": failures},
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
        audit_report = audit_records(spec.root, records, workers=args.workers)
        records, quarantined, deduplicated = apply_processing_decisions(
            records,
            exact_duplicates=audit_report.exact_duplicates,
            near_duplicates=audit_report.near_duplicates,
        )
        path = args.output / f"{spec.name}.jsonl"
        write_jsonl(path, records)
        print(
            f"Wrote {len(records)} records to {path} "
            f"({quarantined} quarantined, {deduplicated} same-label exact duplicates removed)"
        )


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


def _splits(args: argparse.Namespace) -> None:
    all_assignments: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    audit_hashes: dict[str, str] = {}
    source_settings: dict[str, dict[str, object]] = {}
    reports: dict[str, object] = {}
    existing_definition = None
    if args.definition.is_file() and not args.replace_definition:
        existing_definition = json.loads(args.definition.read_text(encoding="utf-8"))
        if existing_definition.get("version") != 1:
            raise ValueError(
                f"Unsupported split definition version: {existing_definition.get('version')}"
            )
    for spec in load_dataset_specs(args.config):
        if not spec.enabled:
            continue
        manifest_path = args.manifest_dir / f"{spec.name}.jsonl"
        audit_path = args.audit_dir / f"{spec.name}.json"
        if not manifest_path.is_file() or not audit_path.is_file():
            raise FileNotFoundError(
                f"Split generation requires {manifest_path} and {audit_path}; run manifest and audit first."
            )
        records = list(read_jsonl(manifest_path))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        preserve_supplied_test = spec.name == "kermany"
        ratios = (
            {"train": 0.85, "val": 0.15}
            if preserve_supplied_test
            else {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            }
        )
        manifest_hash = file_sha256(manifest_path)
        audit_hash = file_sha256(audit_path)
        if existing_definition is not None:
            if existing_definition["manifest_sha256"].get(spec.name) != manifest_hash:
                raise ValueError(
                    f"Manifest hash changed for {spec.name}; create a new split definition."
                )
            if existing_definition["audit_sha256"].get(spec.name) != audit_hash:
                raise ValueError(
                    f"Audit hash changed for {spec.name}; create a new split definition."
                )
            assignments = {
                group: split
                for group, split in existing_definition["assignments"].items()
                if group.startswith(f"{spec.name}:")
            }
            if set(assignments) != {record.group_key for record in records}:
                raise ValueError(f"Split definition groups do not match the {spec.name} manifest.")
            generated = create_derived_splits(records, audit["near_duplicates"], ratios, args.seed)
            derived = DerivedSplit(assignments, generated.linked_group_components)
        else:
            derived = create_derived_splits(
                records,
                audit["near_duplicates"],
                ratios,
                args.seed,
                preserve_supplied_test=preserve_supplied_test,
            )
        split_records = apply_derived_splits(records, derived.assignments)
        failures = validate_derived_splits(split_records, derived.linked_group_components)
        if preserve_supplied_test:
            failures.extend(
                "Kermany supplied test group is not assigned to derived test"
                if record.supplied_split == "test" and record.split != "test"
                else "Kermany supplied training group is not assigned to train or val"
                for record in split_records
                if (record.supplied_split == "test" and record.split != "test")
                or (record.supplied_split == "train" and record.split not in {"train", "val"})
            )
        if failures:
            raise ValueError("Derived split validation failed: " + "; ".join(failures))
        write_jsonl(args.output / f"{spec.name}.jsonl", split_records)
        source_assignments = {
            group: split
            for group, split in derived.assignments.items()
            if group.startswith(f"{spec.name}:")
        }
        all_assignments.update(source_assignments)
        manifest_hashes[spec.name] = manifest_hash
        audit_hashes[spec.name] = audit_hash
        source_settings[spec.name] = {
            "ratios": ratios,
            "preserve_supplied_test": preserve_supplied_test,
        }
        counts: dict[str, dict[str, int]] = {}
        for record in split_records:
            counts.setdefault(record.split or "unassigned", {}).setdefault(record.label.value, 0)
            counts[record.split or "unassigned"][record.label.value] += 1
        reports[spec.name] = {
            "records_by_split_and_label": counts,
            "group_count": len(source_assignments),
            "linked_group_components": sum(
                len(component) > 1 for component in derived.linked_group_components
            ),
        }
    if existing_definition is None:
        write_split_definition(
            args.definition,
            seed=args.seed,
            source_settings=source_settings,
            assignments=all_assignments,
            manifest_hashes=manifest_hashes,
            audit_hashes=audit_hashes,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "summary.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    action = "Reused" if existing_definition is not None else "Wrote"
    print(f"{action} split definition at {args.definition} and wrote manifests to {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCT dataset preparation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler, help_text in (
        ("audit", _audit, "Inspect all configured dataset images."),
        ("manifest", _manifest, "Write canonical JSONL manifests."),
        ("validate-splits", _validate_splits, "Check supplied splits for group leakage."),
        ("splits", _splits, "Create reproducible group-safe derived splits."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
        if command == "audit":
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/audits"))
            command_parser.add_argument("--no-perceptual-hashes", action="store_true")
            command_parser.add_argument("--max-hash-distance", type=int, default=5)
            command_parser.add_argument("--hash-timing-log", type=Path)
            command_parser.add_argument(
                "--workers",
                type=int,
                help="Image-analysis processes; defaults to all available CPU cores.",
            )
        if command == "manifest":
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/manifests"))
            command_parser.add_argument(
                "--workers",
                type=int,
                help="Image-analysis processes; defaults to all available CPU cores.",
            )
        if command == "splits":
            command_parser.add_argument(
                "--manifest-dir", type=Path, default=Path("artifacts/manifests")
            )
            command_parser.add_argument("--audit-dir", type=Path, default=Path("artifacts/audits"))
            command_parser.add_argument(
                "--definition", type=Path, default=Path("configs/splits/v1.json")
            )
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/splits"))
            command_parser.add_argument("--seed", type=int, default=20260913)
            command_parser.add_argument(
                "--replace-definition",
                action="store_true",
                help="Replace an existing tracked split definition after intentionally changing inputs.",
            )
        command_parser.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.handler(args)
