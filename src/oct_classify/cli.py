from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from oct_classify.data.audit import (
    audit_cross_source_records,
    audit_records,
)
from oct_classify.data.config import load_dataset_specs
from oct_classify.data.manifest import apply_processing_decisions, read_jsonl, write_jsonl
from oct_classify.data.preprocessing import PreprocessingSpec
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
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.models import build_model
from oct_classify.training.config import load_training_config
from oct_classify.training.dataset import (
    ManifestImageDataset,
    calculate_normalization,
    labels_for_available,
    load_split_records,
)
from oct_classify.training.engine import (
    checkpoint_state,
    load_checkpoint,
    run_epoch,
    seed_everything,
)
from oct_classify.training.outputs import create_run_directory, write_json, write_predictions


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
            expected_groups = {record.group_key for record in records}
            if set(assignments) != expected_groups:
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


def _spec_for_source(config_path: Path, source: str):
    for spec in load_dataset_specs(config_path):
        if spec.name == source and spec.enabled:
            return spec
    raise ValueError(f"Unknown or disabled dataset source: {source}")


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _loader(
    dataset: ManifestImageDataset, batch_size: int, workers: int, *, shuffle: bool, seed: int
) -> DataLoader[tuple[torch.Tensor, int, str]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _checkpoint_metadata(
    source: str,
    architecture: str,
    class_labels: tuple[UnifiedLabel, ...],
    mean: object,
    stdev: object,
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "source": source,
        "architecture": architecture,
        "class_labels": [label.value for label in class_labels],
        "normalization": {"mean": mean, "stdev": stdev},
        "config": config,
    }


def _log_tensorboard_metrics(
    writer: SummaryWriter,
    split: str,
    loss: float,
    metrics: dict[str, float | list[list[int]] | None],
    step: int,
) -> None:
    writer.add_scalar(f"{split}/loss", loss, step)
    for name, value in metrics.items():
        if isinstance(value, float):
            writer.add_scalar(f"{split}/{name}", value, step)


def _train(args: argparse.Namespace) -> None:
    config = load_training_config(args.training_config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config = replace(config, optimization=replace(config.optimization, epochs=args.epochs))
    spec = _spec_for_source(args.config, args.source)
    manifest_path = args.split_dir / f"{spec.name}.jsonl"
    class_labels = labels_for_available(spec.available_labels)
    preprocessing = PreprocessingSpec(image_size=config.data.image_size)
    train_records = load_split_records(manifest_path, "train")
    val_records = load_split_records(manifest_path, "val")
    test_records = load_split_records(manifest_path, "test")
    seed_everything(config.run.seed)
    mean, stdev = calculate_normalization(spec.root, train_records, preprocessing)
    train_dataset = ManifestImageDataset(
        spec.root,
        train_records,
        class_labels,
        preprocessing,
        mean,
        stdev,
        config.augmentation,
        training=True,
    )
    val_dataset = ManifestImageDataset(
        spec.root,
        val_records,
        class_labels,
        preprocessing,
        mean,
        stdev,
        config.augmentation,
        training=False,
    )
    test_dataset = ManifestImageDataset(
        spec.root,
        test_records,
        class_labels,
        preprocessing,
        mean,
        stdev,
        config.augmentation,
        training=False,
    )
    train_loader = _loader(
        train_dataset,
        config.data.batch_size,
        config.data.num_workers,
        shuffle=True,
        seed=config.run.seed,
    )
    val_loader = _loader(
        val_dataset,
        config.data.batch_size,
        config.data.num_workers,
        shuffle=False,
        seed=config.run.seed,
    )
    test_loader = _loader(
        test_dataset,
        config.data.batch_size,
        config.data.num_workers,
        shuffle=False,
        seed=config.run.seed,
    )
    device = _device(args.device)
    model = build_model(
        config.model.architecture, len(class_labels), pretrained=config.model.pretrained
    ).to(device)
    counts = Counter(train_dataset.targets)
    class_weights = torch.tensor(
        [
            len(train_dataset) / (len(class_labels) * counts[index])
            for index in range(len(class_labels))
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    metadata = _checkpoint_metadata(
        spec.name,
        config.model.architecture,
        class_labels,
        mean.tolist(),
        stdev.tolist(),
        config.to_dict(),
    )
    start_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(str(args.resume), model, optimizer)
        checkpoint_metadata = checkpoint["metadata"]
        if (
            checkpoint_metadata["source"] != spec.name
            or checkpoint_metadata["class_labels"] != metadata["class_labels"]
        ):
            raise ValueError(
                "Resume checkpoint does not match the selected source and class labels."
            )
        run_directory = args.resume.parent
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    else:
        run_directory = create_run_directory(
            args.output, spec.name, config.model.architecture, args.run_name
        )
        write_json(run_directory / "config.json", config.to_dict())
        write_json(
            run_directory / "dataset.json",
            {
                "source": spec.name,
                "split_manifest": str(manifest_path),
                "class_labels": metadata["class_labels"],
                "records": {
                    "train": len(train_dataset),
                    "val": len(val_dataset),
                    "test": len(test_dataset),
                },
            },
        )
        write_json(run_directory / "normalization.json", metadata["normalization"])
    history_path = run_directory / "history.jsonl"
    writer = SummaryWriter(log_dir=str(run_directory / "events"))
    last_epoch = start_epoch - 1
    with history_path.open("a", encoding="utf-8") as history:
        for epoch in range(start_epoch, config.optimization.epochs):
            train_result = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                tuple(label.value for label in class_labels),
                optimizer=optimizer,
                max_batches=args.max_train_batches,
            )
            with torch.no_grad():
                val_result = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    tuple(label.value for label in class_labels),
                    max_batches=args.max_eval_batches,
                )
            epoch_result = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_metrics": train_result.evaluation.metrics,
                "val_loss": val_result.loss,
                "val_metrics": val_result.evaluation.metrics,
            }
            history.write(json.dumps(epoch_result, allow_nan=False) + "\n")
            history.flush()
            _log_tensorboard_metrics(
                writer, "train", train_result.loss, train_result.evaluation.metrics, epoch
            )
            _log_tensorboard_metrics(
                writer, "validation", val_result.loss, val_result.evaluation.metrics, epoch
            )
            writer.flush()
            last_epoch = epoch
            macro_f1 = float(val_result.evaluation.metrics["macro_f1"])
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                epochs_without_improvement = 0
                torch.save(
                    checkpoint_state(
                        model,
                        optimizer,
                        epoch,
                        best_macro_f1,
                        epochs_without_improvement,
                        metadata,
                    ),
                    run_directory / "checkpoint-best.pt",
                )
            else:
                epochs_without_improvement += 1
            torch.save(
                checkpoint_state(
                    model,
                    optimizer,
                    epoch,
                    best_macro_f1,
                    epochs_without_improvement,
                    metadata,
                ),
                run_directory / "checkpoint-last.pt",
            )
            print(json.dumps(epoch_result, sort_keys=True))
            if epochs_without_improvement >= config.optimization.early_stopping_patience:
                print(
                    f"Early stopping after {epoch + 1} epochs: validation macro-F1 did not improve "
                    f"for {epochs_without_improvement} epochs."
                )
                break
    load_checkpoint(str(run_directory / "checkpoint-best.pt"), model)
    with torch.no_grad():
        test_result = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            tuple(label.value for label in class_labels),
            max_batches=args.max_eval_batches,
        )
    write_json(
        run_directory / "metrics-test.json",
        {"loss": test_result.loss, **test_result.evaluation.metrics},
    )
    _log_tensorboard_metrics(
        writer,
        "test",
        test_result.loss,
        test_result.evaluation.metrics,
        last_epoch + 1,
    )
    writer.close()
    write_predictions(
        run_directory / "predictions-test.csv",
        test_result.paths,
        test_result.evaluation.targets,
        test_result.evaluation.predictions,
        test_result.evaluation.probabilities,
        tuple(label.value for label in class_labels),
    )
    print(f"Wrote run artifacts to {run_directory}")


class _RestrictedClassifier(nn.Module):
    def __init__(self, model: nn.Module, indices: list[int]) -> None:
        super().__init__()
        self.model = model
        self.indices = indices

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)[:, self.indices]


def _evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    source_spec = _spec_for_source(args.config, args.source)
    model_labels = tuple(UnifiedLabel(label) for label in metadata["class_labels"])
    class_labels = tuple(label for label in model_labels if label in source_spec.available_labels)
    if len(class_labels) < 2:
        raise ValueError("Model and target source must share at least two labels.")
    config = metadata["config"]
    preprocessing = PreprocessingSpec(image_size=config["data"]["image_size"])
    normalization = metadata["normalization"]
    records = load_split_records(args.split_dir / f"{source_spec.name}.jsonl", "test")
    dataset = ManifestImageDataset(
        source_spec.root,
        records,
        class_labels,
        preprocessing,
        normalization["mean"],
        normalization["stdev"],
        training=False,
    )
    loader = _loader(
        dataset,
        config["data"]["batch_size"],
        config["data"]["num_workers"],
        shuffle=False,
        seed=config["run"]["seed"],
    )
    device = _device(args.device)
    model = build_model(metadata["architecture"], len(model_labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    indices = [model_labels.index(label) for label in class_labels]
    restricted_model = _RestrictedClassifier(model, indices).to(device)
    with torch.no_grad():
        result = run_epoch(
            restricted_model,
            loader,
            nn.CrossEntropyLoss(),
            device,
            tuple(label.value for label in class_labels),
            max_batches=args.max_eval_batches,
        )
    output = args.output or args.checkpoint.parent / "evaluations" / source_spec.name
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "metrics.json",
        {
            "checkpoint": str(args.checkpoint),
            "target_source": source_spec.name,
            "class_labels": [label.value for label in class_labels],
            "loss": result.loss,
            **result.evaluation.metrics,
        },
    )
    write_predictions(
        output / "predictions.csv",
        result.paths,
        result.evaluation.targets,
        result.evaluation.predictions,
        result.evaluation.probabilities,
        tuple(label.value for label in class_labels),
    )
    print(f"Wrote evaluation artifacts to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCT dataset preparation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler, help_text in (
        ("audit", _audit, "Inspect all configured dataset images."),
        ("manifest", _manifest, "Write canonical JSONL manifests."),
        ("validate-splits", _validate_splits, "Check supplied splits for group leakage."),
        ("splits", _splits, "Create reproducible group-safe derived splits."),
        ("train", _train, "Train one independently fine-tuned supervised model."),
        ("evaluate", _evaluate, "Evaluate a checkpoint on one source test split."),
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
        if command == "train":
            command_parser.add_argument("--source", required=True)
            command_parser.add_argument(
                "--training-config", type=Path, default=Path("configs/training/resnet50.toml")
            )
            command_parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
            command_parser.add_argument("--output", type=Path, default=Path("artifacts/runs"))
            command_parser.add_argument("--run-name")
            command_parser.add_argument("--resume", type=Path)
            command_parser.add_argument("--device", default="cuda")
            command_parser.add_argument(
                "--epochs", type=int, help="Override configured epoch count."
            )
            command_parser.add_argument(
                "--max-train-batches",
                type=int,
                help="Cap each training epoch for a local smoke test.",
            )
            command_parser.add_argument(
                "--max-eval-batches",
                type=int,
                help="Cap validation and test batches for a local smoke test.",
            )
        if command == "evaluate":
            command_parser.add_argument("--checkpoint", type=Path, required=True)
            command_parser.add_argument("--source", required=True)
            command_parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
            command_parser.add_argument("--output", type=Path)
            command_parser.add_argument("--device", default="cuda")
            command_parser.add_argument(
                "--max-eval-batches", type=int, help="Cap test batches for a local smoke test."
            )
        command_parser.set_defaults(handler=handler)

    args = parser.parse_args()
    args.handler(args)
