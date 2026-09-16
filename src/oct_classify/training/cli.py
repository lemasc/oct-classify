"""Torch-dependent supervised baseline and evaluation commands."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from oct_classify.data.config import load_dataset_specs
from oct_classify.data.preprocessing import PreprocessingSpec, calculate_normalization
from oct_classify.data.splits import file_sha256
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.models import build_model
from oct_classify.training.config import AugmentationConfig, load_training_config
from oct_classify.training.dataset import (
    CLASS_ORDER,
    ManifestImageDataset,
    labels_for_available,
    load_split_records,
)
from oct_classify.training.engine import (
    checkpoint_state,
    load_checkpoint,
    run_epoch,
    run_source_evaluations,
    seed_everything,
)
from oct_classify.training.evaluation import calculate_grouped_metrics, calculate_metrics
from oct_classify.training.outputs import create_run_directory, write_json, write_predictions
from oct_classify.training.partial_labels import MaskedCrossEntropyLoss
from oct_classify.training.sampling import SourceClassBalancedBatchSampler


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
    dataset: ManifestImageDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader[tuple[torch.Tensor, int, torch.Tensor, str, str]]:
    generator = torch.Generator().manual_seed(seed)
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
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


def _baseline(args: argparse.Namespace) -> None:
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
    test_records = [] if args.skip_test else load_split_records(manifest_path, "test")
    seed_everything(config.run.seed)
    mean, stdev = calculate_normalization(
        spec.root, train_records, preprocessing, workers=args.workers
    )
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
    test_dataset = (
        ManifestImageDataset(
            spec.root,
            test_records,
            class_labels,
            preprocessing,
            mean,
            stdev,
            config.augmentation,
            training=False,
        )
        if test_records
        else None
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
    test_loader = (
        _loader(
            test_dataset,
            config.data.batch_size,
            config.data.num_workers,
            shuffle=False,
            seed=config.run.seed,
        )
        if test_dataset is not None
        else None
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
    criterion = MaskedCrossEntropyLoss(class_weights)
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
                    "test": len(test_dataset) if test_dataset is not None else 0,
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
                        model, optimizer, epoch, best_macro_f1, epochs_without_improvement, metadata
                    ),
                    run_directory / "checkpoint-best.pt",
                )
            else:
                epochs_without_improvement += 1
            torch.save(
                checkpoint_state(
                    model, optimizer, epoch, best_macro_f1, epochs_without_improvement, metadata
                ),
                run_directory / "checkpoint-last.pt",
            )
            print(json.dumps(epoch_result, sort_keys=True))
            if epochs_without_improvement >= config.optimization.early_stopping_patience:
                print(
                    f"Early stopping after {epoch + 1} epochs: validation macro-F1 did not improve for {epochs_without_improvement} epochs."
                )
                break
    if test_loader is not None:
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
        test_metrics = {"loss": test_result.loss, **test_result.evaluation.metrics}
        if spec.name == "duke":
            group_by_path = {record.path: record.eye_id for record in test_records}
            if any(group_by_path[path] is None for path in test_result.paths):
                raise ValueError("Duke test records require an eye ID for volume-level evaluation.")
            eye_result = calculate_grouped_metrics(
                test_result.evaluation.targets,
                test_result.evaluation.probabilities,
                [group_by_path[path] for path in test_result.paths],  # type: ignore[list-item]
                tuple(label.value for label in class_labels),
            )
            test_metrics["eye_level"] = eye_result.metrics
        write_json(run_directory / "metrics-test.json", test_metrics)
        _log_tensorboard_metrics(
            writer, "test", test_result.loss, test_result.evaluation.metrics, last_epoch + 1
        )
        write_predictions(
            run_directory / "predictions-test.csv",
            test_result.paths,
            test_result.evaluation.targets,
            test_result.evaluation.predictions,
            test_result.evaluation.probabilities,
            tuple(label.value for label in class_labels),
        )
    writer.close()
    print(f"Wrote run artifacts to {run_directory}")


def _fused(args: argparse.Namespace) -> None:
    config = load_training_config(args.training_config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config = replace(config, optimization=replace(config.optimization, epochs=args.epochs))
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("--sources must not repeat a source.")
    specs = [_spec_for_source(args.config, source) for source in args.sources]
    if len(specs) < 2:
        raise ValueError("Fused training requires at least two sources.")
    if config.data.batch_size % len(specs):
        raise ValueError("data.batch_size must divide evenly across selected sources.")

    manifests = {spec.name: args.split_dir / f"{spec.name}.jsonl" for spec in specs}
    splits = ("train", "val") if args.skip_test else ("train", "val", "test")
    records_by_split = {
        split: {spec.name: load_split_records(manifests[spec.name], split) for spec in specs}
        for split in splits
    }
    roots = {spec.name: spec.root for spec in specs}
    class_labels = CLASS_ORDER
    preprocessing = PreprocessingSpec(image_size=config.data.image_size)
    train_records = [record for records in records_by_split["train"].values() for record in records]
    seed_everything(config.run.seed)
    mean, stdev = calculate_normalization(roots, train_records, preprocessing, workers=args.workers)
    train_dataset = ManifestImageDataset(
        roots,
        train_records,
        class_labels,
        preprocessing,
        mean,
        stdev,
        config.augmentation,
        training=True,
    )
    source_datasets = {
        split: {
            spec.name: ManifestImageDataset(
                roots,
                records_by_split[split][spec.name],
                class_labels,
                preprocessing,
                mean,
                stdev,
                config.augmentation,
                training=False,
            )
            for spec in specs
        }
        for split in splits
        if split != "train"
    }
    sampler = SourceClassBalancedBatchSampler(
        [record.source for record in train_dataset.records],
        train_dataset.targets,
        config.data.batch_size,
        config.sampling.batches_per_epoch,
        config.run.seed,
    )
    train_loader = _loader(
        train_dataset,
        config.data.batch_size,
        config.data.num_workers,
        shuffle=False,
        seed=config.run.seed,
        batch_sampler=sampler,
    )
    source_loaders = {
        split: {
            source: _loader(
                dataset,
                config.data.batch_size,
                config.data.num_workers,
                shuffle=False,
                seed=config.run.seed,
            )
            for source, dataset in datasets.items()
        }
        for split, datasets in source_datasets.items()
    }
    source_label_indices = {
        spec.name: tuple(index for index, label in enumerate(class_labels) if label in spec.available_labels)
        for spec in specs
    }
    device = _device(args.device)
    model = build_model(config.model.architecture, len(class_labels), pretrained=config.model.pretrained).to(
        device
    )
    criterion = MaskedCrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=config.optimization.learning_rate, weight_decay=config.optimization.weight_decay
    )
    metadata = _checkpoint_metadata(
        "fused",
        config.model.architecture,
        class_labels,
        mean.tolist(),
        stdev.tolist(),
        config.to_dict(),
    )
    metadata["sources"] = [spec.name for spec in specs]
    metadata["split_manifests"] = {source: str(path) for source, path in manifests.items()}
    metadata["split_manifest_sha256"] = {
        source: file_sha256(path) for source, path in manifests.items()
    }
    start_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(str(args.resume), model, optimizer)
        checkpoint_metadata = checkpoint["metadata"]
        if checkpoint_metadata.get("source") != "fused" or checkpoint_metadata.get("sources") != metadata["sources"]:
            raise ValueError("Resume checkpoint does not match the selected fused sources.")
        run_directory = args.resume.parent
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    else:
        run_directory = create_run_directory(args.output, "fused", config.model.architecture, args.run_name)
        write_json(run_directory / "config.json", config.to_dict())
        write_json(
            run_directory / "dataset.json",
            {
                "sources": metadata["sources"],
                "split_manifests": metadata["split_manifests"],
                "split_manifest_sha256": metadata["split_manifest_sha256"],
                "class_labels": metadata["class_labels"],
                "sampling": config.to_dict()["sampling"],
                "records": {
                    split: {source: len(dataset) for source, dataset in datasets.items()}
                    for split, datasets in source_datasets.items()
                }
                | {"train": {source: sum(record.source == source for record in train_dataset.records) for source in metadata["sources"]}},
            },
        )
        write_json(run_directory / "normalization.json", metadata["normalization"])
    history_path = run_directory / "history.jsonl"
    writer = SummaryWriter(log_dir=str(run_directory / "events"))
    with history_path.open("a", encoding="utf-8") as history:
        for epoch in range(start_epoch, config.optimization.epochs):
            train_result = run_epoch(
                model, train_loader, criterion, device, tuple(label.value for label in class_labels), optimizer=optimizer,
                max_batches=args.max_train_batches,
            )
            with torch.no_grad():
                validation_results, validation_score = run_source_evaluations(
                    model, source_loaders["val"], criterion, device,
                    tuple(label.value for label in class_labels), source_label_indices,
                    max_batches=args.max_eval_batches,
                )
            validation_metrics = {
                source: {"loss": result.loss, **result.evaluation.metrics}
                for source, result in validation_results.items()
            }
            epoch_result = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_metrics": train_result.evaluation.metrics,
                "validation_mean_macro_f1": validation_score,
                "validation_by_source": validation_metrics,
            }
            history.write(json.dumps(epoch_result, allow_nan=False) + "\n")
            history.flush()
            _log_tensorboard_metrics(writer, "train", train_result.loss, train_result.evaluation.metrics, epoch)
            writer.add_scalar("validation/mean_macro_f1", validation_score, epoch)
            for source, result in validation_results.items():
                _log_tensorboard_metrics(writer, f"validation/{source}", result.loss, result.evaluation.metrics, epoch)
            writer.flush()
            if validation_score > best_macro_f1:
                best_macro_f1 = validation_score
                epochs_without_improvement = 0
                torch.save(checkpoint_state(model, optimizer, epoch, best_macro_f1, epochs_without_improvement, metadata), run_directory / "checkpoint-best.pt")
            else:
                epochs_without_improvement += 1
            torch.save(checkpoint_state(model, optimizer, epoch, best_macro_f1, epochs_without_improvement, metadata), run_directory / "checkpoint-last.pt")
            print(json.dumps(epoch_result, sort_keys=True))
            if epochs_without_improvement >= config.optimization.early_stopping_patience:
                break
    if not args.skip_test:
        load_checkpoint(str(run_directory / "checkpoint-best.pt"), model)
        with torch.no_grad():
            test_results, _ = run_source_evaluations(
                model, source_loaders["test"], criterion, device, tuple(label.value for label in class_labels),
                source_label_indices, max_batches=args.max_eval_batches,
            )
        test_metrics = {source: {"loss": result.loss, **result.evaluation.metrics} for source, result in test_results.items()}
        normal_amd_metrics = {}
        for source, result in test_results.items():
            normal_amd = result.evaluation.targets < 2
            probabilities = result.evaluation.probabilities[normal_amd, :2]
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            metrics = calculate_metrics(
                result.evaluation.targets[normal_amd], probabilities, ("normal", "amd")
            ).metrics
            normal_amd_metrics[source] = metrics
        duke_result = test_results.get("duke")
        if duke_result is not None:
            group_by_path = {record.path: record.eye_id for record in records_by_split["test"]["duke"]}
            test_metrics["duke"]["eye_level"] = calculate_grouped_metrics(
                duke_result.evaluation.targets,
                duke_result.evaluation.probabilities,
                [group_by_path[path] for path in duke_result.paths],  # type: ignore[list-item]
                tuple(class_labels[index].value for index in source_label_indices["duke"]),
            ).metrics
        write_json(
            run_directory / "metrics-test.json",
            {
                "by_source": test_metrics,
                "three_class_by_source": {
                    spec.name: test_metrics[spec.name] for spec in specs if len(spec.available_labels) == 3
                },
                "normal_amd_by_source": normal_amd_metrics,
            },
        )
        for source, result in test_results.items():
            labels = tuple(class_labels[index].value for index in source_label_indices[source])
            write_predictions(
                run_directory / "predictions-test" / f"{source}.csv",
                result.paths,
                result.evaluation.targets,
                result.evaluation.predictions,
                result.evaluation.probabilities,
                labels,
                sources=result.sources,
            )
    writer.close()
    print(f"Wrote run artifacts to {run_directory}")


def _evaluate(args: argparse.Namespace) -> None:
    class _RestrictedClassifier(nn.Module):
        def __init__(self, model: nn.Module, indices: list[int]) -> None:
            super().__init__()
            self.model = model
            self.indices = indices

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.model(images)[:, self.indices]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    source_spec = _spec_for_source(args.config, args.source)
    model_labels = tuple(UnifiedLabel(label) for label in metadata["class_labels"])
    class_labels = tuple(label for label in model_labels if label in source_spec.available_labels)
    if args.class_labels is not None:
        requested_labels = tuple(UnifiedLabel(label) for label in args.class_labels)
        class_labels = tuple(label for label in class_labels if label in requested_labels)
    if len(class_labels) < 2:
        raise ValueError("Model and target source must share at least two labels.")
    config = metadata["config"]
    preprocessing = PreprocessingSpec(image_size=config["data"]["image_size"])
    normalization = metadata["normalization"]
    augmentation = AugmentationConfig(**config["augmentation"])
    records = load_split_records(args.split_dir / f"{source_spec.name}.jsonl", "test")
    dataset = ManifestImageDataset(
        source_spec.root,
        records,
        class_labels,
        preprocessing,
        normalization["mean"],
        normalization["stdev"],
        augmentation,
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
            MaskedCrossEntropyLoss(),
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCT model training utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser(
        "baseline", help="Train one independently fine-tuned supervised model."
    )
    baseline.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
    baseline.add_argument("--source", required=True)
    baseline.add_argument(
        "--training-config", type=Path, default=Path("configs/training/resnet50.toml")
    )
    baseline.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    baseline.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    baseline.add_argument("--run-name")
    baseline.add_argument("--resume", type=Path)
    baseline.add_argument("--skip-test", action="store_true", help="Do not load or evaluate the test split.")
    baseline.add_argument("--device", default="cuda")
    baseline.add_argument("--epochs", type=int, help="Override configured epoch count.")
    baseline.add_argument(
        "--max-train-batches", type=int, help="Cap each training epoch for a local smoke test."
    )
    baseline.add_argument(
        "--max-eval-batches",
        type=int,
        help="Cap validation and test batches for a local smoke test.",
    )
    baseline.add_argument(
        "--workers",
        type=int,
        help="Processes for normalization statistics; defaults to all available CPU cores.",
    )
    baseline.set_defaults(handler=_baseline)
    fused = subparsers.add_parser(
        "fused", help="Train one partial-label-aware model across multiple sources."
    )
    fused.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
    fused.add_argument("--sources", nargs="+", default=["duke", "paima", "kermany", "octdl"])
    fused.add_argument(
        "--training-config", type=Path, default=Path("configs/training/resnet50.toml")
    )
    fused.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    fused.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    fused.add_argument("--run-name")
    fused.add_argument("--resume", type=Path)
    fused.add_argument("--skip-test", action="store_true", help="Do not load or evaluate test splits.")
    fused.add_argument("--device", default="cuda")
    fused.add_argument("--epochs", type=int, help="Override configured epoch count.")
    fused.add_argument("--max-train-batches", type=int)
    fused.add_argument("--max-eval-batches", type=int)
    fused.add_argument("--workers", type=int)
    fused.set_defaults(handler=_fused)
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint on one source test split."
    )
    evaluate.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--source", required=True)
    evaluate.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument(
        "--class-labels",
        nargs="+",
        choices=[label.value for label in UnifiedLabel],
        help="Restrict evaluation to the selected shared labels.",
    )
    evaluate.add_argument(
        "--max-eval-batches", type=int, help="Cap test batches for a local smoke test."
    )
    evaluate.set_defaults(handler=_evaluate)
    args = parser.parse_args(argv)
    args.handler(args)
