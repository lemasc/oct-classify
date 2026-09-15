from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from oct_classify.training.evaluation import EvaluationResult, calculate_metrics


@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    evaluation: EvaluationResult
    paths: list[str]
    sources: list[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor, str, str]],
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    class_names: tuple[str, ...],
    *,
    optimizer: Optimizer | None = None,
    max_batches: int | None = None,
    metric_label_indices: tuple[int, ...] | None = None,
) -> EpochResult:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and training)
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    paths: list[str] = []
    sources: list[str] = []
    example_count = 0
    for batch_index, (images, batch_targets, available_masks, batch_sources, batch_paths) in enumerate(
        loader
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        available_masks = available_masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, batch_targets, available_masks)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()) * len(images))
        example_count += len(images)
        masked_logits = logits.masked_fill(~available_masks, torch.finfo(logits.dtype).min)
        probabilities.append(torch.softmax(masked_logits, dim=1).detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())
        paths.extend(batch_paths)
        sources.extend(batch_sources)
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    if metric_label_indices is not None:
        if not metric_label_indices:
            raise ValueError("metric_label_indices must not be empty.")
        index_array = np.asarray(metric_label_indices)
        target_map = {index: local_index for local_index, index in enumerate(metric_label_indices)}
        if not np.all(np.isin(target_array, index_array)):
            raise ValueError("Evaluation targets include a label outside metric_label_indices.")
        target_array = np.asarray([target_map[int(target)] for target in target_array])
        probability_array = probability_array[:, index_array]
        probability_array /= probability_array.sum(axis=1, keepdims=True)
    evaluation = calculate_metrics(target_array, probability_array, class_names)
    return EpochResult(sum(losses) / example_count, evaluation, paths, sources)


def run_source_evaluations(
    model: nn.Module,
    loaders: Mapping[str, DataLoader[tuple[torch.Tensor, int, torch.Tensor, str, str]]],
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    class_names: tuple[str, ...],
    class_label_indices: Mapping[str, tuple[int, ...]],
    *,
    max_batches: int | None = None,
) -> tuple[dict[str, EpochResult], float]:
    """Evaluate sources independently and return their unweighted macro-F1 mean."""
    results = {
        source: run_epoch(
            model,
            loader,
            criterion,
            device,
            tuple(class_names[index] for index in class_label_indices[source]),
            max_batches=max_batches,
            metric_label_indices=class_label_indices[source],
        )
        for source, loader in loaders.items()
    }
    score = float(
        np.mean([float(result.evaluation.metrics["macro_f1"]) for result in results.values()])
    )
    return results, score


def checkpoint_state(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_macro_f1: float,
    epochs_without_improvement: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "epochs_without_improvement": epochs_without_improvement,
        "metadata": metadata,
    }


def load_checkpoint(
    path: str, model: nn.Module, optimizer: Optimizer | None = None
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
