from __future__ import annotations

import random
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, str]],
    criterion: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...],
    *,
    optimizer: Optimizer | None = None,
    max_batches: int | None = None,
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
    example_count = 0
    for batch_index, (images, batch_targets, batch_paths) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, batch_targets)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()) * len(images))
        example_count += len(images)
        probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())
        paths.extend(batch_paths)
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    evaluation = calculate_metrics(target_array, probability_array, class_names)
    return EpochResult(sum(losses) / example_count, evaluation, paths)


def checkpoint_state(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_macro_f1: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
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
