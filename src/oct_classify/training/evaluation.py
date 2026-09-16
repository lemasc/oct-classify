from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: dict[str, float | list[list[int]] | None]
    probabilities: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray


def calculate_metrics(
    targets: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]
) -> EvaluationResult:
    if len(targets) == 0:
        raise ValueError("Cannot calculate metrics for no examples.")
    predictions = probabilities.argmax(axis=1)
    metrics: dict[str, float | list[list[int]] | None] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(len(class_names))
        ).tolist(),
    }
    try:
        scores = []
        for index in range(len(class_names)):
            binary_targets = (targets == index).astype(np.int8)
            if len(np.unique(binary_targets)) != 2:
                raise ValueError("A class is absent from the evaluation targets.")
            scores.append(roc_auc_score(binary_targets, probabilities[:, index]))
        metrics["macro_auroc"] = float(np.mean(scores))
    except ValueError:
        # A test split may not contain every class, making AUROC undefined rather than zero.
        metrics["macro_auroc"] = None
    return EvaluationResult(metrics, probabilities, targets, predictions)


def calculate_grouped_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    group_ids: list[str],
    class_names: tuple[str, ...],
) -> EvaluationResult:
    """Average image probabilities within a labeled eye or volume before scoring."""
    if len(group_ids) != len(targets):
        raise ValueError("group_ids must have the same length as targets.")
    grouped_targets: list[int] = []
    grouped_probabilities: list[np.ndarray] = []
    by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        by_group.setdefault(group_id, []).append(index)
    for group_id in sorted(by_group):
        indices = by_group[group_id]
        group_targets = targets[indices]
        if len(np.unique(group_targets)) != 1:
            raise ValueError(f"Group {group_id!r} has multiple target labels.")
        grouped_targets.append(int(group_targets[0]))
        grouped_probabilities.append(probabilities[indices].mean(axis=0))
    return calculate_metrics(
        np.asarray(grouped_targets), np.asarray(grouped_probabilities), class_names
    )
