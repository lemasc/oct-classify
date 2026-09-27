"""Cluster bootstrap confidence intervals that resample patients, eyes, or volumes.

Images from one patient are correlated, so resampling images understates uncertainty. Every
replicate draws groups with replacement; each image then carries its group's multiplicity as a
weight. Count metrics come from weighted confusion matrices and AUROC from a weighted
Mann-Whitney statistic, so replicates stay cheap on large full-source evaluations.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

COUNT_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


@dataclass(frozen=True, slots=True)
class FoldPredictions:
    """Predictions of one checkpoint, row-aligned with the shared group codes."""

    targets: np.ndarray
    probabilities: np.ndarray

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)


def encode_groups(group_keys: Sequence[str]) -> tuple[np.ndarray, int]:
    _, codes = np.unique(np.asarray(group_keys, dtype=object), return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1


def group_confusion(
    codes: np.ndarray, targets: np.ndarray, predictions: np.ndarray, n_groups: int, n_classes: int
) -> np.ndarray:
    """Return a `(groups, classes, classes)` tensor of each group's confusion matrix."""
    flat = (codes * n_classes + targets) * n_classes + predictions
    counts = np.bincount(flat, minlength=n_groups * n_classes * n_classes)
    return counts.reshape(n_groups, n_classes, n_classes).astype(np.float64)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized accuracy, balanced accuracy, and macro-F1 over `(..., C, C)` matrices.

    Matches scikit-learn: balanced accuracy averages recall over classes present in the targets,
    and macro-F1 averages over classes present in either targets or predictions.
    """
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    total = support.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = diagonal.sum(axis=-1) / total
        recall = np.where(support > 0, diagonal / support, np.nan)
        present = (support + predicted) > 0
        f1 = np.where(present, 2 * diagonal / (support + predicted), np.nan)
    with warnings.catch_warnings():
        # A replicate may miss a class entirely; its recall is then undefined, not zero.
        warnings.simplefilter("ignore", RuntimeWarning)
        balanced_accuracy = np.nanmean(recall, axis=-1)
        macro_f1 = np.nanmean(f1, axis=-1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        **{f"recall_{index}": recall[..., index] for index in range(confusion.shape[-1])},
    }


class _WeightedAuroc:
    """One-vs-rest AUROC with per-row weights, exact under ties (equals trapezoidal ROC area)."""

    def __init__(self, targets: np.ndarray, probabilities: np.ndarray) -> None:
        self.classes = []
        for index in range(probabilities.shape[1]):
            _, inverse = np.unique(probabilities[:, index], return_inverse=True)
            self.classes.append((inverse, (targets == index).astype(np.float64)))

    def __call__(self, weights: np.ndarray) -> float | None:
        scores = []
        for inverse, positive in self.classes:
            positive_weight = np.bincount(inverse, weights * positive)
            negative_weight = np.bincount(inverse, weights * (1 - positive))
            total_positive = positive_weight.sum()
            total_negative = negative_weight.sum()
            if total_positive == 0 or total_negative == 0:
                return None
            negative_below = np.cumsum(negative_weight) - negative_weight
            wins = positive_weight * (negative_below + 0.5 * negative_weight)
            scores.append(wins.sum() / (total_positive * total_negative))
        return float(np.mean(scores))


def _interval(values: np.ndarray, point: float | None) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"point": point, "mean": None, "ci_low": None, "ci_high": None, "n_valid": 0}
    return {
        "point": point,
        "mean": float(finite.mean()),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "n_valid": len(finite),
    }


def bootstrap_metrics(
    codes: np.ndarray,
    n_groups: int,
    folds: Sequence[FoldPredictions],
    class_names: Sequence[str],
    *,
    n_boot: int = 2000,
    n_boot_auroc: int = 500,
    seed: int = 20260927,
) -> dict[str, object]:
    """Bootstrap the fold-averaged metrics, resampling groups once per replicate for all folds.

    With one fold this is an ordinary cluster bootstrap. With several checkpoints evaluated on the
    same rows (e.g. five CV checkpoints on a fixed test set), the shared resample yields an interval
    for mean checkpoint performance without counting repeated test images as independent.
    """
    if not folds:
        raise ValueError("At least one fold is required.")
    n_classes = len(class_names)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    boot: dict[str, list[np.ndarray]] = {}
    point: dict[str, list[float]] = {}
    auroc_boot = np.full((len(folds), min(n_boot_auroc, n_boot)), np.nan)
    auroc_point: list[float | None] = []
    for fold_index, fold in enumerate(folds):
        if len(fold.targets) != len(codes):
            raise ValueError("Every fold must be row-aligned with the group codes.")
        confusion = group_confusion(codes, fold.targets, fold.predictions, n_groups, n_classes)
        for name, value in metrics_from_confusion(confusion.sum(axis=0)).items():
            point.setdefault(name, []).append(float(value))
        replicate_confusion = np.einsum("bg,gij->bij", counts, confusion)
        for name, value in metrics_from_confusion(replicate_confusion).items():
            boot.setdefault(name, []).append(value)
        auroc = _WeightedAuroc(fold.targets, fold.probabilities)
        auroc_point.append(auroc(np.ones(len(codes))))
        for replicate in range(auroc_boot.shape[1]):
            score = auroc(counts[replicate][codes].astype(np.float64))
            if score is not None:
                auroc_boot[fold_index, replicate] = score
    metrics: dict[str, object] = {}
    for name, values in boot.items():
        label = name
        if name.startswith("recall_"):
            label = f"recall_{class_names[int(name.removeprefix('recall_'))]}"
        # A replicate counts only when it is defined for every fold.
        fold_mean = np.stack(values).mean(axis=0)
        point_values = [value for value in point[name] if np.isfinite(value)]
        metrics[label] = _interval(
            fold_mean, float(np.mean(point_values)) if point_values else None
        )
    valid_points = [value for value in auroc_point if value is not None]
    auroc_all_folds = len(valid_points) == len(auroc_point)
    metrics["macro_auroc"] = {
        **_interval(
            auroc_boot.mean(axis=0), float(np.mean(valid_points)) if auroc_all_folds else None
        ),
        "n_boot": int(auroc_boot.shape[1]),
    }
    return {
        "unit_count": n_groups,
        "row_count": len(codes),
        "fold_count": len(folds),
        "n_boot": n_boot,
        "seed": seed,
        "metrics": metrics,
    }
