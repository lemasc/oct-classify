"""Probability calibration diagnostics for saved softmax outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip((values * n_bins).astype(np.int64), 0, n_bins - 1)


def _calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int
) -> tuple[float, float, list[dict[str, float | int | None]]]:
    bins = _bin_indices(confidence, n_bins)
    total = len(confidence)
    ece = 0.0
    mce = 0.0
    table: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        members = bins == index
        count = int(members.sum())
        row: dict[str, float | int | None] = {
            "bin_low": index / n_bins,
            "bin_high": (index + 1) / n_bins,
            "count": count,
            "mean_confidence": None,
            "accuracy": None,
        }
        if count:
            mean_confidence = float(confidence[members].mean())
            accuracy = float(correct[members].mean())
            gap = abs(accuracy - mean_confidence)
            ece += count / total * gap
            mce = max(mce, gap)
            row.update(mean_confidence=mean_confidence, accuracy=accuracy)
        table.append(row)
    return ece, mce, table


def calibration_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    n_bins: int = 15,
    histogram_bins: int = 20,
) -> dict[str, object]:
    """Top-label ECE/MCE, classwise ECE, Brier score, NLL, and reliability tables."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece, mce, reliability = _calibration_error(confidence, correct, n_bins)
    classwise = {}
    for index, name in enumerate(class_names):
        class_ece, _, _ = _calibration_error(probabilities[:, index], targets == index, n_bins)
        classwise[name] = class_ece
    one_hot = np.eye(len(class_names))[targets]
    true_probability = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    edges = np.linspace(0, 1, histogram_bins + 1)
    return {
        "n_bins": n_bins,
        "ece": ece,
        "mce": mce,
        "classwise_ece": classwise,
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "nll": float(-np.log(true_probability).mean()),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
        "reliability": reliability,
        "confidence_histogram": {
            "edges": edges.tolist(),
            "correct": np.histogram(confidence[correct], edges)[0].tolist(),
            "wrong": np.histogram(confidence[~correct], edges)[0].tolist(),
        },
    }
