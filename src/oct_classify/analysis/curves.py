"""ROC/PR curves and screening operating points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

SENSITIVITY_TARGETS = (0.90, 0.95)


def _downsample(*arrays: np.ndarray, max_points: int) -> list[list[float | None]]:
    length = len(arrays[0])
    if length > max_points:
        keep = np.unique(np.linspace(0, length - 1, max_points).round().astype(np.int64))
        arrays = tuple(array[keep] for array in arrays)
    # sklearn prepends an infinite ROC threshold; JSON has no infinity.
    return [[float(value) if np.isfinite(value) else None for value in array] for array in arrays]


def roc_pr_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    max_points: int = 200,
) -> dict[str, dict[str, object]]:
    """One-vs-rest curves per class; classes absent from the targets are reported as undefined."""
    curves: dict[str, dict[str, object]] = {}
    for index, name in enumerate(class_names):
        positive = (targets == index).astype(np.int8)
        scores = probabilities[:, index]
        if positive.min() == positive.max():
            curves[name] = {"auroc": None, "average_precision": None, "roc": None, "pr": None}
            continue
        fpr, tpr, roc_thresholds = roc_curve(positive, scores)
        precision, recall, pr_thresholds = precision_recall_curve(positive, scores)
        fpr_points, tpr_points, roc_threshold_points = _downsample(
            fpr, tpr, roc_thresholds, max_points=max_points
        )
        # precision/recall have one more point than thresholds; the last is the (recall 0) anchor.
        precision_points, recall_points, pr_threshold_points = _downsample(
            precision[:-1], recall[:-1], pr_thresholds, max_points=max_points
        )
        curves[name] = {
            "auroc": float(roc_auc_score(positive, scores)),
            "average_precision": float(average_precision_score(positive, scores)),
            "prevalence": float(positive.mean()),
            "roc": {"fpr": fpr_points, "tpr": tpr_points, "threshold": roc_threshold_points},
            "pr": {
                "precision": precision_points,
                "recall": recall_points,
                "threshold": pr_threshold_points,
            },
        }
    return curves


def _binary_rates(positive: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    true_positive = int((positive & predicted).sum())
    false_positive = int((~positive & predicted).sum())
    false_negative = int((positive & ~predicted).sum())
    true_negative = int((~positive & ~predicted).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(true_positive, true_positive + false_positive)
    sensitivity = ratio(true_positive, true_positive + false_negative)
    f1 = ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "sensitivity": sensitivity,
        "specificity": ratio(true_negative, true_negative + false_positive),
        "ppv": precision,
        "npv": ratio(true_negative, true_negative + false_negative),
        "f1": f1,
    }


def per_class_rates(
    targets: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    """One-vs-rest sensitivity/specificity/PPV/NPV of the argmax decision."""
    return {
        name: _binary_rates(targets == index, predictions == index)
        for index, name in enumerate(class_names)
    }


def disease_operating_points(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    normal_label: str = "normal",
) -> dict[str, object] | None:
    """Collapse to disease-versus-normal and describe thresholds on `1 - p(normal)`.

    The argmax row is the deployed decision. Sensitivity-target rows use the highest threshold that
    reaches the target sensitivity. The best-F1 row is tuned on these same labels, so it is an
    optimistic upper bound that measures threshold shift, not a deployable operating point.
    """
    if normal_label not in class_names:
        return None
    normal_index = list(class_names).index(normal_label)
    positive = targets != normal_index
    if positive.all() or not positive.any():
        return None
    disease_score = 1 - probabilities[:, normal_index]
    argmax_positive = probabilities.argmax(axis=1) != normal_index
    points: dict[str, object] = {
        "score": f"1 - p({normal_label})",
        "argmax": _binary_rates(positive, argmax_positive),
    }
    # Candidate thresholds are the distinct scores, visited from strictest to most lenient.
    order = np.argsort(-disease_score, kind="stable")
    sorted_scores = disease_score[order]
    sorted_positive = positive[order]
    true_positive = np.cumsum(sorted_positive)
    false_positive = np.cumsum(~sorted_positive)
    candidates = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    sensitivity = true_positive[candidates] / positive.sum()
    for target in SENSITIVITY_TARGETS:
        threshold = float(sorted_scores[candidates[np.argmax(sensitivity >= target)]])
        points[f"sensitivity_{round(target * 100)}"] = {
            "threshold": threshold,
            **_binary_rates(positive, disease_score >= threshold),
        }
    f1 = 2 * true_positive / (true_positive + false_positive + positive.sum())
    best_threshold = float(sorted_scores[candidates[np.argmax(f1[candidates])]])
    points["best_f1_test_tuned"] = {
        "threshold": best_threshold,
        **_binary_rates(positive, disease_score >= best_threshold),
    }
    return points
