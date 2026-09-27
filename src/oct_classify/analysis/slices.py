"""Metrics on metadata slices and per-group error concentration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import f1_score

from oct_classify.analysis.loading import PredictionSet


def _slice_row(
    prediction_set: PredictionSet, members: np.ndarray, slice_type: str, value: str
) -> dict[str, object]:
    targets = prediction_set.targets[members]
    predictions = prediction_set.predictions[members]
    confidence = prediction_set.probabilities[members].max(axis=1)
    row: dict[str, object] = {
        "slice_type": slice_type,
        "slice_value": value,
        "images": int(members.sum()),
        "groups": len({prediction_set.group_keys[index] for index in np.flatnonzero(members)}),
        "accuracy": float((targets == predictions).mean()),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "mean_confidence": float(confidence.mean()),
    }
    for index, name in enumerate(prediction_set.class_names):
        row[f"true_{name}"] = int((targets == index).sum())
        row[f"predicted_{name}"] = float((predictions == index).mean())
    return row


def slice_metrics(prediction_set: PredictionSet, duplicate_keys: set[str]) -> list[dict[str, object]]:
    """Metrics by cohort, raw source label, and duplicate-cluster membership.

    Raw labels expose merged classes (e.g. PAIMA DRUSEN versus CNV inside `amd`); their
    `predicted_*` columns show where each raw class is sent.
    """
    rows = [_slice_row(prediction_set, np.ones(len(prediction_set), dtype=bool), "all", "all")]
    columns: dict[str, Iterable[str | None]] = {
        "cohort": prediction_set.cohorts,
        "raw_label": prediction_set.raw_labels,
        "in_duplicate_cluster": [
            str(f"{prediction_set.source}:{path}" in duplicate_keys) for path in prediction_set.paths
        ],
    }
    for slice_type, values in columns.items():
        array = np.asarray(list(values), dtype=object)
        if all(value is None for value in array):
            continue
        for value in sorted({str(value) for value in array}):
            rows.append(_slice_row(prediction_set, array.astype(str) == value, slice_type, value))
    return rows


def group_table(prediction_set: PredictionSet) -> list[dict[str, object]]:
    """One row per resampling group, sorted by error count so concentrated failures surface first."""
    by_group: dict[str, list[int]] = {}
    for index, key in enumerate(prediction_set.group_keys):
        by_group.setdefault(key, []).append(index)
    predictions = prediction_set.predictions
    rows = []
    for key, indices in by_group.items():
        targets = prediction_set.targets[indices]
        wrong = targets != predictions[indices]
        target_counts = Counter(prediction_set.class_names[target] for target in targets)
        row: dict[str, object] = {
            "group_key": key,
            "eye_ids": ";".join(sorted({str(prediction_set.eye_ids[i]) for i in indices})),
            "cohort": prediction_set.cohorts[indices[0]],
            "targets": ";".join(f"{name}:{count}" for name, count in sorted(target_counts.items())),
            "images": len(indices),
            "errors": int(wrong.sum()),
            "error_rate": float(wrong.mean()),
            "mean_confidence": float(prediction_set.probabilities[indices].max(axis=1).mean()),
        }
        for index, name in enumerate(prediction_set.class_names):
            row[f"predicted_{name}"] = float((predictions[indices] == index).mean())
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["errors"]), -float(row["error_rate"]), str(row["group_key"])))
    return rows


def error_concentration(groups: list[dict[str, object]]) -> dict[str, object]:
    """How much of the error mass the worst groups hold (Gini-style concentration summary)."""
    errors = np.asarray([int(row["errors"]) for row in groups])
    total = int(errors.sum())
    summary: dict[str, object] = {
        "groups": len(groups),
        "groups_with_errors": int((errors > 0).sum()),
        "total_errors": total,
    }
    for fraction in (0.05, 0.10, 0.25):
        count = max(1, int(np.ceil(len(groups) * fraction)))
        share = float(errors[:count].sum() / total) if total else None
        summary[f"error_share_top_{round(fraction * 100)}pct_groups"] = share
    return summary
