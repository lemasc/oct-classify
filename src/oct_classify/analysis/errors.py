"""Select images for error galleries and matched correct-prediction controls."""

from __future__ import annotations

from collections import Counter

import numpy as np

from oct_classify.analysis.loading import PredictionSet


def _row(
    prediction_set: PredictionSet, index: int, kind: str, row_id: str, rank: int
) -> dict[str, object]:
    target = int(prediction_set.targets[index])
    prediction = int(prediction_set.predictions[index])
    probabilities = prediction_set.probabilities[index]
    row: dict[str, object] = {
        "row_id": row_id,
        "kind": kind,
        "rank": rank,
        "path": prediction_set.paths[index],
        "source": prediction_set.source,
        "group_key": prediction_set.group_keys[index],
        "cohort": prediction_set.cohorts[index],
        "raw_label": prediction_set.raw_labels[index],
        "target": prediction_set.class_names[target],
        "prediction": prediction_set.class_names[prediction],
        "confidence": float(probabilities[prediction]),
        "true_probability": float(probabilities[target]),
    }
    for name, value in zip(prediction_set.class_names, probabilities, strict=True):
        row[f"probability_{name}"] = float(value)
    return row


def _take_capped(
    prediction_set: PredictionSet, ordered: np.ndarray, size: int, max_per_group: int
) -> list[int]:
    taken: list[int] = []
    per_group: Counter[str] = Counter()
    for index in ordered:
        key = prediction_set.group_keys[index]
        if per_group[key] >= max_per_group:
            continue
        per_group[key] += 1
        taken.append(int(index))
        if len(taken) == size:
            break
    return taken


def error_gallery(
    prediction_set: PredictionSet, *, size_per_cell: int = 24, max_per_group: int = 3
) -> list[dict[str, object]]:
    """The most confidently wrong images of each off-diagonal confusion cell.

    Capping images per group keeps one patient's volume from filling a whole cell.
    """
    predictions = prediction_set.predictions
    confidence = prediction_set.probabilities.max(axis=1)
    rows: list[dict[str, object]] = []
    n_classes = len(prediction_set.class_names)
    for target in range(n_classes):
        for prediction in range(n_classes):
            if target == prediction:
                continue
            cell = np.flatnonzero((prediction_set.targets == target) & (predictions == prediction))
            ordered = cell[np.argsort(-confidence[cell], kind="stable")]
            for rank, index in enumerate(
                _take_capped(prediction_set, ordered, size_per_cell, max_per_group)
            ):
                rows.append(_row(prediction_set, index, "error", f"e{len(rows):04d}", rank))
    return rows


def control_sample(
    prediction_set: PredictionSet,
    *,
    size_per_class: int = 24,
    max_per_group: int = 3,
    seed: int = 20260927,
) -> list[dict[str, object]]:
    """A random sample of correct predictions per class, as a baseline for attention maps."""
    rng = np.random.default_rng(seed)
    predictions = prediction_set.predictions
    rows: list[dict[str, object]] = []
    for target in range(len(prediction_set.class_names)):
        correct = np.flatnonzero((prediction_set.targets == target) & (predictions == target))
        ordered = rng.permutation(correct)
        for rank, index in enumerate(
            _take_capped(prediction_set, ordered, size_per_class, max_per_group)
        ):
            rows.append(_row(prediction_set, index, "control", f"c{len(rows):04d}", rank))
    return rows
