"""Aggregate one prediction set across the checkpoints of a campaign (CV folds or LOSO repeats)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oct_classify.analysis.bootstrap import FoldPredictions, bootstrap_metrics, encode_groups
from oct_classify.analysis.calibration import calibration_summary
from oct_classify.analysis.loading import PredictionSet
from oct_classify.analysis.report import eye_level_set
from oct_classify.training.evaluation import calculate_metrics

POINT_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc")


def _fold_row(prediction_set: PredictionSet, targets, probabilities) -> dict[str, object]:
    metrics = calculate_metrics(targets, probabilities, prediction_set.class_names).metrics
    return {
        "run": prediction_set.run_dir.name,
        "rows": len(targets),
        **{name: metrics[name] for name in POINT_METRICS},
        "ece": calibration_summary(targets, probabilities, prediction_set.class_names)["ece"],
    }


def _spread(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    spread: dict[str, dict[str, float | None]] = {}
    for name in (*POINT_METRICS, "ece"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        spread[name] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return spread


def _mode(sets: Sequence[PredictionSet]) -> str:
    path_sets = [frozenset(prediction_set.paths) for prediction_set in sets]
    if all(paths == path_sets[0] for paths in path_sets):
        return "shared"
    if sum(len(paths) for paths in path_sets) == len(frozenset().union(*path_sets)):
        return "pooled"
    return "overlapping"


def aggregate_set(
    sets: Sequence[PredictionSet], *, n_boot: int, n_boot_auroc: int, seed: int
) -> dict[str, object]:
    """Aggregate checkpoints evaluated on the same rows (`shared`) or on disjoint rows (`pooled`).

    `shared`: e.g. five CV checkpoints on the fixed Kermany test set. The bootstrap resamples groups
    once per replicate for every checkpoint, so the interval is for mean checkpoint performance.
    `pooled`: e.g. Duke CV outer folds. Out-of-fold predictions are concatenated so every patient
    appears once, giving a single cross-validated estimate.
    """
    first = sets[0]
    class_names = first.class_names
    if any(prediction_set.class_names != class_names for prediction_set in sets):
        return {"mode": "skipped", "reason": "class names differ between runs"}
    mode = _mode(sets)
    result: dict[str, object] = {
        "mode": mode,
        "source": first.source,
        "class_names": list(class_names),
        "label_unit": first.label_unit,
        "runs": [str(prediction_set.run_dir) for prediction_set in sets],
    }
    if mode == "overlapping":
        result["reason"] = "runs share some but not all rows; neither shared nor pooled applies"
        return result
    result["per_run"] = [
        _fold_row(prediction_set, prediction_set.targets, prediction_set.probabilities)
        for prediction_set in sets
    ]
    result["per_run_spread"] = _spread(result["per_run"])  # type: ignore[arg-type]
    if mode == "shared":
        order = {path: index for index, path in enumerate(first.paths)}
        folds = []
        for prediction_set in sets:
            alignment = np.empty(len(first), dtype=np.int64)
            alignment[[order[path] for path in prediction_set.paths]] = np.arange(len(first))
            folds.append(
                FoldPredictions(
                    prediction_set.targets[alignment], prediction_set.probabilities[alignment]
                )
            )
        group_keys = first.group_keys
    else:
        folds = [
            FoldPredictions(
                np.concatenate([prediction_set.targets for prediction_set in sets]),
                np.concatenate([prediction_set.probabilities for prediction_set in sets]),
            )
        ]
        group_keys = [key for prediction_set in sets for key in prediction_set.group_keys]
        result["pooled_metrics"] = calculate_metrics(
            folds[0].targets, folds[0].probabilities, class_names
        ).metrics
    codes, n_groups = encode_groups(group_keys)
    result["bootstrap"] = {
        "unit": "source:group_id (patient or volume)",
        **bootstrap_metrics(
            codes, n_groups, folds, class_names, n_boot=n_boot, n_boot_auroc=n_boot_auroc, seed=seed
        ),
    }
    if first.label_unit == "eye":
        eye_sets = [eye_level_set(prediction_set) for prediction_set in sets]
        if mode == "shared":
            eye_folds = [FoldPredictions(targets, probabilities) for targets, probabilities, _ in eye_sets]
            eye_count = len(eye_sets[0][2])
        else:
            eye_folds = [
                FoldPredictions(
                    np.concatenate([targets for targets, _, _ in eye_sets]),
                    np.concatenate([probabilities for _, probabilities, _ in eye_sets]),
                )
            ]
            eye_count = sum(len(ids) for _, _, ids in eye_sets)
            result["eye_level_pooled_metrics"] = calculate_metrics(
                eye_folds[0].targets, eye_folds[0].probabilities, class_names
            ).metrics
        result["eye_level_per_run"] = [
            _fold_row(prediction_set, targets, probabilities)
            for prediction_set, (targets, probabilities, _) in zip(sets, eye_sets, strict=True)
        ]
        result["eye_level_bootstrap"] = {
            "unit": "eye (mean B-scan probability)",
            **bootstrap_metrics(
                np.arange(eye_count),
                eye_count,
                eye_folds,
                class_names,
                n_boot=n_boot,
                n_boot_auroc=n_boot_auroc,
                seed=seed,
            ),
        }
    return result


def campaign_table(name: str, sets: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Flatten aggregated sets into one row per (set, level, metric) for reports."""
    rows = []
    for set_name, result in sets.items():
        for level, key in (("image", "bootstrap"), ("eye", "eye_level_bootstrap")):
            bootstrap = result.get(key)
            if not isinstance(bootstrap, dict):
                continue
            per_run = result.get("per_run" if level == "image" else "eye_level_per_run", [])
            spread = _spread(per_run)  # type: ignore[arg-type]
            for metric, interval in bootstrap["metrics"].items():
                rows.append(
                    {
                        "campaign": name,
                        "set": set_name,
                        "level": level,
                        "mode": result["mode"],
                        "runs": len(result["runs"]),  # type: ignore[arg-type]
                        "rows": bootstrap["row_count"],
                        "units": bootstrap["unit_count"],
                        "metric": metric,
                        "point": interval["point"],
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "run_sd": spread.get(metric, {}).get("sd"),
                    }
                )
    return rows


def group_sets_by_name(
    sets_by_run: Sequence[Sequence[PredictionSet]],
) -> dict[str, list[PredictionSet]]:
    grouped: dict[str, list[PredictionSet]] = {}
    for run_sets in sets_by_run:
        for prediction_set in run_sets:
            grouped.setdefault(prediction_set.name, []).append(prediction_set)
    return grouped


def default_output(name: str) -> Path:
    return Path("data") / "analysis" / name
