"""Write the torch-free analysis bundle for one prediction set."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from oct_classify.analysis.bootstrap import FoldPredictions, bootstrap_metrics, encode_groups
from oct_classify.analysis.calibration import calibration_summary
from oct_classify.analysis.curves import disease_operating_points, per_class_rates, roc_pr_curves
from oct_classify.analysis.errors import control_sample, error_gallery
from oct_classify.analysis.loading import PredictionSet
from oct_classify.analysis.slices import error_concentration, group_table, slice_metrics
from oct_classify.training.evaluation import calculate_grouped_metrics, calculate_metrics
from oct_classify.training.outputs import write_json

REFERENCE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reference_check(
    prediction_set: PredictionSet, metrics: dict[str, object]
) -> dict[str, object] | None:
    """Compare recomputed metrics with those the training CLI wrote alongside the predictions."""
    reference = prediction_set.reference_metrics
    if reference is None:
        return None
    differences = {}
    for name in REFERENCE_METRICS:
        expected, actual = reference.get(name), metrics.get(name)
        if expected is None or actual is None:
            differences[name] = None if expected is actual else "undefined in one side"
        else:
            differences[name] = abs(float(expected) - float(actual))
    matches = all(value is None or (isinstance(value, float) and value < 1e-9) for value in differences.values())
    return {"matches": matches, "absolute_differences": differences}


def eye_level_set(prediction_set: PredictionSet) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Mean eye probabilities for eye-labelled sources, matching the training CLI's eye metrics."""
    if prediction_set.label_unit != "eye":
        return None
    if any(eye_id is None for eye_id in prediction_set.eye_ids):
        raise ValueError(f"{prediction_set.name}: eye-labelled rows require an eye ID.")
    eye_ids = [f"{prediction_set.source}:{eye_id}" for eye_id in prediction_set.eye_ids]
    result = calculate_grouped_metrics(
        prediction_set.targets, prediction_set.probabilities, eye_ids, prediction_set.class_names
    )
    return result.targets, result.probabilities, sorted(set(eye_ids))


def analyze_prediction_set(
    prediction_set: PredictionSet,
    output: Path,
    duplicate_keys: set[str],
    *,
    n_boot: int,
    n_boot_auroc: int,
    seed: int,
    gallery_size: int,
    max_per_group: int,
) -> dict[str, object]:
    targets = prediction_set.targets
    probabilities = prediction_set.probabilities
    class_names = prediction_set.class_names
    point = calculate_metrics(targets, probabilities, class_names).metrics
    codes, n_groups = encode_groups(prediction_set.group_keys)
    summary: dict[str, object] = {
        "set": prediction_set.name,
        "run_dir": str(prediction_set.run_dir),
        "predictions": str(prediction_set.predictions_path),
        "source": prediction_set.source,
        "class_names": list(class_names),
        "label_unit": prediction_set.label_unit,
        "full_source": prediction_set.full_source,
        "images": len(prediction_set),
        "groups": n_groups,
        "metrics": point,
        "reference_check": _reference_check(prediction_set, point),
        "bootstrap": {
            "unit": "source:group_id (patient or volume)",
            **bootstrap_metrics(
                codes,
                n_groups,
                [FoldPredictions(targets, probabilities)],
                class_names,
                n_boot=n_boot,
                n_boot_auroc=n_boot_auroc,
                seed=seed,
            ),
        },
        "per_class": per_class_rates(targets, prediction_set.predictions, class_names),
        "disease_operating_points": disease_operating_points(targets, probabilities, class_names),
    }
    eye = eye_level_set(prediction_set)
    if eye is not None:
        eye_targets, eye_probabilities, eye_ids = eye
        eye_codes = np.arange(len(eye_ids))
        summary["eye_level"] = {
            "metrics": calculate_metrics(eye_targets, eye_probabilities, class_names).metrics,
            "bootstrap": {
                "unit": "eye (mean B-scan probability)",
                **bootstrap_metrics(
                    eye_codes,
                    len(eye_ids),
                    [FoldPredictions(eye_targets, eye_probabilities)],
                    class_names,
                    n_boot=n_boot,
                    n_boot_auroc=n_boot_auroc,
                    seed=seed,
                ),
            },
            "calibration": calibration_summary(eye_targets, eye_probabilities, class_names),
        }
    calibration = calibration_summary(targets, probabilities, class_names)
    summary["calibration"] = {
        key: value for key, value in calibration.items() if key not in ("reliability", "confidence_histogram")
    }
    groups = group_table(prediction_set)
    summary["error_concentration"] = error_concentration(groups)
    errors = error_gallery(prediction_set, size_per_cell=gallery_size, max_per_group=max_per_group)
    controls = control_sample(
        prediction_set, size_per_class=gallery_size, max_per_group=max_per_group, seed=seed
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", json_safe(summary))
    write_json(
        output / "calibration.json",
        json_safe({key: calibration[key] for key in ("reliability", "confidence_histogram")}),
    )
    write_rows(output / "reliability.csv", calibration["reliability"])  # type: ignore[arg-type]
    write_json(output / "curves.json", roc_pr_curves(targets, probabilities, class_names))
    write_rows(output / "slices.csv", slice_metrics(prediction_set, duplicate_keys))
    write_rows(output / "groups.csv", groups)
    write_rows(output / "errors.csv", errors)
    write_rows(output / "controls.csv", controls)
    return summary


def json_safe(value: object) -> object:
    """Replace NaN/inf (e.g. undefined recall) with null so strict JSON can be written."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
