"""Discover saved prediction CSVs in a run directory and join them to manifest metadata."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from oct_classify.data.manifest import read_jsonl
from oct_classify.data.models import ImageRecord

PROBABILITY_PREFIX = "probability_"


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """One saved prediction file together with the manifest metadata of each row."""

    name: str
    run_dir: Path
    predictions_path: Path
    source: str
    class_names: tuple[str, ...]
    paths: list[str]
    targets: np.ndarray
    probabilities: np.ndarray
    group_keys: list[str]
    eye_ids: list[str | None]
    cohorts: list[str | None]
    raw_labels: list[str]
    label_unit: str
    full_source: bool
    reference_metrics: dict[str, Any] | None

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    def __len__(self) -> int:
        return len(self.paths)


@cache
def _manifest_index(manifest_path: Path) -> dict[str, ImageRecord]:
    return {record.path: record for record in read_jsonl(manifest_path)}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_set(
    name: str,
    run_dir: Path,
    predictions_path: Path,
    source: str,
    manifests_dir: Path,
    *,
    full_source: bool = False,
    reference_metrics: dict[str, Any] | None = None,
) -> PredictionSet:
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    class_names = tuple(
        field.removeprefix(PROBABILITY_PREFIX)
        for field in fieldnames
        if field.startswith(PROBABILITY_PREFIX)
    )
    if len(class_names) < 2:
        raise ValueError(f"{predictions_path} has fewer than two probability columns.")
    if not rows:
        raise ValueError(f"{predictions_path} has no rows.")
    class_index = {label: index for index, label in enumerate(class_names)}
    index = _manifest_index(manifests_dir / f"{source}.jsonl")
    missing = [row["path"] for row in rows if row["path"] not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} prediction paths in {predictions_path} are absent from the "
            f"{source} manifest, e.g. {missing[0]!r}."
        )
    records = [index[row["path"]] for row in rows]
    for row, record in zip(rows, records, strict=True):
        if row["target"] != record.label.value:
            raise ValueError(
                f"{predictions_path}: target {row['target']!r} for {row['path']!r} does not "
                f"match the manifest label {record.label.value!r}."
            )
    label_units = {record.label_unit for record in records}
    if len(label_units) != 1:
        raise ValueError(f"{predictions_path} mixes label units: {sorted(label_units)}")
    return PredictionSet(
        name=name,
        run_dir=run_dir,
        predictions_path=predictions_path,
        source=source,
        class_names=class_names,
        paths=[row["path"] for row in rows],
        targets=np.asarray([class_index[row["target"]] for row in rows]),
        probabilities=np.asarray(
            [[float(row[f"{PROBABILITY_PREFIX}{label}"]) for label in class_names] for row in rows]
        ),
        # Records without a group are their own resampling unit.
        group_keys=[record.group_key or f"{record.source}:path:{record.path}" for record in records],
        eye_ids=[record.eye_id for record in records],
        cohorts=[record.cohort for record in records],
        raw_labels=[record.raw_label for record in records],
        label_unit=label_units.pop(),
        full_source=full_source,
        reference_metrics=reference_metrics,
    )


def discover_prediction_sets(run_dir: Path, manifests_dir: Path) -> list[PredictionSet]:
    """Find every test and evaluation prediction file written by the training CLI."""
    sets: list[PredictionSet] = []
    metrics_test = _read_json(run_dir / "metrics-test.json")
    single_test = run_dir / "predictions-test.csv"
    if single_test.is_file():
        dataset = _read_json(run_dir / "dataset.json") or {}
        source = dataset.get("source")
        if not source:
            raise ValueError(f"{run_dir}/dataset.json does not name the run source.")
        sets.append(
            read_prediction_set(
                f"test-{source}",
                run_dir,
                single_test,
                source,
                manifests_dir,
                reference_metrics=metrics_test,
            )
        )
    for path in sorted((run_dir / "predictions-test").glob("*.csv")):
        source = path.stem
        reference = (metrics_test or {}).get("by_source", {}).get(source)
        sets.append(
            read_prediction_set(
                f"test-{source}", run_dir, path, source, manifests_dir, reference_metrics=reference
            )
        )
    for path in sorted((run_dir / "evaluations").glob("*/predictions.csv")):
        metrics = _read_json(path.parent / "metrics.json") or {}
        source = metrics.get("target_source")
        if not source:
            raise ValueError(f"{path.parent}/metrics.json does not name the target source.")
        sets.append(
            read_prediction_set(
                f"eval-{path.parent.name}",
                run_dir,
                path,
                source,
                manifests_dir,
                full_source=bool(metrics.get("full_source", False)),
                reference_metrics=metrics,
            )
        )
    if not sets:
        raise FileNotFoundError(f"No prediction files found under {run_dir}")
    return sets


def load_duplicate_paths(audits_dir: Path, source: str) -> set[str]:
    """Return `source:path` keys that belong to any exact or pHash duplicate cluster."""
    report = _read_json(audits_dir / f"{source}.json")
    if report is None:
        return set()
    return {
        path
        for kind in ("exact_duplicates", "near_duplicates")
        for cluster in report.get(kind, [])
        for path in cluster["paths"]
    }
