import csv
import json
from pathlib import Path

import numpy as np
import pytest

from oct_classify.analysis.campaign import aggregate_set
from oct_classify.analysis.loading import discover_prediction_sets
from oct_classify.analysis.report import analyze_prediction_set
from oct_classify.data.manifest import write_jsonl
from oct_classify.data.models import ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.training.evaluation import calculate_metrics

LABELS = frozenset({UnifiedLabel.NORMAL, UnifiedLabel.AMD, UnifiedLabel.DME})


def _manifest(directory: Path) -> list[ImageRecord]:
    records = [
        ImageRecord(
            path=f"eye{eye}/{scan}.tif",
            source="duke",
            raw_label=label.value.upper(),
            label=label,
            available_labels=LABELS,
            group_id=f"eye{eye}",
            label_unit="eye",
            eye_id=f"eye{eye}",
        )
        for eye, label in enumerate([UnifiedLabel.NORMAL, UnifiedLabel.AMD, UnifiedLabel.DME] * 2)
        for scan in range(3)
    ]
    write_jsonl(directory / "manifests" / "duke.jsonl", records)
    return records


def _write_run(run: Path, records: list[ImageRecord], seed: int) -> None:
    rng = np.random.default_rng(seed)
    names = ("normal", "amd", "dme")
    probabilities = rng.dirichlet(np.ones(3), len(records))
    targets = np.array([names.index(record.label.value) for record in records])
    probabilities[np.arange(len(records)), targets] += 0.6
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    run.mkdir(parents=True)
    with (run / "predictions-test.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "target", "prediction", *[f"probability_{n}" for n in names]])
        for record, target, probability in zip(records, targets, probabilities, strict=True):
            writer.writerow(
                [record.path, names[target], names[probability.argmax()], *probability.tolist()]
            )
    metrics = calculate_metrics(targets, probabilities, names).metrics
    (run / "metrics-test.json").write_text(json.dumps(metrics))
    (run / "dataset.json").write_text(json.dumps({"source": "duke"}))


def test_analysis_matches_saved_metrics_and_reports_eye_level(tmp_path: Path) -> None:
    records = _manifest(tmp_path)
    _write_run(tmp_path / "run", records, seed=1)

    [prediction_set] = discover_prediction_sets(tmp_path / "run", tmp_path / "manifests")
    summary = analyze_prediction_set(
        prediction_set,
        tmp_path / "out",
        set(),
        n_boot=50,
        n_boot_auroc=20,
        seed=0,
        gallery_size=2,
        max_per_group=1,
    )

    assert prediction_set.name == "test-duke"
    assert summary["groups"] == 6
    assert summary["reference_check"]["matches"]
    assert summary["eye_level"]["bootstrap"]["unit_count"] == 6
    for name in ("summary.json", "curves.json", "slices.csv", "groups.csv", "errors.csv"):
        assert (tmp_path / "out" / name).is_file()


def test_unmatched_prediction_path_fails(tmp_path: Path) -> None:
    records = _manifest(tmp_path)
    _write_run(tmp_path / "run", records, seed=1)
    write_jsonl(tmp_path / "manifests" / "duke.jsonl", records[1:])

    with pytest.raises(ValueError, match="absent from the duke manifest"):
        discover_prediction_sets(tmp_path / "run", tmp_path / "manifests")


def test_campaign_pools_disjoint_folds_and_shares_identical_folds(tmp_path: Path) -> None:
    records = _manifest(tmp_path)
    _write_run(tmp_path / "a", records[:9], seed=1)
    _write_run(tmp_path / "b", records[9:], seed=2)
    _write_run(tmp_path / "c", records[:9], seed=3)
    load = lambda name: discover_prediction_sets(tmp_path / name, tmp_path / "manifests")[0]

    pooled = aggregate_set([load("a"), load("b")], n_boot=20, n_boot_auroc=10, seed=0)
    shared = aggregate_set([load("a"), load("c")], n_boot=20, n_boot_auroc=10, seed=0)

    assert pooled["mode"] == "pooled"
    assert pooled["bootstrap"]["row_count"] == 18
    assert pooled["eye_level_bootstrap"]["unit_count"] == 6
    assert shared["mode"] == "shared"
    assert shared["bootstrap"]["fold_count"] == 2
