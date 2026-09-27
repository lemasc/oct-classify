import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from oct_classify.analysis.bootstrap import (
    FoldPredictions,
    _WeightedAuroc,
    bootstrap_metrics,
    encode_groups,
    group_confusion,
    metrics_from_confusion,
)


def test_vectorized_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 3, 200)
    predictions = np.where(rng.random(200) < 0.7, targets, rng.integers(0, 3, 200))
    metrics = metrics_from_confusion(confusion_matrix(targets, predictions).astype(float))

    assert np.isclose(metrics["accuracy"], (targets == predictions).mean())
    assert np.isclose(metrics["balanced_accuracy"], balanced_accuracy_score(targets, predictions))
    assert np.isclose(metrics["macro_f1"], f1_score(targets, predictions, average="macro"))


def test_group_confusion_sums_to_full_confusion() -> None:
    codes, n_groups = encode_groups(["a", "b", "a", "c"])
    targets = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 1])

    by_group = group_confusion(codes, targets, predictions, n_groups, 2)

    assert by_group.shape == (3, 2, 2)
    assert by_group.sum(axis=0).tolist() == confusion_matrix(targets, predictions).tolist()


def test_weighted_auroc_matches_sklearn_with_ties() -> None:
    targets = np.array([0, 0, 1, 1, 1, 0])
    positive = np.array([0.2, 0.5, 0.5, 0.9, 0.3, 0.1])
    probabilities = np.stack([1 - positive, positive], axis=1)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.0])

    score = _WeightedAuroc(targets, probabilities)(weights)

    assert np.isclose(score, roc_auc_score(targets, positive, sample_weight=weights))


def test_single_group_bootstrap_has_zero_width_interval() -> None:
    targets = np.array([0, 1, 1, 0])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    codes, n_groups = encode_groups(["p1"] * 4)

    result = bootstrap_metrics(
        codes, n_groups, [FoldPredictions(targets, probabilities)], ("normal", "amd"), n_boot=50
    )

    accuracy = result["metrics"]["accuracy"]
    assert accuracy["point"] == 0.75
    assert accuracy["ci_low"] == accuracy["ci_high"] == 0.75


def test_shared_folds_average_point_metrics() -> None:
    targets = np.array([0, 1, 0, 1])
    perfect = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    inverted = perfect[:, ::-1]
    codes, n_groups = encode_groups(["a", "b", "c", "d"])

    result = bootstrap_metrics(
        codes,
        n_groups,
        [FoldPredictions(targets, perfect), FoldPredictions(targets, inverted)],
        ("normal", "amd"),
        n_boot=100,
    )

    assert result["fold_count"] == 2
    assert result["metrics"]["accuracy"]["point"] == 0.5
    assert np.isclose(result["metrics"]["accuracy"]["mean"], 0.5)
