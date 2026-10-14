import numpy as np
import pytest

from oct_classify.training.evaluation import calculate_grouped_metrics, calculate_metrics


def test_metrics_include_binary_auroc_and_confusion_matrix() -> None:
    result = calculate_metrics(
        np.array([0, 1, 0, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
        ("normal", "amd"),
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["macro_auroc"] == 1.0
    assert result.metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_metrics_leaves_auroc_undefined_when_a_class_is_absent() -> None:
    result = calculate_metrics(
        np.array([0, 0]), np.array([[0.9, 0.1], [0.8, 0.2]]), ("normal", "amd")
    )

    assert result.metrics["macro_auroc"] is None


def test_grouped_metrics_average_probabilities_before_scoring() -> None:
    result = calculate_grouped_metrics(
        np.array([0, 0, 1, 1]),
        np.array([[0.9, 0.1], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]]),
        ["eye-1", "eye-1", "eye-2", "eye-2"],
        ("normal", "amd"),
    )

    assert result.targets.tolist() == [0, 1]
    assert result.probabilities.tolist() == [[0.8, 0.2], [0.25, 0.75]]
    assert result.metrics["accuracy"] == 1.0


def test_macro_f1_averages_over_classes_present_in_targets() -> None:
    # Normal/AMD targets only; one normal image is predicted DME (class 2).
    targets = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.eye(3)[[0, 0, 2, 1, 1, 1]]
    result = calculate_metrics(targets, probabilities, ("normal", "amd", "dme"))
    # Normal: precision 1, recall 2/3 -> F1 0.8; AMD: F1 1.0. Mean over the two present classes.
    assert result.metrics["macro_f1"] == pytest.approx(0.9)


def test_macro_f1_unchanged_when_every_class_is_present() -> None:
    from sklearn.metrics import f1_score

    targets = np.array([0, 1, 2, 0, 1, 2, 2])
    predictions = np.array([0, 2, 2, 1, 1, 2, 0])
    result = calculate_metrics(targets, np.eye(3)[predictions], ("normal", "amd", "dme"))
    assert result.metrics["macro_f1"] == pytest.approx(f1_score(targets, predictions, average="macro"))
