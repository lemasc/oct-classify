import numpy as np

from oct_classify.training.evaluation import calculate_metrics


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
