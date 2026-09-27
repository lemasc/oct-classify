import numpy as np

from oct_classify.analysis.calibration import calibration_summary
from oct_classify.analysis.curves import disease_operating_points, roc_pr_curves


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    # Ten predictions at 0.8 confidence, eight correct.
    probabilities = np.tile([0.8, 0.2], (10, 1))
    targets = np.array([0] * 8 + [1] * 2)

    summary = calibration_summary(targets, probabilities, ("normal", "amd"))

    assert np.isclose(summary["ece"], 0.0)
    assert summary["reliability"][12]["count"] == 10


def test_ece_measures_overconfidence() -> None:
    probabilities = np.tile([0.9, 0.1], (10, 1))
    targets = np.array([0] * 5 + [1] * 5)

    summary = calibration_summary(targets, probabilities, ("normal", "amd"))

    assert np.isclose(summary["ece"], 0.4)


def test_sensitivity_operating_point_uses_highest_qualifying_threshold() -> None:
    targets = np.array([0, 0, 0, 1, 1, 1, 1])
    disease = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9])
    probabilities = np.stack([1 - disease, disease], axis=1)

    points = disease_operating_points(targets, probabilities, ("normal", "amd"))

    assert np.isclose(points["sensitivity_90"]["threshold"], 0.3)
    assert points["sensitivity_90"]["sensitivity"] == 1.0
    assert np.isclose(points["sensitivity_90"]["specificity"], 2 / 3)
    assert points["argmax"]["sensitivity"] == 0.75


def test_curves_are_json_safe_and_skip_absent_classes() -> None:
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])

    curves = roc_pr_curves(targets, probabilities, ("normal", "amd", "dme"))

    assert curves["dme"]["auroc"] is None
    assert curves["amd"]["auroc"] == 1.0
    assert all(value is None or np.isfinite(value) for value in curves["amd"]["roc"]["threshold"])
