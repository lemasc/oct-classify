from oct_classify.data.models import ImageRecord
from oct_classify.data.splits import assign_group_splits, validate_supplied_splits
from oct_classify.data.taxonomy import ALL_LABELS, UnifiedLabel


def _record(path: str, group_id: str | None, split: str | None = None) -> ImageRecord:
    return ImageRecord(
        path=path,
        source="octdl",
        raw_label="AMD",
        label=UnifiedLabel.AMD,
        available_labels=ALL_LABELS,
        group_id=group_id,
        supplied_split=split,
    )


def test_validation_detects_group_leakage() -> None:
    report = validate_supplied_splits(
        [_record("a.jpg", "1", "train"), _record("b.jpg", "1", "test")]
    )

    assert report.groups_in_multiple_splits == {"octdl:1": ("test", "train")}
    assert not report.is_leakage_safe


def test_assignments_keep_each_group_in_one_split() -> None:
    records = [_record("a.jpg", "1"), _record("b.jpg", "1"), _record("c.jpg", "2")]

    assignments = assign_group_splits(records, {"train": 0.8, "test": 0.2}, seed=7)

    assert assignments.keys() == {"octdl:1", "octdl:2"}


def test_assignments_require_a_group_id() -> None:
    try:
        assign_group_splits([_record("a.jpg", None)], {"train": 1.0}, seed=7)
    except ValueError as error:
        assert "group ID" in str(error)
    else:
        raise AssertionError("Expected missing group IDs to prevent split assignment")
