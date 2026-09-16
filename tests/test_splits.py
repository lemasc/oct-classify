from oct_classify.data.models import ImageRecord
from oct_classify.data.splits import (
    apply_derived_splits,
    assign_group_splits,
    create_derived_splits,
    create_stratified_cross_validation_splits,
    validate_derived_splits,
    validate_supplied_splits,
)
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


def test_derived_splits_keep_linked_groups_together_and_are_deterministic() -> None:
    records = [
        _record("a.jpg", "1"),
        _record("b.jpg", "2"),
        _record("c.jpg", "3"),
        _record("d.jpg", "4"),
    ]
    near_duplicates = [{"distance": 0, "paths": ["octdl:a.jpg", "octdl:b.jpg"]}]

    first = create_derived_splits(records, near_duplicates, {"train": 0.5, "test": 0.5}, seed=3)
    second = create_derived_splits(records, near_duplicates, {"train": 0.5, "test": 0.5}, seed=3)

    assert first.assignments == second.assignments
    assert first.assignments["octdl:1"] == first.assignments["octdl:2"]
    assert (
        validate_derived_splits(
            apply_derived_splits(records, first.assignments), first.linked_group_components
        )
        == []
    )


def test_derived_splits_preserve_supplied_kermany_test_partition() -> None:
    records = [
        _record("train-a.jpg", "1", "train"),
        _record("train-b.jpg", "2", "train"),
        _record("test-a.jpg", "3", "test"),
    ]

    result = create_derived_splits(
        records,
        [],
        {"train": 0.85, "val": 0.15},
        seed=3,
        preserve_supplied_test=True,
    )

    assert result.assignments["octdl:3"] == "test"
    assert {result.assignments["octdl:1"], result.assignments["octdl:2"]} <= {"train", "val"}


def test_derived_split_validation_rejects_linked_groups_in_different_splits() -> None:
    records = apply_derived_splits(
        [_record("a.jpg", "1"), _record("b.jpg", "2")],
        {"octdl:1": "train", "octdl:2": "test"},
    )

    assert validate_derived_splits(records, [("octdl:1", "octdl:2")]) == [
        "pHash-zero linked groups cross splits: octdl:1, octdl:2"
    ]


def test_cross_validation_covers_each_group_once_in_outer_test_folds() -> None:
    records = [
        ImageRecord(
            path=f"{label.value}-{number}.jpg",
            source="duke",
            raw_label=label.value,
            label=label,
            available_labels=ALL_LABELS,
            group_id=f"{label.value}-{number}",
            label_unit="eye",
            eye_id=f"{label.value}-{number}",
        )
        for label in ALL_LABELS
        for number in range(15)
    ]

    result = create_stratified_cross_validation_splits(
        records, [], folds=5, validation_groups_per_class=3, seed=7
    )

    assert len(result.folds) == 5
    for fold in result.folds:
        test_groups = {group for group, split in fold.assignments.items() if split == "test"}
        validation_groups = {group for group, split in fold.assignments.items() if split == "val"}
        assert len(test_groups) == 9
        assert len(validation_groups) == 9
        for label in ALL_LABELS:
            assert sum(group.startswith(f"duke:{label.value}-") for group in test_groups) == 3
            assert sum(group.startswith(f"duke:{label.value}-") for group in validation_groups) == 3
        assert validate_derived_splits(
            apply_derived_splits(records, fold.assignments), fold.linked_group_components
        ) == []
    assert {
        group
        for fold in result.folds
        for group, split in fold.assignments.items()
        if split == "test"
    } == {record.group_key for record in records}
