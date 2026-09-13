from pathlib import Path

import pytest

from oct_classify.data.manifest import apply_processing_decisions, read_jsonl, write_jsonl
from oct_classify.data.models import ImageRecord
from oct_classify.data.taxonomy import ALL_LABELS, UnifiedLabel


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    record = ImageRecord(
        path="AMD1/scan.png",
        source="duke",
        raw_label="AMD",
        label=UnifiedLabel.AMD,
        available_labels=ALL_LABELS,
        group_id="AMD1",
    )

    write_jsonl(path, [record])

    assert list(read_jsonl(path)) == [record]


def test_record_rejects_label_not_screened_by_source() -> None:
    with pytest.raises(ValueError, match="not available"):
        ImageRecord(
            path="scan.png",
            source="octid",
            raw_label="DME",
            label=UnifiedLabel.DME,
            available_labels=frozenset({UnifiedLabel.NORMAL, UnifiedLabel.AMD}),
        )


def test_processing_decisions_quarantine_conflicts_and_deduplicate_exact_copies() -> None:
    canonical = ImageRecord(
        path="a.png",
        source="source",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
    )
    duplicate = ImageRecord(
        path="b.png",
        source="source",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
    )
    repeated_path = ImageRecord(
        path="a.png",
        source="source",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
    )
    conflicting = ImageRecord(
        path="c.png",
        source="source",
        raw_label="AMD",
        label=UnifiedLabel.AMD,
        available_labels=ALL_LABELS,
    )

    retained, quarantined, deduplicated = apply_processing_decisions(
        [canonical, duplicate, repeated_path, conflicting],
        exact_duplicates=[
            {
                "paths": ["source:a.png", "source:b.png"],
                "quarantine_eligible": False,
            }
        ],
        near_duplicates=[
            {
                "paths": ["source:b.png", "source:c.png"],
                "quarantine_eligible": True,
            }
        ],
    )

    assert retained == [canonical]
    assert quarantined == 2
    assert deduplicated == 1
