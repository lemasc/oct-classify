from pathlib import Path

import pytest

from oct_classify.data.manifest import read_jsonl, write_jsonl
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
