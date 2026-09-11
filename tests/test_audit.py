from io import StringIO
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from oct_classify.data.audit import _perceptual_duplicate_pairs, audit_records
from oct_classify.data.models import ImageRecord
from oct_classify.data.taxonomy import ALL_LABELS, UnifiedLabel


def _record(path: str, group_id: str, split: str | None = None) -> ImageRecord:
    return ImageRecord(
        path=path,
        source="source",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
        group_id=group_id,
        supplied_split=split,
    )


def test_audit_profiles_images_and_detects_cross_split_exact_duplicates(tmp_path: Path) -> None:
    Image.new("L", (8, 4), color=64).save(tmp_path / "first.png")
    Image.new("L", (8, 4), color=64).save(tmp_path / "second.png")
    Image.new("RGB", (2, 2), color="red").save(tmp_path / "unmanifested.jpg")

    report = audit_records(
        tmp_path,
        [_record("first.png", "patient-1", "train"), _record("second.png", "patient-2", "test")],
    )

    assert report.decoded_format_counts == {"PNG": 2}
    assert report.image_mode_counts == {"L": 2}
    assert report.width_distribution == {
        "min": 8.0,
        "p05": 8.0,
        "median": 8.0,
        "p95": 8.0,
        "max": 8.0,
    }
    assert report.intensity_distribution is not None
    assert report.unmanifested_images == ["unmanifested.jpg"]
    assert report.exact_duplicates[0]["crosses_supplied_splits"] is True
    assert report.integrity_failures == ["duplicate images cross supplied splits"]


def test_perceptual_hash_candidates_include_nearby_hashes() -> None:
    records = [_record("first.png", "one"), _record("second.png", "two")]
    first = imagehash.ImageHash(np.zeros((8, 8), dtype=bool))
    second_bits = np.zeros((8, 8), dtype=bool)
    second_bits.flat[:3] = True
    second = imagehash.ImageHash(second_bits)

    duplicates = _perceptual_duplicate_pairs(records, {0: first, 1: second}, max_distance=3)

    assert duplicates[0]["distance"] == 3
    assert duplicates[0]["crosses_groups"] is True


def test_audit_rejects_negative_hash_distance(tmp_path: Path) -> None:
    try:
        audit_records(tmp_path, [], max_hash_distance=-1)
    except ValueError as error:
        assert "distance" in str(error)
    else:
        raise AssertionError("Expected a negative perceptual hash distance to fail")


def test_audit_writes_per_image_phash_timings(tmp_path: Path) -> None:
    Image.new("L", (8, 4), color=64).save(tmp_path / "image.png")
    timing_log = StringIO()

    audit_records(tmp_path, [_record("image.png", "patient-1")], hash_timing_log=timing_log)

    source, path, seconds = timing_log.getvalue().strip().split("\t")
    assert (source, path) == ("source", "image.png")
    assert float(seconds) >= 0
