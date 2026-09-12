from io import StringIO
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from oct_classify.data.audit import (
    _perceptual_duplicate_pairs,
    audit_cross_source_records,
    audit_records,
    load_perceptual_hash_cache,
    write_perceptual_hash_cache,
)
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


def test_audit_reuses_content_keyed_perceptual_hash_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("L", (8, 4), color=64).save(image_path)
    cache: dict[str, str] = {}
    audit_records(tmp_path, [_record("image.png", "patient-1")], perceptual_hash_cache=cache)
    timing_log = StringIO()

    audit_records(
        tmp_path,
        [_record("image.png", "patient-1")],
        hash_timing_log=timing_log,
        perceptual_hash_cache=cache,
    )

    assert len(cache) == 1
    assert timing_log.getvalue() == ""


def test_perceptual_hash_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "phash-cache.json"
    hashes = {"digest": "0123456789abcdef"}

    write_perceptual_hash_cache(cache_path, hashes)

    assert load_perceptual_hash_cache(cache_path) == hashes


def test_cross_source_audit_reports_only_cross_source_duplicates(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    Image.new("L", (8, 4), color=64).save(first_root / "image.png")
    Image.new("L", (8, 4), color=64).save(second_root / "image.png")

    first = _record("image.png", "patient-1")
    second = ImageRecord(
        path="image.png",
        source="other",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
        group_id="patient-2",
    )
    report = audit_cross_source_records([(first_root, [first]), (second_root, [second])])

    assert report.invalid_images == []
    assert report.exact_duplicates[0]["crosses_sources"] is True
    assert report.near_duplicates[0]["distance"] == 0


def test_cross_source_audit_reuses_computed_perceptual_hashes(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_path = first_root / "image.png"
    second_path = second_root / "image.png"
    Image.new("L", (8, 4), color=64).save(first_path)
    Image.new("L", (8, 4), color=64).save(second_path)
    first = _record("image.png", "patient-1")
    second = ImageRecord(
        path="image.png",
        source="other",
        raw_label="NORMAL",
        label=UnifiedLabel.NORMAL,
        available_labels=ALL_LABELS,
        group_id="patient-2",
    )
    computed_hashes = {
        first_path: imagehash.phash(Image.open(first_path)),
        second_path: imagehash.phash(Image.open(second_path)),
    }
    timing_log = StringIO()

    audit_cross_source_records(
        [(first_root, [first]), (second_root, [second])],
        hash_timing_log=timing_log,
        computed_perceptual_hashes=computed_hashes,
    )

    assert timing_log.getvalue() == ""
