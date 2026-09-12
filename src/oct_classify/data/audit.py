from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from oct_classify.data.models import ImageRecord

_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class AuditReport:
    image_count: int
    label_counts: dict[str, int]
    supplied_split_counts: dict[str, int]
    images_without_group: int
    invalid_images: list[str]
    widths: tuple[int, int] | None
    heights: tuple[int, int] | None
    raw_label_counts: dict[str, int]
    available_labels: list[str]
    group_count: int
    images_per_group: dict[str, float | int] | None
    file_suffix_counts: dict[str, int]
    decoded_format_counts: dict[str, int]
    image_mode_counts: dict[str, int]
    width_distribution: dict[str, float | int] | None
    height_distribution: dict[str, float | int] | None
    aspect_ratio_distribution: dict[str, float | int] | None
    intensity_distribution: dict[str, dict[str, float | int]] | None
    unmanifested_images: list[str]
    exact_duplicates: list[dict[str, object]]
    near_duplicates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def integrity_failures(self) -> list[str]:
        failures: list[str] = []
        if self.invalid_images:
            failures.append(f"{len(self.invalid_images)} invalid or missing manifest image(s)")
        for duplicate in [*self.exact_duplicates, *self.near_duplicates]:
            if duplicate["crosses_supplied_splits"]:
                failures.append("duplicate images cross supplied splits")
                break
        for duplicate in [*self.exact_duplicates, *self.near_duplicates]:
            if duplicate["conflicting_labels"]:
                failures.append("duplicate images have conflicting labels")
                break
        return failures


@dataclass(frozen=True, slots=True)
class CrossSourceAuditReport:
    image_count: int
    invalid_images: list[str]
    exact_duplicates: list[dict[str, object]]
    near_duplicates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _duplicate_summary(records: list[ImageRecord], indices: Iterable[int]) -> dict[str, object]:
    duplicate_records = [records[index] for index in sorted(indices)]
    labels = {record.label.value for record in duplicate_records}
    supplied_splits = {
        record.supplied_split for record in duplicate_records if record.supplied_split
    }
    group_keys = {record.group_key for record in duplicate_records if record.group_key}
    return {
        "paths": [f"{record.source}:{record.path}" for record in duplicate_records],
        "group_keys": sorted(group_keys),
        "labels": sorted(labels),
        "supplied_splits": sorted(supplied_splits),
        "crosses_groups": len(group_keys) > 1,
        "crosses_supplied_splits": len(supplied_splits) > 1,
        "conflicting_labels": len(labels) > 1,
        "crosses_sources": len({record.source for record in duplicate_records}) > 1,
    }


def _perceptual_duplicate_pairs(
    records: list[ImageRecord], hashes: dict[int, imagehash.ImageHash], max_distance: int
) -> list[dict[str, object]]:
    """Find nearby perceptual hash clusters without comparing every image pair."""
    hash_indices: dict[int, list[int]] = {}
    for index, value in hashes.items():
        hash_indices.setdefault(int(str(value), 16), []).append(index)

    buckets: dict[tuple[int, int], set[int]] = {}
    for bits in hash_indices:
        for chunk in range(4):
            bucket = (chunk, (bits >> (chunk * 16)) & 0xFFFF)
            buckets.setdefault(bucket, set()).add(bits)

    candidates = {
        pair
        for hashes_in_bucket in buckets.values()
        for pair in itertools.combinations(sorted(hashes_in_bucket), 2)
    }

    duplicates: list[dict[str, object]] = []
    for bits, indices in sorted(hash_indices.items()):
        if len(indices) > 1:
            summary = _duplicate_summary(records, indices)
            summary["distance"] = 0
            duplicates.append(summary)
    for first, second in sorted(candidates):
        distance = (first ^ second).bit_count()
        if distance <= max_distance:
            summary = _duplicate_summary(records, [*hash_indices[first], *hash_indices[second]])
            summary["distance"] = distance
            duplicates.append(summary)
    return duplicates


def audit_records(
    root: Path,
    records: Iterable[ImageRecord],
    *,
    perceptual_hashes: bool = True,
    max_hash_distance: int = 5,
    hash_timing_log: TextIO | None = None,
) -> AuditReport:
    if max_hash_distance < 0:
        raise ValueError("The perceptual hash distance must be non-negative.")
    records = list(records)
    labels = Counter(record.label.value for record in records)
    raw_labels = Counter(record.raw_label for record in records)
    splits = Counter(record.supplied_split for record in records if record.supplied_split)
    invalid_images: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    image_means: list[float] = []
    image_stdevs: list[float] = []
    image_p01s: list[float] = []
    image_p99s: list[float] = []
    suffixes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    group_sizes = Counter(record.group_key for record in records if record.group_key)
    exact_hashes: dict[str, list[int]] = {}
    perceptual_hash_values: dict[int, imagehash.ImageHash] = {}

    for index, record in enumerate(records):
        suffixes[Path(record.path).suffix.lower() or "<none>"] += 1
        image_path = root / record.path
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                formats[image.format or "<unknown>"] += 1
                modes[image.mode] += 1
                pixels = np.asarray(image.convert("L"), dtype=np.float32)
                image_means.append(float(pixels.mean()))
                image_stdevs.append(float(pixels.std()))
                image_p01s.append(float(np.percentile(pixels, 1)))
                image_p99s.append(float(np.percentile(pixels, 99)))
                if perceptual_hashes:
                    hash_start = perf_counter()
                    perceptual_hash_values[index] = imagehash.phash(image)
                    if hash_timing_log is not None:
                        hash_timing_log.write(
                            f"{record.source}\t{record.path}\t{perf_counter() - hash_start:.6f}\n"
                        )
        except (OSError, UnidentifiedImageError):
            invalid_images.append(record.path)
        else:
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
            with image_path.open("rb") as handle:
                digest = hashlib.blake2b(handle.read(), digest_size=16).hexdigest()
            exact_hashes.setdefault(digest, []).append(index)

    manifest_paths = {record.path for record in records}
    unmanifested_images = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and str(path.relative_to(root)) not in manifest_paths
    )
    exact_duplicates = [
        _duplicate_summary(records, indices)
        for indices in exact_hashes.values()
        if len(indices) > 1
    ]
    near_duplicates = (
        _perceptual_duplicate_pairs(records, perceptual_hash_values, max_hash_distance)
        if perceptual_hashes
        else []
    )

    return AuditReport(
        image_count=len(records),
        label_counts=dict(sorted(labels.items())),
        supplied_split_counts=dict(sorted(splits.items())),
        images_without_group=sum(record.group_id is None for record in records),
        invalid_images=invalid_images,
        widths=(min(widths), max(widths)) if widths else None,
        heights=(min(heights), max(heights)) if heights else None,
        raw_label_counts=dict(sorted(raw_labels.items())),
        available_labels=sorted(
            {label.value for record in records for label in record.available_labels}
        ),
        group_count=len(group_sizes),
        images_per_group=_distribution(list(group_sizes.values())),
        file_suffix_counts=dict(sorted(suffixes.items())),
        decoded_format_counts=dict(sorted(formats.items())),
        image_mode_counts=dict(sorted(modes.items())),
        width_distribution=_distribution(widths),
        height_distribution=_distribution(heights),
        aspect_ratio_distribution=_distribution(aspect_ratios),
        intensity_distribution={
            "mean": _distribution(image_means),
            "stdev": _distribution(image_stdevs),
            "p01": _distribution(image_p01s),
            "p99": _distribution(image_p99s),
        }
        if image_means
        else None,
        unmanifested_images=unmanifested_images,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )


def audit_cross_source_records(
    record_sets: Iterable[tuple[Path, Iterable[ImageRecord]]],
    *,
    perceptual_hashes: bool = True,
    max_hash_distance: int = 5,
    hash_timing_log: TextIO | None = None,
) -> CrossSourceAuditReport:
    """Report duplicate candidates shared by distinct configured dataset roots."""
    if max_hash_distance < 0:
        raise ValueError("The perceptual hash distance must be non-negative.")

    locations = [
        (root / record.path, record) for root, records in record_sets for record in records
    ]
    records = [record for _, record in locations]
    invalid_images: list[str] = []
    exact_hashes: dict[str, list[int]] = {}
    perceptual_hash_values: dict[int, imagehash.ImageHash] = {}

    for index, (image_path, record) in enumerate(locations):
        try:
            with Image.open(image_path) as image:
                image.load()
                if perceptual_hashes:
                    hash_start = perf_counter()
                    perceptual_hash_values[index] = imagehash.phash(image)
                    if hash_timing_log is not None:
                        hash_timing_log.write(
                            f"{record.source}\t{record.path}\t{perf_counter() - hash_start:.6f}\n"
                        )
        except (OSError, UnidentifiedImageError):
            invalid_images.append(f"{record.source}:{record.path}")
        else:
            with image_path.open("rb") as handle:
                digest = hashlib.blake2b(handle.read(), digest_size=16).hexdigest()
            exact_hashes.setdefault(digest, []).append(index)

    exact_duplicates = [
        _duplicate_summary(records, indices)
        for indices in exact_hashes.values()
        if len(indices) > 1 and len({records[index].source for index in indices}) > 1
    ]
    near_duplicates = (
        [
            duplicate
            for duplicate in _perceptual_duplicate_pairs(
                records, perceptual_hash_values, max_hash_distance
            )
            if duplicate["crosses_sources"]
        ]
        if perceptual_hashes
        else []
    )
    return CrossSourceAuditReport(
        image_count=len(records),
        invalid_images=invalid_images,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )
