from __future__ import annotations

import hashlib
import itertools
import multiprocessing
import os
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from oct_classify.data.models import ImageRecord

_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        exact_reasons = {
            reason
            for duplicate in self.exact_duplicates
            for reason in duplicate["quarantine_reasons"]
        }
        near_reasons = {
            reason
            for duplicate in self.near_duplicates
            for reason in duplicate["quarantine_reasons"]
        }
        if "exact_label_conflict" in exact_reasons:
            failures.append("exact duplicate components have conflicting labels")
        if "phash_zero_label_conflict" in near_reasons:
            failures.append("pHash-distance-zero components have conflicting labels")
        if "phash_zero_split_crossing" in near_reasons:
            failures.append("pHash-distance-zero components cross supplied splits")
        return failures


@dataclass(frozen=True, slots=True)
class CrossSourceAuditReport:
    image_count: int
    invalid_images: list[str]
    exact_duplicates: list[dict[str, object]]
    near_duplicates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComputedImageAudit:
    file_digest: str
    perceptual_hash: imagehash.ImageHash | None


@dataclass(frozen=True, slots=True)
class _ImageAnalysis:
    index: int
    image_path: Path
    valid: bool
    file_digest: str | None = None
    perceptual_hash: imagehash.ImageHash | None = None
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    image_mode: str | None = None
    mean: float | None = None
    stdev: float | None = None
    p01: float | None = None
    p99: float | None = None
    phash_seconds: float | None = None


def _analyze_image(
    job: tuple[int, Path, bool, bool],
) -> _ImageAnalysis:
    """Analyze one file in a subprocess; report aggregation remains in the parent."""
    index, image_path, perceptual_hashes, profile_statistics = job
    try:
        digest = _file_digest(image_path)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            image_format = image.format or "<unknown>"
            image_mode = image.mode
            perceptual_hash = None
            phash_seconds = None
            if profile_statistics:
                pixels = np.asarray(image.convert("L"), dtype=np.float32)
                mean = float(pixels.mean())
                stdev = float(pixels.std())
                p01 = float(np.percentile(pixels, 1))
                p99 = float(np.percentile(pixels, 99))
            else:
                mean = stdev = p01 = p99 = None
            if perceptual_hashes:
                hash_start = perf_counter()
                perceptual_hash = imagehash.phash(image)
                phash_seconds = perf_counter() - hash_start
    except (OSError, UnidentifiedImageError):
        return _ImageAnalysis(index=index, image_path=image_path, valid=False)
    return _ImageAnalysis(
        index=index,
        image_path=image_path,
        valid=True,
        file_digest=digest,
        perceptual_hash=perceptual_hash,
        width=width,
        height=height,
        image_format=image_format,
        image_mode=image_mode,
        mean=mean,
        stdev=stdev,
        p01=p01,
        p99=p99,
        phash_seconds=phash_seconds,
    )


def _analyze_images(
    jobs: Iterable[tuple[int, Path, bool, bool]], *, workers: int | None
) -> list[_ImageAnalysis]:
    if workers is not None and workers < 1:
        raise ValueError("The audit worker count must be at least one.")
    jobs = list(jobs)
    if not jobs:
        return []
    if workers == 1:
        return [_analyze_image(job) for job in jobs]

    worker_count = workers or os.cpu_count() or 1
    chunksize = max(1, len(jobs) // (worker_count * 4))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(_analyze_image, jobs, chunksize=chunksize))


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


def classify_duplicate_for_quarantine(
    duplicate: dict[str, object], *, exact: bool
) -> dict[str, object]:
    """Classify an audit duplicate component under the locked quarantine policy."""
    reasons: list[str] = []
    if exact:
        if duplicate["conflicting_labels"]:
            reasons.append("exact_label_conflict")
    elif duplicate["distance"] == 0:
        if duplicate["conflicting_labels"]:
            reasons.append("phash_zero_label_conflict")
        if duplicate["crosses_supplied_splits"]:
            reasons.append("phash_zero_split_crossing")
    return {
        **duplicate,
        "quarantine_eligible": bool(reasons),
        "quarantine_reasons": reasons,
    }


def _classify_duplicates(
    duplicates: Iterable[dict[str, object]], *, exact: bool
) -> list[dict[str, object]]:
    return [classify_duplicate_for_quarantine(duplicate, exact=exact) for duplicate in duplicates]


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
    computed_images: dict[Path, ComputedImageAudit] | None = None,
    workers: int | None = None,
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

    for record in records:
        suffixes[Path(record.path).suffix.lower() or "<none>"] += 1
    analyses = _analyze_images(
        ((index, root / record.path, perceptual_hashes, True) for index, record in enumerate(records)),
        workers=workers,
    )
    for analysis in analyses:
        record = records[analysis.index]
        if not analysis.valid:
            invalid_images.append(record.path)
            continue
        assert analysis.file_digest is not None
        assert analysis.width is not None and analysis.height is not None
        assert analysis.image_format is not None and analysis.image_mode is not None
        formats[analysis.image_format] += 1
        modes[analysis.image_mode] += 1
        widths.append(analysis.width)
        heights.append(analysis.height)
        aspect_ratios.append(analysis.width / analysis.height)
        assert analysis.mean is not None
        assert analysis.stdev is not None
        assert analysis.p01 is not None and analysis.p99 is not None
        image_means.append(analysis.mean)
        image_stdevs.append(analysis.stdev)
        image_p01s.append(analysis.p01)
        image_p99s.append(analysis.p99)
        exact_hashes.setdefault(analysis.file_digest, []).append(analysis.index)
        if perceptual_hashes:
            assert analysis.perceptual_hash is not None
            perceptual_hash_values[analysis.index] = analysis.perceptual_hash
            if hash_timing_log is not None:
                assert analysis.phash_seconds is not None
                hash_timing_log.write(
                    f"{record.source}\t{record.path}\t{analysis.phash_seconds:.6f}\n"
                )
        if computed_images is not None:
            computed_images[analysis.image_path] = ComputedImageAudit(
                file_digest=analysis.file_digest,
                perceptual_hash=analysis.perceptual_hash if perceptual_hashes else None,
            )

    manifest_paths = {record.path for record in records}
    unmanifested_images = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and str(path.relative_to(root)) not in manifest_paths
    )
    exact_duplicates = _classify_duplicates(
        [
            _duplicate_summary(records, indices)
            for indices in exact_hashes.values()
            if len(indices) > 1
        ],
        exact=True,
    )
    near_duplicates = _classify_duplicates(
        _perceptual_duplicate_pairs(records, perceptual_hash_values, max_hash_distance)
        if perceptual_hashes
        else [],
        exact=False,
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
    computed_images: dict[Path, ComputedImageAudit] | None = None,
    workers: int | None = None,
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

    cached_images: dict[int, ComputedImageAudit] = {}
    jobs = []
    for index, (image_path, _) in enumerate(locations):
        computed_image = computed_images.get(image_path) if computed_images is not None else None
        if computed_image is not None and (
            not perceptual_hashes or computed_image.perceptual_hash is not None
        ):
            cached_images[index] = computed_image
        else:
            jobs.append((index, image_path, perceptual_hashes, False))
    analyses = {analysis.index: analysis for analysis in _analyze_images(jobs, workers=workers)}

    for index, (image_path, record) in enumerate(locations):
        computed_image = cached_images.get(index)
        analysis = analyses.get(index)
        if analysis is not None and not analysis.valid:
            invalid_images.append(f"{record.source}:{record.path}")
            continue
        if computed_image is not None:
            digest = computed_image.file_digest
            perceptual_hash = computed_image.perceptual_hash
        else:
            assert analysis is not None and analysis.file_digest is not None
            digest = analysis.file_digest
            perceptual_hash = analysis.perceptual_hash
            if computed_images is not None:
                computed_images[image_path] = ComputedImageAudit(digest, perceptual_hash)
            if perceptual_hashes and hash_timing_log is not None:
                assert analysis.phash_seconds is not None
                hash_timing_log.write(
                    f"{record.source}\t{record.path}\t{analysis.phash_seconds:.6f}\n"
                )
        if perceptual_hashes:
            assert perceptual_hash is not None
            perceptual_hash_values[index] = perceptual_hash
        exact_hashes.setdefault(digest, []).append(index)

    exact_duplicates = _classify_duplicates(
        [
            _duplicate_summary(records, indices)
            for indices in exact_hashes.values()
            if len(indices) > 1 and len({records[index].source for index in indices}) > 1
        ],
        exact=True,
    )
    near_duplicates = _classify_duplicates(
        [
            duplicate
            for duplicate in _perceptual_duplicate_pairs(
                records, perceptual_hash_values, max_hash_distance
            )
            if duplicate["crosses_sources"]
        ]
        if perceptual_hashes
        else [],
        exact=False,
    )
    return CrossSourceAuditReport(
        image_count=len(records),
        invalid_images=invalid_images,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )
