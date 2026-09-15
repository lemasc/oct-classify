from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from oct_classify.data.models import ImageRecord


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    image_size: int = 224
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Intensity percentiles must satisfy 0 <= lower < upper <= 100.")


def preprocess_image(image: Image.Image, spec: PreprocessingSpec | None = None) -> np.ndarray:
    """Convert an OCT image to a robustly scaled, centered RGB square canvas."""
    spec = spec or PreprocessingSpec()
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")
    width, height = image.size
    scale = min(spec.image_size / width, spec.image_size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (spec.image_size, spec.image_size))
    offset = ((spec.image_size - resized.width) // 2, (spec.image_size - resized.height) // 2)
    canvas.paste(resized, offset)
    array = np.asarray(canvas, dtype=np.float32)
    low, high = np.percentile(array, [spec.lower_percentile, spec.upper_percentile])
    if high > low:
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
    else:
        array.fill(0.0)
    # Percentile arithmetic may promote the float32 pixel array to float64.
    return array.astype(np.float32, copy=False)


def channel_statistics(images: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate channel-wise statistics from training images after deterministic preprocessing."""
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for image in images:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected HWC RGB images.")
        flattened = image.reshape(-1, 3)
        total += flattened.sum(axis=0)
        total_squared += np.square(flattened).sum(axis=0)
        pixel_count += len(flattened)
    if pixel_count == 0:
        raise ValueError("Cannot calculate statistics from no images.")
    mean = total / pixel_count
    variance = np.maximum(total_squared / pixel_count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _preprocess_for_normalization(job: tuple[Path, str, PreprocessingSpec]) -> np.ndarray:
    root, path, spec = job
    with Image.open(root / path) as image:
        return preprocess_image(image, spec)


def _partial_statistics_for_normalization(
    job: tuple[Path, str, PreprocessingSpec],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce one image to its channel sums in the worker process.

    Returning only three floats per channel (instead of the full decoded
    image) keeps inter-process messages tiny regardless of dataset size --
    shipping whole preprocessed images back to the main process instead
    scales its memory use with the number of images processed.
    """
    flattened = _preprocess_for_normalization(job).reshape(-1, 3)
    total = flattened.sum(axis=0, dtype=np.float64)
    total_squared = np.square(flattened, dtype=np.float64).sum(axis=0)
    return total, total_squared, flattened.shape[0]


def _combine_partial_statistics(
    partials: Iterable[tuple[np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for partial_total, partial_total_squared, count in partials:
        total += partial_total
        total_squared += partial_total_squared
        pixel_count += count
    if pixel_count == 0:
        raise ValueError("Cannot calculate statistics from no images.")
    mean = total / pixel_count
    variance = np.maximum(total_squared / pixel_count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def calculate_normalization(
    root: Path | Mapping[str, Path],
    records: Iterable[ImageRecord],
    spec: PreprocessingSpec,
    *,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute channel statistics in worker processes that stay free of torch/torchvision.

    The worker function lives here (not in training.dataset) so that spawned
    processes only import this lightweight, torch-free module instead of
    pulling in the full torch/CUDA stack for plain PIL/numpy work.
    """
    if workers is not None and workers < 1:
        raise ValueError("The normalization worker count must be at least one.")
    records = list(records)
    if isinstance(root, Path):
        jobs = [(root, record.path, spec) for record in records]
    else:
        missing_sources = {record.source for record in records} - root.keys()
        if missing_sources:
            raise ValueError(f"No normalization root for sources: {sorted(missing_sources)}")
        jobs = [(root[record.source], record.path, spec) for record in records]
    if workers == 1 or len(jobs) < 2:
        return channel_statistics(_preprocess_for_normalization(job) for job in jobs)

    worker_count = workers or os.cpu_count() or 1
    chunksize = max(1, min(64, len(jobs) // (worker_count * 4)))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return _combine_partial_statistics(
            executor.map(_partial_statistics_for_normalization, jobs, chunksize=chunksize)
        )
