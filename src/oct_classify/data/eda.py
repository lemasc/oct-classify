"""Per-image exploratory features for the derived manifests (no torch).

The features describe acquisition artifacts that a classifier can exploit as shortcuts: canvas
geometry, saturated (white) and empty (black) fill regions connected to the image border, and
the intensity statistics that the percentile normalization in `preprocessing` is sensitive to.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from oct_classify.data.fill import fill_masks
from oct_classify.data.models import ImageRecord

FULL_ROW_FRACTION = 0.95


def _leading_full(profile: np.ndarray) -> int:
    """Length of the leading run where the profile (fraction of fill per row/col) is ~full."""
    below = np.flatnonzero(profile < FULL_ROW_FRACTION)
    return int(len(profile) if below.size == 0 else below[0])


def _edge_runs(mask: np.ndarray) -> dict[str, int]:
    rows, cols = mask.mean(axis=1), mask.mean(axis=0)
    return {
        "top": _leading_full(rows),
        "bottom": _leading_full(rows[::-1]),
        "left": _leading_full(cols),
        "right": _leading_full(cols[::-1]),
    }


def image_features(image: Image.Image) -> dict[str, float | int | str]:
    """Describe one decoded image at its native resolution."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = rgb.mean(axis=2)
    white, black = fill_masks(rgb)
    white_runs, black_runs = _edge_runs(white), _edge_runs(black)
    content = gray[~(white | black)]
    if content.size == 0:
        content = gray.ravel()
    p1_all, p99_all = np.percentile(gray, [1, 99])
    p1_content, p50_content, p99_content = np.percentile(content, [1, 50, 99])
    features: dict[str, float | int | str] = {
        "mode": image.mode,
        "width": width,
        "height": height,
        "aspect": width / height,
        "white_frac": float(white.mean()),
        "black_frac": float(black.mean()),
        "gray_mean": float(gray.mean()),
        "p1_all": float(p1_all),
        "p99_all": float(p99_all),
        "p1_content": float(p1_content),
        "p50_content": float(p50_content),
        "p99_content": float(p99_content),
        "channel_spread": float(np.abs(rgb[..., 0].astype(int) - rgb[..., 2]).mean()),
    }
    for edge in ("top", "bottom", "left", "right"):
        features[f"white_{edge}"] = white_runs[edge]
        features[f"black_{edge}"] = black_runs[edge]
        # Border-connected fill touching an edge at all, even as a wedge rather than a full band.
        features[f"white_touch_{edge}"] = bool(
            {"top": white[0], "bottom": white[-1], "left": white[:, 0], "right": white[:, -1]}[
                edge
            ].any()
        )
    return features


def _features_for_job(job: tuple[Path, str]) -> dict[str, float | int | str]:
    root, path = job
    with Image.open(root / path) as image:
        return image_features(image)


def collect_features(
    roots: dict[str, Path], records: Sequence[ImageRecord], *, workers: int | None = None
) -> list[dict[str, float | int | str]]:
    """Features for each record, in record order, keyed by the record's own source root."""
    jobs = [(roots[record.source], record.path) for record in records]
    worker_count = workers or os.cpu_count() or 1
    if worker_count == 1 or len(jobs) < 2:
        return [_features_for_job(job) for job in jobs]
    chunksize = max(1, min(64, len(jobs) // (worker_count * 4)))
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return list(executor.map(_features_for_job, jobs, chunksize=chunksize))


def static_overlay_maps(
    images: Iterable[np.ndarray], *, bright: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel frequency of being bright, and the per-pixel intensity std, over same-size images.

    A fixed overlay (scale bar, watermark) is bright in nearly every image at the same pixels, so
    it shows up as frequency ~1 with a std far below that of the anatomy around it.
    """
    count = 0
    bright_count = total = total_squared = None
    for image in images:
        gray = image.astype(np.float64)
        if bright_count is None:
            bright_count = np.zeros_like(gray)
            total = np.zeros_like(gray)
            total_squared = np.zeros_like(gray)
        bright_count += gray >= bright
        total += gray
        total_squared += gray**2
        count += 1
    if count == 0:
        raise ValueError("No images supplied.")
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - mean**2, 0.0))
    return bright_count / count, std
