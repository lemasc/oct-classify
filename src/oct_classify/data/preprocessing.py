from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image


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
