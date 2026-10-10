"""Training-time geometry changes that hide image shape (exp. 3a of the cross-device plan).

Duke's image size predicts its class, and models trained on it read the size as a label cue. Both changes act on the
raw image, before `preprocess_image`, so letterboxing and intensity scaling see the changed image.
"""

from __future__ import annotations

import random
from typing import Literal, Protocol

import numpy as np
from PIL import Image

Geometry = Literal["none", "mirror_pad", "square_crop"]
GEOMETRIES = ("none", "mirror_pad", "square_crop")

# The widest native aspect among Duke, Kermany and PAIMA is 1536x496 (3.1:1); mirror padding goes up to about that.
MAX_ASPECT = 3.2


class _Random(Protocol):
    def uniform(self, a: float, b: float) -> float: ...

    def randint(self, a: int, b: int) -> int: ...


def _as_rgb(image: Image.Image) -> Image.Image:
    # Same conversion as `preprocess_image`, so the pixels it sees are unchanged.
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    return image.convert("RGB")


def random_mirror_pad(image: Image.Image, rng: _Random = random) -> Image.Image:
    """Extend the tissue sideways by mirroring it to a random width between its own and `MAX_ASPECT` times its height."""
    image = _as_rgb(image)
    width, height = image.size
    extra = round(rng.uniform(width, max(width, MAX_ASPECT * height))) - width
    if extra <= 0:
        return image
    left = rng.randint(0, extra)
    padded = np.pad(np.asarray(image), [(0, 0), (left, extra - left), (0, 0)], mode="reflect")
    return Image.fromarray(padded)


def square_crop(image: Image.Image, rng: _Random | None = None) -> Image.Image:
    """Crop a `min(width, height)` square: a random window with `rng`, the center one without."""
    width, height = image.size
    side = min(width, height)
    if rng is None:
        left, top = (width - side) // 2, (height - side) // 2
    else:
        left, top = rng.randint(0, width - side), rng.randint(0, height - side)
    return image.crop((left, top, left + side, top + side))


def apply_geometry(image: Image.Image, geometry: str, *, training: bool, rng: _Random = random) -> Image.Image:
    """Mirror padding changes training images only. Square crops are random in training and centered otherwise."""
    if geometry == "mirror_pad":
        return random_mirror_pad(image, rng) if training else image
    if geometry == "square_crop":
        return square_crop(image, rng if training else None)
    return image
