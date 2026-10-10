import random

import numpy as np
from PIL import Image

from oct_classify.data.geometry import MAX_ASPECT, apply_geometry, random_mirror_pad, square_crop


def _gradient(width: int, height: int) -> Image.Image:
    return Image.fromarray(np.tile(np.arange(width, dtype=np.uint8), (height, 1)))


def test_mirror_pad_widens_within_range_and_keeps_tissue() -> None:
    image = _gradient(40, 20)
    rng = random.Random(0)
    widths = set()
    for _ in range(50):
        padded = random_mirror_pad(image, rng)
        widths.add(padded.width)
        assert padded.height == 20
        assert 40 <= padded.width <= MAX_ASPECT * 20 + 1
        array = np.asarray(padded)[:, :, 0]
        # The original columns appear unchanged somewhere in the padded image.
        assert any((array[:, i : i + 40] == np.asarray(image)).all() for i in range(padded.width - 39))
    assert len(widths) > 1


def test_mirror_pad_leaves_images_already_wider_than_the_limit() -> None:
    image = _gradient(100, 20)
    assert random_mirror_pad(image, random.Random(0)).size == (100, 20)


def test_square_crop_is_centered_without_rng_and_random_with_one() -> None:
    image = _gradient(40, 20)
    assert square_crop(image).size == (20, 20)
    assert np.asarray(square_crop(image))[0, 0] == 10
    lefts = {np.asarray(square_crop(image, random.Random(seed)))[0, 0] for seed in range(20)}
    assert len(lefts) > 1


def test_apply_geometry_changes_training_only_except_square_crop() -> None:
    image = _gradient(40, 20)
    assert apply_geometry(image, "none", training=True) is image
    assert apply_geometry(image, "mirror_pad", training=False) is image
    assert apply_geometry(image, "square_crop", training=False).size == (20, 20)
