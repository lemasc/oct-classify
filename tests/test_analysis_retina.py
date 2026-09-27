import numpy as np
from PIL import Image

from oct_classify.analysis.retina import (
    canvas_box,
    content_mask,
    fill_mask,
    mass_fraction,
    retina_mask,
)
from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image


def test_content_mask_matches_preprocessing_letterbox() -> None:
    image = Image.new("RGB", (400, 200), (200, 200, 200))
    image.paste((100, 100, 100), (0, 0, 200, 200))
    array = preprocess_image(image, PreprocessingSpec(image_size=64))

    mask = content_mask(400, 200, 64)

    assert canvas_box(400, 200, 64) == (0, 16, 64, 32)
    assert mask.sum() == 64 * 32
    assert np.all(array[~mask] == 0)
    assert np.all(array[mask].max(axis=1) > 0)


def _synthetic_bscan(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.normal(0.08, 0.03, (size, size))
    image[50:80] += 0.45  # retina
    image[76:80] += 0.4  # RPE
    return np.clip(image, 0, 1)


def test_retina_mask_finds_the_bright_band() -> None:
    image = _synthetic_bscan()
    content = np.ones_like(image, dtype=bool)

    mask = retina_mask(image, content)

    rows = np.flatnonzero(mask.any(axis=1))
    assert 40 <= rows.min() <= 52
    assert 78 <= rows.max() <= 96
    assert mask[:30].sum() == 0


def test_fill_mask_removes_white_wedge_on_top_edge() -> None:
    image = _synthetic_bscan()
    for column in range(128):
        image[: max(0, 20 - column // 6), column] = 1.0
    content = np.ones_like(image, dtype=bool)

    fill = fill_mask(image, content)
    mask = retina_mask(image, content & ~fill)

    assert fill[0, 0] and not fill[60, 60]
    assert mask[:30].sum() == 0


def test_mass_fraction_is_undefined_for_empty_map() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True

    assert mass_fraction(np.zeros((4, 4)), mask) is None
    assert mass_fraction(np.ones((4, 4)), mask) == 0.5
