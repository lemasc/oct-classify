import numpy as np
from PIL import Image

from oct_classify.data.eda import image_features, static_overlay_maps
from oct_classify.data.fill import fill_masks


def _speckle(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 120, size=(height, width, 3), dtype=np.uint8)


def test_white_band_on_top_is_border_connected_fill():
    pixels = _speckle(100, 120)
    pixels[:20] = 255
    features = image_features(Image.fromarray(pixels))
    assert features["white_top"] == 20
    assert features["white_touch_top"] and not features["white_touch_bottom"]
    assert np.isclose(features["white_frac"], 0.2)
    # Content percentiles ignore the fill that saturates the whole-image upper percentile.
    assert features["p99_all"] == 255 and features["p99_content"] < 130


def test_interior_bright_blob_and_speckle_are_not_fill():
    pixels = _speckle(100, 120)
    pixels[40:60, 40:60] = 255  # touches no border
    pixels[0, 0] = 255  # border-touching but far below the minimum fill area
    white, black = fill_masks(pixels)
    assert not white.any() and not black.any()


def test_black_wedge_is_detected_separately():
    pixels = _speckle(100, 120)
    pixels[-30:, :40] = 0
    white, black = fill_masks(pixels)
    assert black.sum() == 30 * 40 and not white.any()


def test_static_overlay_shows_up_as_always_bright_and_zero_variance():
    images = []
    for seed in range(8):
        pixels = _speckle(50, 60, seed)[..., 0]
        pixels[40:48, 2:10] = 255
        images.append(pixels)
    frequency, std = static_overlay_maps(images)
    assert (frequency[40:48, 2:10] == 1).all() and (std[40:48, 2:10] == 0).all()
    assert frequency[:30].max() < 1


def test_fill_aware_removes_fill_and_matches_background():
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image

    pixels = _speckle(100, 120)
    pixels[:20] = 255
    pixels[20:23] = 248  # JPEG-style edge just under the saturation threshold
    spec = PreprocessingSpec(image_size=64, padding="fill_aware")
    canvas = preprocess_image(Image.fromarray(pixels), spec)
    assert canvas.shape == (64, 64, 3) and canvas.max() <= 1.0
    # No bright residue from the fill: the top of the image is as dark as tissue background.
    assert canvas[(64 - round(100 * 64 / 120)) // 2 :][:6].mean() < 0.6
    assert np.array_equal(canvas, preprocess_image(Image.fromarray(pixels), spec))


def test_fill_aware_without_fill_equals_median_padding():
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image

    image = Image.fromarray(_speckle(64, 80))
    fill_aware = preprocess_image(image, PreprocessingSpec(image_size=32, padding="fill_aware"))
    median = preprocess_image(image, PreprocessingSpec(image_size=32, padding="median"))
    assert np.array_equal(fill_aware, median)


def test_fill_aware_leaves_black_background_alone():
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image

    pixels = _speckle(100, 120)
    pixels[-40:] = 0  # genuine empty background, e.g. PAIMA's zeroed deep layers
    image = Image.fromarray(pixels)
    spec = {"image_size": 32}
    assert np.array_equal(
        preprocess_image(image, PreprocessingSpec(padding="fill_aware", **spec)),
        preprocess_image(image, PreprocessingSpec(padding="median", **spec)),
    )
