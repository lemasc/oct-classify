import numpy as np
import pytest
from PIL import Image

from oct_classify.data.preprocessing import PreprocessingSpec, channel_statistics, preprocess_image


def test_preprocess_preserves_aspect_ratio_with_center_padding() -> None:
    image = Image.new("L", (8, 4), color=100)

    result = preprocess_image(
        image, PreprocessingSpec(image_size=8, lower_percentile=0, upper_percentile=100)
    )

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.float32
    assert np.all(result[:2] == 0)
    assert np.all(result[2:6] == 1)
    assert np.all(result[6:] == 0)


def test_preprocess_median_padding_fills_with_content_median() -> None:
    image = Image.new("L", (8, 4), color=200)
    image.paste(100, (0, 0, 8, 1))

    result = preprocess_image(
        image,
        PreprocessingSpec(image_size=8, lower_percentile=0, upper_percentile=100, padding="median"),
    )

    # Percentiles come from the content alone (100 -> 0, 200 -> 1), not the black canvas.
    assert result.dtype == np.float32
    assert np.all(result[2] == 0)
    assert np.all(result[3:6] == 1)
    # The content median (1) fills the padding instead of 0.
    assert np.all(result[:2] == 1)
    assert np.all(result[6:] == 1)


def test_preprocess_median_padding_is_independent_of_aspect_ratio() -> None:
    rng = np.random.default_rng(0)
    content = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    square = Image.fromarray(content)
    wide = Image.fromarray(np.tile(content, (1, 2)))
    spec = PreprocessingSpec(image_size=32, padding="median")

    square_result = preprocess_image(square, spec)
    wide_result = preprocess_image(wide, spec)

    assert np.allclose(wide_result[:8], np.median(wide_result[8:24].reshape(-1, 3), axis=0))
    assert np.isclose(np.median(square_result), np.median(wide_result), atol=0.02)


def test_preprocessing_spec_rejects_unknown_padding() -> None:
    with pytest.raises(ValueError, match="padding"):
        PreprocessingSpec(padding="edge")


def test_preprocess_flattens_a_transparent_pixel_against_black() -> None:
    image = Image.new("RGBA", (1, 1), color=(255, 255, 255, 0))

    result = preprocess_image(image, PreprocessingSpec(image_size=1))

    assert np.array_equal(result, np.zeros((1, 1, 3), dtype=np.float32))


def test_channel_statistics_calculates_training_image_moments() -> None:
    first = np.zeros((1, 1, 3), dtype=np.float32)
    second = np.ones((1, 1, 3), dtype=np.float32)

    mean, stdev = channel_statistics([first, second])

    assert np.array_equal(mean, np.full(3, 0.5, dtype=np.float32))
    assert np.allclose(stdev, np.full(3, 0.5, dtype=np.float32))


def test_preprocessing_spec_rejects_invalid_percentiles() -> None:
    with pytest.raises(ValueError, match="percentiles"):
        PreprocessingSpec(lower_percentile=99, upper_percentile=1)
