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
    assert np.all(result[:2] == 0)
    assert np.all(result[2:6] == 1)
    assert np.all(result[6:] == 0)


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
