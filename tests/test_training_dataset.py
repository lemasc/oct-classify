from pathlib import Path

import numpy as np
from PIL import Image

from oct_classify.data.models import ImageRecord
from oct_classify.data.preprocessing import PreprocessingSpec
from oct_classify.data.taxonomy import ALL_LABELS, UnifiedLabel
from oct_classify.training.config import AugmentationConfig
from oct_classify.training.dataset import ManifestImageDataset, labels_for_available


def _record(label: UnifiedLabel) -> ImageRecord:
    return ImageRecord(
        path="image.png",
        source="test",
        raw_label=label.value,
        label=label,
        available_labels=ALL_LABELS,
        split="train",
    )


def test_labels_for_available_preserves_canonical_order() -> None:
    assert labels_for_available({UnifiedLabel.DME, UnifiedLabel.NORMAL}) == (
        UnifiedLabel.NORMAL,
        UnifiedLabel.DME,
    )


def test_dataset_filters_to_requested_labels_and_normalizes(tmp_path: Path) -> None:
    Image.new("L", (4, 2), color=100).save(tmp_path / "image.png")
    dataset = ManifestImageDataset(
        tmp_path,
        [_record(UnifiedLabel.NORMAL), _record(UnifiedLabel.DME)],
        (UnifiedLabel.NORMAL, UnifiedLabel.DME),
        PreprocessingSpec(image_size=4, lower_percentile=0, upper_percentile=100),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        AugmentationConfig(0.5, 10.0, 0.1, 0.1, 0.25, 0.5),
        training=False,
    )

    image, target, available_mask, source, path = dataset[1]

    assert image.shape == (3, 4, 4)
    assert target == 1
    assert available_mask.tolist() == [True, True]
    assert source == "test"
    assert path == "image.png"


def test_dataset_rotation_fills_corners_with_median_padding(tmp_path: Path) -> None:
    image = Image.new("L", (16, 8), color=200)
    image.paste(100, (0, 0, 16, 2))
    image.save(tmp_path / "image.png")
    dataset = ManifestImageDataset(
        tmp_path,
        [_record(UnifiedLabel.NORMAL)],
        (UnifiedLabel.NORMAL, UnifiedLabel.DME),
        PreprocessingSpec(
            image_size=16, lower_percentile=0, upper_percentile=100, padding="median"
        ),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        AugmentationConfig(0.0, 45.0, 0.0, 0.0, 0.0, 0.5),
        training=True,
    )

    image_tensor, *_ = dataset[0]

    # The content median is 1, so rotated-in corners must be 1 rather than black 0.
    corners = image_tensor[:, [0, 0, -1, -1], [0, -1, 0, -1]]
    assert np.allclose(corners.numpy(), 1.0)
