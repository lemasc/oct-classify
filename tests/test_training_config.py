from pathlib import Path

import pytest

from oct_classify.training.config import load_training_config


def test_load_training_config_reads_resnet_baseline() -> None:
    config = load_training_config(Path("configs/training/resnet50.toml"))

    assert config.model.architecture == "resnet50"
    assert config.model.pretrained
    assert config.data.image_size == 224


def test_load_training_config_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        """
[model]
architecture = "resnet50"
pretrained = true
[data]
image_size = 224
batch_size = 0
num_workers = 0
[optimization]
epochs = 1
learning_rate = 0.001
weight_decay = 0.0
[run]
seed = 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size"):
        load_training_config(path)
