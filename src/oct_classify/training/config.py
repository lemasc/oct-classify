from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str
    pretrained: bool


@dataclass(frozen=True, slots=True)
class DataConfig:
    image_size: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    horizontal_flip_probability: float
    rotation_degrees: float
    brightness_jitter: float
    contrast_jitter: float
    gaussian_blur_probability: float
    gaussian_blur_sigma: float


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    strategy: str = "source_class_balanced"
    batches_per_epoch: int = 1000


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    optimization: OptimizationConfig
    run: RunConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    try:
        config = TrainingConfig(
            model=ModelConfig(**value["model"]),
            data=DataConfig(**value["data"]),
            augmentation=AugmentationConfig(**value["augmentation"]),
                optimization=OptimizationConfig(**value["optimization"]),
                run=RunConfig(**value["run"]),
                sampling=SamplingConfig(**value.get("sampling", {})),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid training configuration in {path}") from error
    if not config.model.architecture:
        raise ValueError("model.architecture must not be empty.")
    if config.data.image_size <= 0 or config.data.batch_size <= 0:
        raise ValueError("data.image_size and data.batch_size must be positive.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must not be negative.")
    if not 0 <= config.augmentation.horizontal_flip_probability <= 1:
        raise ValueError("augmentation.horizontal_flip_probability must be between 0 and 1.")
    if config.augmentation.rotation_degrees < 0:
        raise ValueError("augmentation.rotation_degrees must not be negative.")
    if not 0 <= config.augmentation.brightness_jitter <= 1:
        raise ValueError("augmentation.brightness_jitter must be between 0 and 1.")
    if not 0 <= config.augmentation.contrast_jitter <= 1:
        raise ValueError("augmentation.contrast_jitter must be between 0 and 1.")
    if not 0 <= config.augmentation.gaussian_blur_probability <= 1:
        raise ValueError("augmentation.gaussian_blur_probability must be between 0 and 1.")
    if config.augmentation.gaussian_blur_sigma <= 0:
        raise ValueError("augmentation.gaussian_blur_sigma must be positive.")
    if config.optimization.epochs <= 0 or config.optimization.learning_rate <= 0:
        raise ValueError("optimization.epochs and optimization.learning_rate must be positive.")
    if config.optimization.weight_decay < 0:
        raise ValueError("optimization.weight_decay must not be negative.")
    if config.optimization.early_stopping_patience <= 0:
        raise ValueError("optimization.early_stopping_patience must be positive.")
    if config.sampling.strategy != "source_class_balanced":
        raise ValueError("sampling.strategy must be 'source_class_balanced'.")
    if config.sampling.batches_per_epoch <= 0:
        raise ValueError("sampling.batches_per_epoch must be positive.")
    return config
