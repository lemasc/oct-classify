from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
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
class OptimizationConfig:
    epochs: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: ModelConfig
    data: DataConfig
    optimization: OptimizationConfig
    run: RunConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    try:
        config = TrainingConfig(
            model=ModelConfig(**value["model"]),
            data=DataConfig(**value["data"]),
            optimization=OptimizationConfig(**value["optimization"]),
            run=RunConfig(**value["run"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid training configuration in {path}") from error
    if not config.model.architecture:
        raise ValueError("model.architecture must not be empty.")
    if config.data.image_size <= 0 or config.data.batch_size <= 0:
        raise ValueError("data.image_size and data.batch_size must be positive.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must not be negative.")
    if config.optimization.epochs <= 0 or config.optimization.learning_rate <= 0:
        raise ValueError("optimization.epochs and optimization.learning_rate must be positive.")
    if config.optimization.weight_decay < 0:
        raise ValueError("optimization.weight_decay must not be negative.")
    return config
