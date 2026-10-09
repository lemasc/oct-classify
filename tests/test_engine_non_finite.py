import json
from pathlib import Path

import pytest
import torch
from torch import nn

from oct_classify.training.cli import _record_non_finite_failure
from oct_classify.training.engine import NonFiniteLossError, run_epoch, run_source_evaluations
from oct_classify.training.partial_labels import MaskedCrossEntropyLoss


class _PoisonedModel(nn.Module):
    """Returns NaN logits for any image whose first pixel is negative."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.linear(images.flatten(1)[:, :1])
        return torch.where(images.flatten(1)[:, :1] < 0, torch.full_like(logits, float("nan")), logits)


def _loader(values: list[float]):
    images = torch.tensor(values).view(-1, 1, 1, 1)
    batch = (
        images,
        torch.zeros(len(values), dtype=torch.long),
        torch.ones(len(values), 3, dtype=torch.bool),
        ["paima"] * len(values),
        [f"img{index}.tif" for index in range(len(values))],
    )
    return [batch]


def test_run_epoch_raises_with_offending_paths() -> None:
    with pytest.raises(NonFiniteLossError) as raised:
        run_epoch(_PoisonedModel(), _loader([1.0, -1.0]), MaskedCrossEntropyLoss(), torch.device("cpu"), ("a", "b", "c"))
    assert raised.value.non_finite_paths == ["img1.tif"]
    assert not raised.value.training


def test_source_evaluation_names_source_and_failure_dump_is_valid_json(tmp_path: Path) -> None:
    model = _PoisonedModel()
    with pytest.raises(NonFiniteLossError) as raised:
        run_source_evaluations(
            model, {"paima": _loader([1.0, -1.0])}, MaskedCrossEntropyLoss(), torch.device("cpu"),
            ("a", "b", "c"), {"paima": (0, 1, 2)},
        )
    assert raised.value.source == "paima"
    optimizer = torch.optim.AdamW(model.parameters())
    _record_non_finite_failure(tmp_path, raised.value, model, optimizer, 3, 0.9, 0, {})
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["phase"] == "validation" and failure["source"] == "paima"
    assert (tmp_path / "checkpoint-failed.pt").exists()
    assert "nan" in failure["batch"][1]["logits"]
