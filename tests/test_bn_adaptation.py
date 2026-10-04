import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oct_classify.data.models import ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.training.bn_adaptation import adapt_batch_norm, split_groups

LABELS = frozenset({UnifiedLabel.NORMAL, UnifiedLabel.AMD})


def _record(group: str, index: int, label: UnifiedLabel) -> ImageRecord:
    return ImageRecord(
        path=f"{group}/{index}.png",
        source="paima",
        raw_label=label.value,
        label=label,
        available_labels=LABELS,
        group_id=group,
    )


def _records() -> list[ImageRecord]:
    records = []
    for group in range(10):
        label = UnifiedLabel.NORMAL if group < 6 else UnifiedLabel.AMD
        records.extend(_record(f"p{group}", index, label) for index in range(group % 3 + 1))
    return records


def test_split_groups_keeps_groups_whole_and_stratifies_labels() -> None:
    folds = split_groups(_records(), 2, seed=1)

    groups = [{record.group_id for record in fold} for fold in folds]
    assert groups[0].isdisjoint(groups[1])
    assert sum(len(fold) for fold in folds) == len(_records())
    for fold in folds:
        labels = {record.group_id: record.label for record in fold}
        assert sum(label is UnifiedLabel.NORMAL for label in labels.values()) == 3
        assert sum(label is UnifiedLabel.AMD for label in labels.values()) == 2


def test_split_groups_is_deterministic_for_a_seed() -> None:
    first = [[record.path for record in fold] for fold in split_groups(_records(), 2, seed=7)]
    second = [[record.path for record in fold] for fold in split_groups(_records(), 2, seed=7)]

    assert first == second


def test_split_groups_rejects_more_folds_than_groups() -> None:
    try:
        split_groups(_records()[:1], 2, seed=0)
    except ValueError as error:
        assert "groups" in str(error)
    else:
        raise AssertionError("Expected too few groups to be rejected")


def test_adapt_batch_norm_matches_target_statistics_without_changing_weights() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
    norm = model[1]
    weights = {name: value.clone() for name, value in model.named_parameters()}
    images = torch.randn(64, 4) * 3 + 5
    loader = DataLoader(TensorDataset(images, torch.zeros(64)), batch_size=64)

    seen = adapt_batch_norm(model, loader, torch.device("cpu"))

    features = model[0](images).detach()
    assert seen == 64
    assert torch.allclose(norm.running_mean, features.mean(dim=0), atol=1e-5)
    assert torch.allclose(norm.running_var, features.var(dim=0), atol=1e-4)
    assert norm.momentum == 0.1
    assert not model.training
    for name, value in model.named_parameters():
        assert torch.equal(value, weights[name])


def test_adapt_batch_norm_rejects_model_without_batch_norm() -> None:
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4)), batch_size=2)

    try:
        adapt_batch_norm(nn.Linear(2, 2), loader, torch.device("cpu"))
    except ValueError as error:
        assert "BatchNorm" in str(error)
    else:
        raise AssertionError("Expected a model without BatchNorm to be rejected")


def test_adapt_batch_norm_prefixes_leave_other_layers_untouched() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.Linear(3, 3), nn.BatchNorm1d(3))
    late_mean = model[3].running_mean.clone()
    loader = DataLoader(TensorDataset(torch.randn(32, 4) + 5, torch.zeros(32)), batch_size=32)

    adapt_batch_norm(model, loader, torch.device("cpu"), prefixes=("1",))

    assert not torch.allclose(model[1].running_mean, torch.zeros(3))
    assert torch.equal(model[3].running_mean, late_mean)
