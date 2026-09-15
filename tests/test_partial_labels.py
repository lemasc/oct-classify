import torch
from torch.nn import functional

from oct_classify.training.partial_labels import MaskedCrossEntropyLoss


def test_masked_loss_matches_cross_entropy_when_all_labels_are_available() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]])
    targets = torch.tensor([1, 2])

    actual = MaskedCrossEntropyLoss()(logits, targets, torch.ones_like(logits, dtype=torch.bool))

    assert torch.allclose(actual, functional.cross_entropy(logits, targets))


def test_unavailable_logit_has_no_gradient() -> None:
    logits = torch.tensor([[0.2, 0.7, 3.0]], requires_grad=True)
    targets = torch.tensor([1])
    available = torch.tensor([[True, True, False]])

    MaskedCrossEntropyLoss()(logits, targets, available).backward()

    assert logits.grad is not None
    assert logits.grad[0, 2] == 0


def test_masked_loss_rejects_unavailable_target() -> None:
    logits = torch.tensor([[0.2, 0.7, 3.0]])
    targets = torch.tensor([2])
    available = torch.tensor([[True, True, False]])

    try:
        MaskedCrossEntropyLoss()(logits, targets, available)
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("Expected unavailable target to be rejected")
