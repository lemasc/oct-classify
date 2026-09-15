from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy restricted to labels screened by each sample's source."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, available_masks: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, classes].")
        if targets.shape != (len(logits),):
            raise ValueError("targets must have shape [batch].")
        if available_masks.shape != logits.shape or available_masks.dtype != torch.bool:
            raise ValueError("available_masks must be a boolean tensor matching logits.")
        if not torch.all(available_masks.any(dim=1)):
            raise ValueError("Every sample must have at least one available label.")
        if torch.any(targets < 0) or torch.any(targets >= logits.shape[1]):
            raise ValueError("targets contain an out-of-range class index.")
        if not torch.all(available_masks.gather(1, targets[:, None])):
            raise ValueError("A target label is unavailable for its sample.")

        masked_logits = logits.masked_fill(~available_masks, torch.finfo(logits.dtype).min)
        losses = functional.cross_entropy(masked_logits, targets, reduction="none")
        if self.weight is None:
            return losses.mean()
        sample_weights = self.weight[targets]
        return (losses * sample_weights).sum() / sample_weights.sum()
