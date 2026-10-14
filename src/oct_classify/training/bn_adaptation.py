"""AdaBN: re-estimate BatchNorm statistics on unlabeled target images without changing weights."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from oct_classify.data.models import ImageRecord


def _group_key(record: ImageRecord) -> str:
    # Records without a group are their own unit, matching the analysis bootstrap.
    return record.group_key or f"{record.source}:path:{record.path}"


def split_groups(records: Iterable[ImageRecord], folds: int, seed: int) -> list[list[ImageRecord]]:
    """Assign whole patient groups to folds, stratified by each group's majority label.

    Adapting on one fold and predicting another keeps a patient's own images out of the statistics
    used to classify it.
    """
    if folds < 2:
        raise ValueError("BatchNorm adaptation needs at least two folds.")
    by_group: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_group.setdefault(_group_key(record), []).append(record)
    if len(by_group) < folds:
        raise ValueError(f"Cannot split {len(by_group)} groups into {folds} folds.")
    by_label: dict[str, list[str]] = {}
    for key in sorted(by_group):
        counts = Counter(record.label.value for record in by_group[key])
        majority = max(sorted(counts), key=counts.__getitem__)
        by_label.setdefault(majority, []).append(key)
    rng = random.Random(seed)
    assignment: list[list[ImageRecord]] = [[] for _ in range(folds)]
    offset = 0
    for label in sorted(by_label):
        keys = by_label[label]
        rng.shuffle(keys)
        for index, key in enumerate(keys):
            assignment[(offset + index) % folds].extend(by_group[key])
        offset += len(keys)
    return assignment


# Module-name prefixes of the BatchNorm layers that `--adapt-bn-layers early` adapts: the stem and
# first stage, which see low-level intensity and texture rather than class-specific features.
EARLY_LAYER_PREFIXES = {"resnet50": ("bn1", "layer1.")}


def adapt_batch_norm(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor, str, str]],
    device: torch.device,
    *,
    prefixes: tuple[str, ...] | None = None,
    max_batches: int | None = None,
) -> int:
    """Replace BatchNorm running statistics with a cumulative average over `loader`.

    `prefixes` limits adaptation to BatchNorm layers whose module names start with one of them;
    the others keep their trained statistics. Only running statistics change; learned weights are
    untouched and the model is left in eval mode. The loader should shuffle so that each batch is a
    random sample of the target: the cumulative average is over per-batch statistics, so ordered
    batches would drop between-batch variance. Returns the number of images seen.
    """
    norms = [
        module
        for name, module in model.named_modules()
        if isinstance(module, _BatchNorm) and (prefixes is None or name.startswith(prefixes))
    ]
    if not norms:
        raise ValueError("The model has no BatchNorm layers to adapt.")
    momenta = [norm.momentum for norm in norms]
    model.eval()
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None
        norm.train()
    seen = 0
    # fp32, like evaluation: fp16 overflowed to NaN logits/statistics on some checkpoints.
    with torch.no_grad():
        for batch_index, (images, *_) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            model(images.to(device, non_blocking=True))
            seen += len(images)
    for norm, momentum in zip(norms, momenta, strict=True):
        norm.momentum = momentum
    model.eval()
    if seen == 0:
        raise ValueError("No images were available for BatchNorm adaptation.")
    return seen
