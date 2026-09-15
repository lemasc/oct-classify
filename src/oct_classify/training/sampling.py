from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class SourceClassBalancedBatchSampler(Sampler[list[int]]):
    """Draw equal source quotas and uniform available classes within each source."""

    def __init__(
        self,
        sources: Sequence[str],
        targets: Sequence[int],
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or batches_per_epoch <= 0:
            raise ValueError("batch_size and batches_per_epoch must be positive.")
        if len(sources) != len(targets) or not sources:
            raise ValueError("sources and targets must be non-empty and have equal length.")
        self.source_names = tuple(sorted(set(sources)))
        if batch_size % len(self.source_names):
            raise ValueError("batch_size must divide evenly across selected sources.")
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        indices: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            indices[source][target].append(index)
        self.indices = {source: dict(by_class) for source, by_class in indices.items()}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        quota = self.batch_size // len(self.source_names)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for source in self.source_names:
                by_class = self.indices[source]
                classes = tuple(sorted(by_class))
                for _ in range(quota):
                    label = generator.choice(classes)
                    batch.append(generator.choice(by_class[label]))
            generator.shuffle(batch)
            yield batch
