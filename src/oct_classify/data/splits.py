from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from oct_classify.data.models import ImageRecord


@dataclass(frozen=True, slots=True)
class SplitValidation:
    duplicate_paths: tuple[str, ...]
    groups_in_multiple_splits: dict[str, tuple[str, ...]]
    records_without_group: int

    @property
    def is_leakage_safe(self) -> bool:
        return (
            not self.duplicate_paths
            and not self.groups_in_multiple_splits
            and self.records_without_group == 0
        )


def validate_supplied_splits(records: Iterable[ImageRecord]) -> SplitValidation:
    seen_paths: set[str] = set()
    duplicates: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    missing_groups = 0

    for record in records:
        if record.path in seen_paths:
            duplicates.add(record.path)
        seen_paths.add(record.path)
        if record.group_id is None:
            missing_groups += 1
        elif record.supplied_split is not None:
            group_splits[record.group_key].add(record.supplied_split)

    leaked_groups = {
        group: tuple(sorted(splits)) for group, splits in group_splits.items() if len(splits) > 1
    }
    return SplitValidation(
        duplicate_paths=tuple(sorted(duplicates)),
        groups_in_multiple_splits=leaked_groups,
        records_without_group=missing_groups,
    )


def assign_group_splits(
    records: Iterable[ImageRecord],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each known group to exactly one split using deterministic randomization."""
    if not ratios or any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Split ratios must be non-empty and positive.")

    groups = {record.group_key for record in records}
    if None in groups:
        raise ValueError("Cannot create group-safe splits until every record has a group ID.")

    group_ids = sorted(group for group in groups if group is not None)
    random.Random(seed).shuffle(group_ids)
    boundaries: list[tuple[str, float]] = []
    cumulative = 0.0
    total = sum(ratios.values())
    for split, ratio in ratios.items():
        cumulative += ratio / total
        boundaries.append((split, cumulative))
    boundaries[-1] = (boundaries[-1][0], 1.0)

    assignments: dict[str, str] = {}
    for index, group_id in enumerate(group_ids):
        position = (index + 1) / len(group_ids)
        assignments[group_id] = next(
            split for split, boundary in boundaries if position <= boundary
        )
    return assignments
