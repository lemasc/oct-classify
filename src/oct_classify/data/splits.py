from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from oct_classify.data.models import ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel


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


@dataclass(frozen=True, slots=True)
class DerivedSplit:
    assignments: dict[str, str]
    linked_group_components: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class CrossValidationSplits:
    folds: tuple[DerivedSplit, ...]
    linked_group_components: tuple[tuple[str, ...], ...]


def _source_group_components(
    records: Iterable[ImageRecord], near_duplicates: Iterable[dict[str, object]]
) -> tuple[tuple[str, ...], ...]:
    """Link groups joined by retained pHash-distance-zero image pairs."""
    records = list(records)
    path_to_group = {
        f"{record.source}:{record.path}": record.group_key
        for record in records
        if record.group_key is not None
    }
    parent = {group: group for group in path_to_group.values()}

    def root(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for duplicate in near_duplicates:
        if duplicate.get("distance") != 0:
            continue
        groups = {
            path_to_group[path]
            for path in duplicate["paths"]  # type: ignore[index]
            if path in path_to_group
        }
        if len(groups) < 2:
            continue
        canonical = min(groups)
        for group in groups:
            first, second = root(canonical), root(group)
            if first != second:
                parent[second] = first

    components: dict[str, list[str]] = defaultdict(list)
    for group in parent:
        components[root(group)].append(group)
    return tuple(
        sorted(
            (tuple(sorted(component)) for component in components.values()),
            key=lambda component: component[0],
        )
    )


def _assign_units(
    units: list[tuple[tuple[str, ...], dict[UnifiedLabel, int]]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    if not ratios or any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Split ratios must be non-empty and positive.")
    total_ratio = sum(ratios.values())
    normalized = {split: ratio / total_ratio for split, ratio in ratios.items()}
    totals: dict[UnifiedLabel, int] = defaultdict(int)
    for _, counts in units:
        for label, count in counts.items():
            totals[label] += count
    targets = {
        split: {label: total * normalized[split] for label, total in totals.items()}
        for split in ratios
    }
    current = {split: defaultdict(int) for split in ratios}
    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda unit: sum(unit[1].values()), reverse=True)
    assignments: dict[str, str] = {}
    for groups, counts in shuffled:

        def score(split: str, unit_counts: dict[UnifiedLabel, int] = counts) -> float:
            score = 0.0
            for label, target in targets[split].items():
                before = current[split][label] - target
                after = before + unit_counts.get(label, 0)
                score += (after * after - before * before) / max(target, 1.0)
            return score

        split = min(ratios, key=score)
        for label, count in counts.items():
            current[split][label] += count
        assignments.update(dict.fromkeys(groups, split))
    return assignments


def create_derived_splits(
    records: Iterable[ImageRecord],
    near_duplicates: Iterable[dict[str, object]],
    ratios: dict[str, float],
    seed: int,
    *,
    preserve_supplied_test: bool = False,
) -> DerivedSplit:
    """Create group-safe, label-balanced assignments with pHash-zero linkage constraints."""
    records = list(records)
    if any(record.group_key is None for record in records):
        raise ValueError("Cannot create group-safe splits until every record has a group ID.")
    components = _source_group_components(records, near_duplicates)
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group_key].append(record)  # type: ignore[index]
    assignments: dict[str, str] = {}
    candidate_units: list[tuple[tuple[str, ...], dict[UnifiedLabel, int]]] = []
    for groups in components:
        unit_records = [record for group in groups for record in by_group[group]]
        supplied = {
            record.supplied_split for record in unit_records if record.supplied_split is not None
        }
        if preserve_supplied_test and "test" in supplied:
            if supplied != {"test"}:
                raise ValueError("A pHash-zero linked unit crosses Kermany supplied splits.")
            assignments.update(dict.fromkeys(groups, "test"))
            continue
        if preserve_supplied_test and supplied - {"train"}:
            raise ValueError("Kermany has an unexpected supplied split.")
        counts: dict[UnifiedLabel, int] = defaultdict(int)
        for record in unit_records:
            counts[record.label] += 1
        candidate_units.append((groups, counts))
    assignments.update(_assign_units(candidate_units, ratios, seed))
    return DerivedSplit(assignments, components)


def create_stratified_cross_validation_splits(
    records: Iterable[ImageRecord],
    near_duplicates: Iterable[dict[str, object]],
    *,
    folds: int,
    validation_groups_per_class: int,
    seed: int,
) -> CrossValidationSplits:
    """Create group-safe outer folds with a class-balanced inner validation set."""
    if folds < 2 or validation_groups_per_class < 1:
        raise ValueError("folds must be at least 2 and validation_groups_per_class must be positive.")
    records = list(records)
    if any(record.group_key is None for record in records):
        raise ValueError("Cannot create group-safe folds until every record has a group ID.")
    components = _source_group_components(records, near_duplicates)
    records_by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_group[record.group_key].append(record)  # type: ignore[index]

    units_by_label: dict[UnifiedLabel, list[tuple[str, ...]]] = defaultdict(list)
    for component in components:
        labels = {
            record.label
            for group in component
            for record in records_by_group[group]
        }
        if len(labels) != 1:
            raise ValueError("Cross-validation components must contain exactly one label.")
        units_by_label[labels.pop()].append(component)

    outer_groups: list[set[str]] = [set() for _ in range(folds)]
    for label, units in units_by_label.items():
        if len(units) % folds:
            raise ValueError(f"{label.value} groups cannot be evenly distributed across {folds} folds.")
        shuffled = list(units)
        random.Random(f"{seed}:{label.value}").shuffle(shuffled)
        for index, unit in enumerate(shuffled):
            outer_groups[index % folds].update(unit)

    all_groups = set(records_by_group)
    results: list[DerivedSplit] = []
    for fold_index, test_groups in enumerate(outer_groups):
        validation_groups: set[str] = set()
        for label, units in units_by_label.items():
            available = [unit for unit in units if not set(unit) & test_groups]
            random.Random(f"{seed}:{fold_index}:{label.value}").shuffle(available)
            selected: list[tuple[str, ...]] = []
            selected_count = 0
            for unit in available:
                if selected_count + len(unit) > validation_groups_per_class:
                    continue
                selected.append(unit)
                selected_count += len(unit)
                if selected_count == validation_groups_per_class:
                    break
            if selected_count != validation_groups_per_class:
                raise ValueError(
                    f"Cannot select {validation_groups_per_class} validation groups for {label.value}."
                )
            validation_groups.update(group for unit in selected for group in unit)
        assignments = {
            group: "test" if group in test_groups else "val" if group in validation_groups else "train"
            for group in all_groups
        }
        results.append(DerivedSplit(assignments, components))
    return CrossValidationSplits(tuple(results), components)


def apply_derived_splits(
    records: Iterable[ImageRecord], assignments: dict[str, str]
) -> list[ImageRecord]:
    output = []
    for record in records:
        if record.group_key not in assignments:
            raise ValueError(f"No split assignment for {record.group_key}.")
        output.append(
            ImageRecord(
                path=record.path,
                source=record.source,
                raw_label=record.raw_label,
                label=record.label,
            available_labels=record.available_labels,
            group_id=record.group_id,
            label_unit=record.label_unit,
            eye_id=record.eye_id,
            cohort=record.cohort,
            supplied_split=record.supplied_split,
                split=assignments[record.group_key],
            )
        )
    return output


def validate_derived_splits(
    records: Iterable[ImageRecord], linked_group_components: Iterable[tuple[str, ...]]
) -> list[str]:
    """Return violations of derived split exclusivity and pHash-zero linkage."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.group_key is None or record.split is None:
            return ["every record must have a group and derived split"]
        group_splits[record.group_key].add(record.split)
    failures = [
        f"group assigned to multiple splits: {group}"
        for group, splits in sorted(group_splits.items())
        if len(splits) > 1
    ]
    for component in linked_group_components:
        splits = {next(iter(group_splits[group])) for group in component}
        if len(splits) > 1:
            failures.append(f"pHash-zero linked groups cross splits: {', '.join(component)}")
    return failures


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split_definition(
    path: Path,
    *,
    seed: int,
    source_settings: dict[str, dict[str, object]],
    assignments: dict[str, str],
    manifest_hashes: dict[str, str],
    audit_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": seed,
                "source_settings": source_settings,
                "assignments": dict(sorted(assignments.items())),
                "manifest_sha256": manifest_hashes,
                "audit_sha256": audit_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_cross_validation_definition(
    path: Path,
    *,
    seed: int,
    source: str,
    folds: int,
    validation_groups_per_class: int,
    assignments: Iterable[dict[str, str]],
    manifest_hash: str,
    audit_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": source,
                "folds": folds,
                "seed": seed,
                "validation_groups_per_class": validation_groups_per_class,
                "assignments": [dict(sorted(fold.items())) for fold in assignments],
                "manifest_sha256": manifest_hash,
                "audit_sha256": audit_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
