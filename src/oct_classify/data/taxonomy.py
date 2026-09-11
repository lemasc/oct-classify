from __future__ import annotations

from enum import StrEnum


class UnifiedLabel(StrEnum):
    NORMAL = "normal"
    AMD = "amd"
    DME = "dme"


ALL_LABELS = frozenset(UnifiedLabel)


def parse_labels(values: list[str]) -> frozenset[UnifiedLabel]:
    labels = frozenset(UnifiedLabel(value.lower()) for value in values)
    if not labels:
        raise ValueError("A dataset must screen for at least one unified label.")
    return labels
