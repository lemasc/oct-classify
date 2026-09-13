from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oct_classify.data.taxonomy import UnifiedLabel


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    source: str
    root: Path
    available_labels: frozenset[UnifiedLabel]
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """A source-independent description of one image in a dataset manifest."""

    path: str
    source: str
    raw_label: str
    label: UnifiedLabel
    available_labels: frozenset[UnifiedLabel]
    group_id: str | None = None
    supplied_split: str | None = None

    def __post_init__(self) -> None:
        if self.label not in self.available_labels:
            raise ValueError(
                f"{self.source} record has label {self.label.value!r}, which is not available "
                "for that dataset."
            )

    @property
    def group_key(self) -> str | None:
        return f"{self.source}:{self.group_id}" if self.group_id is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "raw_label": self.raw_label,
            "label": self.label.value,
            "available_labels": sorted(label.value for label in self.available_labels),
            "group_id": self.group_id,
            "supplied_split": self.supplied_split,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImageRecord:
        return cls(
            path=value["path"],
            source=value["source"],
            raw_label=value["raw_label"],
            label=UnifiedLabel(value["label"]),
            available_labels=frozenset(UnifiedLabel(label) for label in value["available_labels"]),
            group_id=value.get("group_id"),
            supplied_split=value.get("supplied_split"),
        )
