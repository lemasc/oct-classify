from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from oct_classify.data.models import ImageRecord


@dataclass(frozen=True, slots=True)
class AuditReport:
    image_count: int
    label_counts: dict[str, int]
    supplied_split_counts: dict[str, int]
    images_without_group: int
    invalid_images: list[str]
    widths: tuple[int, int] | None
    heights: tuple[int, int] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_records(root: Path, records: Iterable[ImageRecord]) -> AuditReport:
    records = list(records)
    labels = Counter(record.label.value for record in records)
    splits = Counter(record.supplied_split for record in records if record.supplied_split)
    invalid_images: list[str] = []
    widths: list[int] = []
    heights: list[int] = []

    for record in records:
        try:
            with Image.open(root / record.path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError):
            invalid_images.append(record.path)
        else:
            widths.append(width)
            heights.append(height)

    return AuditReport(
        image_count=len(records),
        label_counts=dict(sorted(labels.items())),
        supplied_split_counts=dict(sorted(splits.items())),
        images_without_group=sum(record.group_id is None for record in records),
        invalid_images=invalid_images,
        widths=(min(widths), max(widths)) if widths else None,
        heights=(min(heights), max(heights)) if heights else None,
    )
