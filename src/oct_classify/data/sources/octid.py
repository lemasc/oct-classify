from __future__ import annotations

from collections.abc import Iterator

from oct_classify.data.models import DatasetSpec, ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel

_RAW_TO_LABEL = {"AMRD": UnifiedLabel.AMD, "NORMAL": UnifiedLabel.NORMAL}
_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


class OctidSource:
    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        for raw_label, label in _RAW_TO_LABEL.items():
            class_dir = spec.root / raw_label
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    yield ImageRecord(
                        path=str(path.relative_to(spec.root)),
                        source=spec.name,
                        raw_label=raw_label,
                        label=label,
                        available_labels=spec.available_labels,
                    )
