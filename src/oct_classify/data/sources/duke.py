from __future__ import annotations

from collections.abc import Iterator

from oct_classify.data.models import DatasetSpec, ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel

_PREFIX_TO_LABEL = {
    "AMD": UnifiedLabel.AMD,
    "DME": UnifiedLabel.DME,
    "NORMAL": UnifiedLabel.NORMAL,
}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class DukeSource:
    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        for group_dir in sorted(path for path in spec.root.iterdir() if path.is_dir()):
            prefix = "".join(character for character in group_dir.name if character.isalpha())
            label = _PREFIX_TO_LABEL.get(prefix.upper())
            if label is None:
                continue
            for path in sorted(group_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    yield ImageRecord(
                        path=str(path.relative_to(spec.root)),
                        source=spec.name,
                        raw_label=prefix,
                        label=label,
                        available_labels=spec.available_labels,
                        group_id=group_dir.name,
                    )
