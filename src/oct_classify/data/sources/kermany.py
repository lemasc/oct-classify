from __future__ import annotations

import re
from collections.abc import Iterator

from oct_classify.data.models import DatasetSpec, ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel

_RAW_TO_LABEL = {
    "CNV": UnifiedLabel.AMD,
    "DME": UnifiedLabel.DME,
    "DRUSEN": UnifiedLabel.AMD,
    "NORMAL": UnifiedLabel.NORMAL,
}
_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
_FILENAME_PATTERN = re.compile(r"^(?P<label>[^-]+)-(?P<group>.+)-(?P<scan>\d+)$")


class KermanySource:
    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        for split_dir in sorted(path for path in spec.root.iterdir() if path.is_dir()):
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                raw_label = class_dir.name.upper()
                label = _RAW_TO_LABEL.get(raw_label)
                if label is None:
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                        match = _FILENAME_PATTERN.fullmatch(path.stem)
                        if match is None or match["label"].upper() != raw_label:
                            raise ValueError(f"Unexpected Kermany filename: {path.name}")
                        yield ImageRecord(
                            path=str(path.relative_to(spec.root)),
                            source=spec.name,
                            raw_label=raw_label,
                            label=label,
                            available_labels=spec.available_labels,
                            group_id=match["group"],
                            supplied_split=split_dir.name,
                        )
