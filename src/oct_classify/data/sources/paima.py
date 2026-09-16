from __future__ import annotations

import csv
from collections.abc import Iterator

from oct_classify.data.models import DatasetSpec, ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel

_RAW_TO_LABEL = {
    "CNV": UnifiedLabel.AMD,
    "DRUSEN": UnifiedLabel.AMD,
    "NORMAL": UnifiedLabel.NORMAL,
}


class PaimaSource:
    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        metadata_path = spec.root / "data_information.csv"
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_label = row["Label"].upper()
                label = _RAW_TO_LABEL.get(raw_label)
                if label is None:
                    continue
                path = row["Directory"]
                image_path = spec.root / path
                if not image_path.is_file():
                    image_path = next(
                        (
                            candidate
                            for candidate in image_path.parent.iterdir()
                            if candidate.name.casefold() == image_path.name.casefold()
                        ),
                        None,
                    )
                if image_path is None or not image_path.is_file():
                    raise FileNotFoundError(
                        f"PAIMA metadata references missing image: {spec.root / path}"
                    )
                yield ImageRecord(
                    path=str(image_path.relative_to(spec.root)),
                    source=spec.name,
                    raw_label=raw_label,
                    label=label,
                    available_labels=spec.available_labels,
                    group_id=(f"{row['Class']}/{row['Patient ID']}" if row["Patient ID"] else None),
                    label_unit="image",
                    eye_id=(
                        f"{row['Class']}/{row['Patient ID']}/{row['Eye']}"
                        if row["Patient ID"] and row["Eye"]
                        else None
                    ),
                    cohort=row["Class"],
                )
