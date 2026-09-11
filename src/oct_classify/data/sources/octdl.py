from __future__ import annotations

import csv
from collections.abc import Iterator

from oct_classify.data.models import DatasetSpec, ImageRecord
from oct_classify.data.taxonomy import UnifiedLabel

_RAW_TO_LABEL = {"AMD": UnifiedLabel.AMD, "DME": UnifiedLabel.DME, "NO": UnifiedLabel.NORMAL}


class OctdlSource:
    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        labels_path = spec.root / "labels.csv"
        with labels_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_label = row["disease"].upper()
                label = _RAW_TO_LABEL.get(raw_label)
                if label is None:
                    continue
                image_path = spec.root / raw_label / f"{row['file_name']}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(f"OCTDL metadata references missing image: {image_path}")
                yield ImageRecord(
                    path=str(image_path.relative_to(spec.root)),
                    source=spec.name,
                    raw_label=raw_label,
                    label=label,
                    available_labels=spec.available_labels,
                    group_id=row["patient_id"] or None,
                )
