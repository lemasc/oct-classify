from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from oct_classify.data.models import ImageRecord


def write_jsonl(path: Path, records: Iterable[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> Iterator[ImageRecord]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield ImageRecord.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    message = f"Invalid manifest record at {path}:{line_number}"
                    raise ValueError(message) from error


def apply_processing_decisions(
    records: Iterable[ImageRecord],
    *,
    exact_duplicates: Iterable[dict[str, Any]],
    near_duplicates: Iterable[dict[str, Any]],
) -> tuple[list[ImageRecord], int, int]:
    """Apply the locked quarantine and exact-deduplication policy to source records."""
    records = list(records)
    exact_duplicates = list(exact_duplicates)
    near_duplicates = list(near_duplicates)
    quarantined: set[str] = set()
    for duplicate in [*exact_duplicates, *near_duplicates]:
        if duplicate["quarantine_eligible"]:
            quarantined.update(duplicate["paths"])

    retained = {
        f"{record.source}:{record.path}": record
        for record in records
        if f"{record.source}:{record.path}" not in quarantined
    }
    deduplicated = 0
    for duplicate in exact_duplicates:
        paths = [path for path in duplicate["paths"] if path in retained]
        if len(paths) > 1:
            canonical = min(paths)
            for path in paths:
                if path != canonical:
                    del retained[path]
                    deduplicated += 1

    output: list[ImageRecord] = []
    seen: set[str] = set()
    for record in records:
        path = f"{record.source}:{record.path}"
        if path not in retained:
            continue
        if path in seen:
            deduplicated += 1
            continue
        seen.add(path)
        output.append(record)
    return output, len(quarantined), deduplicated
