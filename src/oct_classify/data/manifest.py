from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

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
