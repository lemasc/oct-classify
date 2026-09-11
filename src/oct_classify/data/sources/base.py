from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from oct_classify.data.models import DatasetSpec, ImageRecord


class DatasetSource(Protocol):
    """Turns one source's layout and labels into canonical image records."""

    def records(self, spec: DatasetSpec) -> Iterator[ImageRecord]: ...
