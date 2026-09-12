from __future__ import annotations

from oct_classify.data.sources.base import DatasetSource
from oct_classify.data.sources.duke import DukeSource
from oct_classify.data.sources.kermany import KermanySource
from oct_classify.data.sources.octdl import OctdlSource
from oct_classify.data.sources.octid import OctidSource
from oct_classify.data.sources.paima import PaimaSource

_SOURCES: dict[str, DatasetSource] = {
    "duke": DukeSource(),
    "kermany": KermanySource(),
    "octdl": OctdlSource(),
    "octid": OctidSource(),
    "paima": PaimaSource(),
}


def get_source(name: str) -> DatasetSource:
    try:
        return _SOURCES[name]
    except KeyError as error:
        known_sources = ", ".join(sorted(_SOURCES))
        raise ValueError(f"Unknown source {name!r}. Known sources: {known_sources}") from error
