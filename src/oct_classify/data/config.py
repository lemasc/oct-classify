from __future__ import annotations

import tomllib
from pathlib import Path

from oct_classify.data.models import DatasetSpec
from oct_classify.data.taxonomy import parse_labels


def load_dataset_specs(path: Path) -> list[DatasetSpec]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    specs: list[DatasetSpec] = []
    for value in config.get("datasets", []):
        try:
            specs.append(
                DatasetSpec(
                    name=value["name"],
                    source=value["source"],
                    root=Path(value["root"]),
                    available_labels=parse_labels(value["available_labels"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid dataset entry in {path}: {value!r}") from error
    if not specs:
        raise ValueError(f"No [[datasets]] entries found in {path}")
    return specs
