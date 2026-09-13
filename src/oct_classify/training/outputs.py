from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def create_run_directory(
    output_root: Path, source: str, architecture: str, name: str | None
) -> Path:
    run_name = name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / source / architecture / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_predictions(
    path: Path,
    paths: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "target",
                "prediction",
                *[f"probability_{name}" for name in class_names],
            ],
        )
        writer.writeheader()
        for image_path, target, prediction, probability in zip(
            paths, targets, predictions, probabilities, strict=True
        ):
            writer.writerow(
                {
                    "path": image_path,
                    "target": class_names[int(target)],
                    "prediction": class_names[int(prediction)],
                    **{
                        f"probability_{name}": float(score)
                        for name, score in zip(class_names, probability, strict=True)
                    },
                }
            )
