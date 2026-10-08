"""Extract per-image EDA features for all derived manifests into artifacts/eda/features.csv.

uv run python scripts/extract_eda_features.py --workers 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oct_classify.data.config import load_dataset_specs
from oct_classify.data.eda import collect_features
from oct_classify.data.manifest import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "eda" / "features.csv")
    args = parser.parse_args()

    specs = [s for s in load_dataset_specs(ROOT / "configs" / "datasets.toml") if s.enabled]
    frames = []
    for spec in specs:
        records = list(read_jsonl(ROOT / "artifacts" / "manifests" / f"{spec.name}.jsonl"))
        features = collect_features({spec.source: ROOT / spec.root}, records, workers=args.workers)
        frame = pd.DataFrame(features)
        meta = pd.DataFrame(
            {
                "source": [r.source for r in records],
                "path": [r.path for r in records],
                "label": [r.label.value for r in records],
                "raw_label": [r.raw_label for r in records],
                "group_id": [r.group_id for r in records],
                "supplied_split": [r.supplied_split for r in records],
            }
        )
        frames.append(pd.concat([meta, frame], axis=1))
        print(spec.name, len(records), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
