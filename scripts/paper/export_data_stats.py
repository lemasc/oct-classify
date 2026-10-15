"""Dataset statistics for the paper's data section (counts, group units, image sizes, deduplication).

Reads the derived manifests, splits and audits under `artifacts/` and writes `data/paper/data-stats.json`, which
`docs/paper/lib.typ` loads. AMD stage shares come from `data/paper/amd-stage/results.json`, not from here.
Run from the repository root: `uv run python scripts/paper/export_data_stats.py`.
"""

import json
from pathlib import Path

import pandas as pd

ARTIFACTS = Path("artifacts")
OUT = Path("data/paper/data-stats.json")
SOURCES = ["kermany", "octdl", "paima", "duke"]
CLASSES = ["normal", "amd", "dme"]
DUKE_FOLDS = range(1, 6)


def dedup_counts(source: str, kept: int) -> dict:
    """Records removed from the raw source by quarantine and exact-copy removal (docs/data-audit-log.md)."""
    audit = json.loads((ARTIFACTS / f"audits/{source}.json").read_text())
    quarantined = {
        path
        for duplicate in [*audit["exact_duplicates"], *audit["near_duplicates"]]
        if duplicate["quarantine_eligible"]
        for path in duplicate["paths"]
    }
    return {
        "raw_images": audit["image_count"],
        "quarantined": len(quarantined),
        "same_label_copies_removed": audit["image_count"] - len(quarantined) - kept,
    }


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "eda/features.csv", usecols=["source", "path", "label", "width", "height", "group_id"],
                           dtype={"group_id": str})
    features["size"] = features["width"].astype(str) + "x" + features["height"].astype(str)
    stats: dict = {"sources": {}}
    for source in SOURCES:
        split = pd.read_json(ARTIFACTS / f"splits/{source}.jsonl", lines=True)
        feats = features[features["source"] == source]
        assert len(feats) == len(split), source
        unit = "eye_id" if source == "duke" else "group_id"
        entry = {
            "n_images": len(split),
            "n_groups": int(split[unit].nunique()),
            "label_unit": split["label_unit"].iloc[0],
            "classes": {c: int((split["label"] == c).sum()) for c in CLASSES},
            "sizes": {k: int(v) for k, v in feats["size"].value_counts().items()},
            "size_by_class": {
                size: {c: int(n) for c, n in part["label"].value_counts().items()} for size, part in feats.groupby("size")
            },
            "dedup": dedup_counts(source, len(split)),
        }
        if source == "kermany":
            test = feats.merge(split[["path", "split"]], on="path")
            test = test[test["split"] == "test"]
            entry["n_test"] = len(test)
            entry["test_sizes"] = {k: int(v) for k, v in test["size"].value_counts().items()}
        if source == "duke":
            entry["eyes_by_size_class"] = {
                size: {c: int(n) for c, n in part.groupby("label")["group_id"].nunique().items()}
                for size, part in feats.groupby("size")
            }
            folds = [pd.read_json(ARTIFACTS / f"splits/duke-cv/fold-{i}/duke.jsonl", lines=True) for i in DUKE_FOLDS]
            tests = [fold[fold["split"] == "test"] for fold in folds]
            entry["test_eyes_per_fold"] = sorted({int(t["eye_id"].nunique()) for t in tests})
            entry["test_eyes_per_class_per_fold"] = sorted(
                {int(n) for t in tests for n in t.groupby("label")["eye_id"].nunique()}
            )
        stats["sources"][source] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
