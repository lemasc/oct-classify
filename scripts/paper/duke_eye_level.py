"""Duke eye-level scores for the paper footnote (main metrics stay per B-scan, see data/final-scoring).

Each eye's B-scan probabilities are averaged (as the training CLI's `eye_level` metrics do), then the 45 eyes are
scored. Duke 5-fold arms are pooled out-of-fold, as in data/final-scoring. 95% bootstrap intervals resample eyes.
Run from the repository root: `uv run python scripts/paper/duke_eye_level.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

RUNS = Path("artifacts/runs")
OUT = Path("data/paper/duke-eye-level.json")
CLASSES = ["normal", "amd", "dme"]
DRAWS, SEED = 2000, 20261014

ARMS = {
    "per-dataset": [RUNS / f"duke/resnet50/local-final-fill-aware-duke-fold{i}/predictions-test.csv" for i in range(1, 6)],
    "fused-fill": [RUNS / f"fused/resnet50/exp4-fill-aware-58577-fold{i}/predictions-test/duke.csv" for i in range(1, 6)],
    "fused-sq": [RUNS / f"fused/resnet50/fused-square-crop-local-fold{i}/predictions-test/duke.csv" for i in range(1, 6)],
    "loso-fill": [RUNS / "fused/resnet50/loso-duke-local-20261008-174938/evaluations/duke-loso/predictions.csv"],
    "loso-sq": [RUNS / "fused/resnet50/loso-duke-local-3a-r2-square-crop/evaluations/duke-loso/predictions.csv"],
}


def eye_table(paths: list[Path], eyes: pd.Series) -> pd.DataFrame:
    rows = pd.concat([pd.read_csv(path) for path in paths]).set_index("path")
    assert rows.index.is_unique and len(rows) == len(eyes), "expected every Duke B-scan exactly once"
    rows["eye"] = eyes.loc[rows.index]
    columns = [f"probability_{name}" for name in CLASSES]
    table = rows.groupby("eye").agg({**{column: "mean" for column in columns}, "target": "first"})
    assert (rows.groupby("eye")["target"].nunique() == 1).all(), "an eye has mixed labels"
    return table


def scores(table: pd.DataFrame) -> tuple[float, float]:
    target = table["target"].map(CLASSES.index).to_numpy()
    probabilities = table[[f"probability_{name}" for name in CLASSES]].to_numpy()
    probabilities = probabilities / probabilities.sum(1, keepdims=True)  # stored fp32 rows sum to 1 only approximately
    f1 = f1_score(target, probabilities.argmax(1), labels=[0, 1, 2], average="macro")
    auroc = roc_auc_score(target, probabilities, multi_class="ovr", average="macro", labels=[0, 1, 2])
    return f1, auroc


def main() -> None:
    splits = pd.read_json("artifacts/splits/duke.jsonl", lines=True).set_index("path")
    eyes = splits["eye_id"].astype(str)
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, eyes.nunique(), eyes.nunique()) for _ in range(DRAWS)]
    result = {"n_eyes": int(eyes.nunique()), "draws": DRAWS, "seed": SEED, "arms": {}}
    for arm, paths in ARMS.items():
        table = eye_table(paths, eyes).sort_index()
        f1, auroc = scores(table)
        boot = []
        for index in draws:
            sample = table.iloc[index]
            if sample["target"].nunique() == len(CLASSES):
                boot.append(scores(sample))
        boot = np.array(boot)
        result["arms"][arm] = {
            "n_models": len(paths),
            "macro_f1": {"point": f1, "lo": float(np.percentile(boot[:, 0], 2.5)), "hi": float(np.percentile(boot[:, 0], 97.5))},
            "macro_auroc": {"point": auroc, "lo": float(np.percentile(boot[:, 1], 2.5)), "hi": float(np.percentile(boot[:, 1], 97.5))},
            "misclassified_eyes": int((table[[f"probability_{n}" for n in CLASSES]].to_numpy().argmax(1) != table["target"].map(CLASSES.index).to_numpy()).sum()),
        }
        print(arm, {k: round(v["point"] * 100, 1) for k, v in result["arms"][arm].items() if isinstance(v, dict)},
              "wrong eyes:", result["arms"][arm]["misclassified_eyes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
