"""PAIMA held out (full source): AUROC of normal vs dry AMD and normal vs wet AMD, fill-aware vs square-crop LOSO.

Score is p(AMD) renormalised over normal + AMD, per model; an arm's AUROC is the mean over its 5 models. Seed
20260913 uses the final runs, with a paired patient-level bootstrap (500 draws, seed 20261014, shared by both arms);
seeds 20260914/5 (exp. 2, repo-retry .scratchpad/049579f_exp2-seeds) give point estimates for seed spread.
Run: uv run python data/paper/amd-stage/paima_auroc.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from oct_classify.analysis.bootstrap import _WeightedAuroc

REPO = Path(".").resolve()
OUT = Path(__file__).resolve().parent
BASE = REPO / "artifacts/runs/fused/resnet50"
EXP2 = REPO.parent / "repo-retry/.scratchpad/049579f_exp2-seeds/runs/fused/resnet50"
RUNS = {
    20260913: {"fill": BASE / "loso-paima-local-20261008-174938-fold{i}", "sq": BASE / "loso-paima-local-3a-r2-square-crop-fold{i}"},
    **{s: {"fill": EXP2 / f"loso-paima-exp2-fill-aware-s{s}-fold{{i}}",
           "sq": EXP2 / f"loso-paima-exp2-fill-aware-square-crop-s{s}-fold{{i}}"} for s in (20260914, 20260915)},
}
split = pd.read_json(REPO / "artifacts/splits/paima.jsonl", lines=True).set_index("path")
stage = np.where(split.label == "normal", "normal", np.where(split.raw_label == "DRUSEN", "dry", "wet"))
split["stage"] = stage
BA, SEED = 500, 20261014


def scores(run):
    d = pd.read_csv(str(run) + "/evaluations/paima-loso/predictions.csv").set_index("path")
    return d.probability_amd / (d.probability_amd + d.probability_normal)


R = {}
for seed, arms in RUNS.items():
    p = {a: [scores(str(t).format(i=i)) for i in range(1, 6)] for a, t in arms.items()}
    paths = sorted(set.intersection(*[set(x.index) for L in p.values() for x in L]))
    sp = split.loc[paths]
    for st in ("dry", "wet"):
        m = sp.stage.isin(["normal", st]).to_numpy()
        y = (sp.stage[m] == st).to_numpy().astype(int)
        codes, G = pd.factorize(sp.group_id[m].astype(str))
        G = len(G)
        counts = np.random.default_rng(SEED).multinomial(G, np.full(G, 1 / G), size=BA) if seed == 20260913 else None
        res = {}
        for a, L in p.items():
            aus = [_WeightedAuroc(y, np.c_[1 - x.loc[np.array(paths)[m]].to_numpy(), x.loc[np.array(paths)[m]].to_numpy()]) for x in L]
            point = float(np.mean([au(np.ones(len(y))) for au in aus]))
            reps = None if counts is None else np.array([np.mean([au(c[codes].astype(float)) for au in aus]) for c in counts])
            res[a] = (point, reps)
        row = {a: res[a][0] for a in res}
        if counts is not None:
            d = res["sq"][1] - res["fill"][1]
            row["delta_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        R[f"{seed}|normal_vs_{st}"] = row
        ci = row.get("delta_ci")
        print(seed, f"normal vs {st}: fill {100*row['fill']:.1f}  sq {100*row['sq']:.1f}  delta {100*(row['sq']-row['fill']):+.1f}"
              + (f" [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]" if ci else ""), f"n={m.sum()}", flush=True)
json.dump(R, open(OUT / "paima-auroc.json", "w"), indent=1)
