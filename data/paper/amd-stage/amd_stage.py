"""AMD recall split by stage (dry = drusen, wet = CNV/MNV) for Kermany, OCTDL and PAIMA, every variant and arm.

Stage comes from the source labels: Kermany and PAIMA raw_label DRUSEN vs CNV; OCTDL labels.csv subcategory early
(drusen) vs late (MNV), with intermediate (MNV suspected) kept as its own stage. Duke's AMD is dry only and has no
stage split. Runs, splits and the patient-level shared-draw bootstrap (seed 20261014, 2000 draws) follow
data/paper/zero-dme/zero_dme.py and .scratchpad/6fc182a_final-scoring/score.py. A prediction counts as AMD by argmax
over the source's classes ("all": three classes; PAIMA and the "normal_amd" view: normal vs AMD renormalised).
Recall of an arm with several models is the mean over models. Run: uv run python data/paper/amd-stage/amd_stage.py
"""
import csv, json, math
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(".").resolve()
RUNS = REPO / "artifacts/runs"
OUT = Path(__file__).resolve().parent
B, SEED = 2000, 20261014
SRC = ["kermany", "octdl", "paima"]
SPLITS = {s: pd.read_json(REPO / f"artifacts/splits/{s}.jsonl", lines=True).set_index("path") for s in SRC}
TEST = {s: set(SPLITS[s].index[SPLITS[s].split == "test"]) for s in SRC}
OCTDL_STAGE = {r["file_name"]: r["subcategory"] for r in csv.DictReader(open(REPO / "datasets/octdl/labels.csv"))}


def stage(src, path, raw):
    if src == "octdl":
        return {"early": "dry", "late": "wet", "intermediate": "intermediate"}[OCTDL_STAGE[Path(path).stem]]
    return {"DRUSEN": "dry", "CNV": "wet"}.get(raw)


def read(p):
    d = pd.read_csv(p).set_index("path"); assert d.index.is_unique, p; return d


FILL_F, SQ_F = "exp4-fill-aware-58577-fold", "fused-square-crop-local-fold"
FILL_L, SQ_L = "20261008-174938", "3a-r2-square-crop"
base = RUNS / "fused/resnet50"
ARMS = {
    "per-dataset": lambda s: [read(RUNS / f"{s}/resnet50/local-final-fill-aware-{s}/predictions-test.csv")],
    "fused-fill": lambda s: [read(base / f"{FILL_F}{i}/predictions-test/{s}.csv") for i in range(1, 6)],
    "fused-sq": lambda s: [read(base / f"{SQ_F}{i}/predictions-test/{s}.csv") for i in range(1, 6)],
    "loso-fill": lambda s: [read(base / f"loso-{s}-local-{FILL_L}-fold{i}/evaluations/{s}-loso/predictions.csv")
                            for i in range(1, 6)],
    "loso-sq": lambda s: [read(base / f"loso-{s}-local-{SQ_L}-fold{i}/evaluations/{s}-loso/predictions.csv")
                          for i in range(1, 6)],
}
PAIRS = [("loso-fill", "fused-fill"), ("fused-fill", "per-dataset"), ("loso-sq", "loso-fill"),
         ("fused-sq", "fused-fill"), ("loso-sq", "fused-sq")]
SCOPES = {"test": list(ARMS), "full": ["loso-fill", "loso-sq"]}


def predicted(d, classes):
    P = d[[f"probability_{c}" for c in classes]].to_numpy(float)
    return np.array(classes)[P.argmax(1)]


def pct(x, q):
    return float(np.percentile(x[np.isfinite(x)], q))


def analyse(src, scope, view):
    dfs = {a: ARMS[a](src) for a in SCOPES[scope]}
    paths = set.intersection(*[set(d.index) for L in dfs.values() for d in L])
    if scope == "test":
        paths &= TEST[src]
    paths = np.array(sorted(paths))
    sp = SPLITS[src].loc[paths]
    classes = ["normal", "amd"] if (src == "paima" or view == "normal_amd") else ["normal", "amd", "dme"]
    keep = sp.label.isin(classes).to_numpy()
    paths, sp = paths[keep], sp[keep]
    stages = np.array([stage(src, p, r) if l == "amd" else l for p, r, l in zip(paths, sp.raw_label, sp.label)])
    groups, G = pd.factorize(sp.group_id.astype(str))[0], sp.group_id.nunique()
    counts = np.random.default_rng(SEED).multinomial(G, np.full(G, 1 / G), size=B)  # shared by all arms
    out = {"n_images": int(len(paths)), "n_groups": int(G), "stages": {}, "deltas": {}}
    reps = {}
    for st in sorted(set(stages)):
        m = stages == st
        n_g = np.bincount(groups[m], minlength=G).astype(float)
        row = {"n_images": int(m.sum()), "n_groups": int((n_g > 0).sum()), "arms": {}}
        for a, L in dfs.items():
            preds = np.stack([predicted(d.loc[paths[m]], classes) for d in L])  # models x images
            target = "amd" if st in ("dry", "wet", "intermediate") else st
            share = {c: float((preds == c).mean()) for c in classes}
            hit_g = np.stack([np.bincount(groups[m], weights=(p == target), minlength=G) for p in preds]).mean(0)
            r = (counts @ hit_g) / (counts @ n_g)
            reps[(st, a)] = r
            row["arms"][a] = dict(recall=share[target], lo=pct(r, 2.5), hi=pct(r, 97.5),
                                  predicted_as=share, n_models=len(L))
        out["stages"][st] = row
        for a, b in PAIRS:
            if a in dfs and b in dfs:
                d = reps[(st, a)] - reps[(st, b)]
                out["deltas"].setdefault(st, {})[f"{a} - {b}"] = dict(
                    point=row["arms"][a]["recall"] - row["arms"][b]["recall"], lo=pct(d, 2.5), hi=pct(d, 97.5))
    return out


def strict(v):
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, dict): return {k: strict(x) for k, x in v.items()}
    return v


R = {}
for s in SRC:
    for scope in SCOPES:
        for view in (["normal_amd"] if s == "paima" else ["all", "normal_amd"]):
            R[f"{scope}|{s}|{view}"] = r = analyse(s, scope, view)
            print(f"\n== {scope} {s} {view}: {r['n_images']} images, {r['n_groups']} groups")
            for st, row in r["stages"].items():
                arms = "  ".join(f"{a} {100 * v['recall']:.1f}" for a, v in row["arms"].items())
                print(f"  {st:12s} n={row['n_images']:6d}  {arms}")
                for p, d in r["deltas"].get(st, {}).items():
                    print(f"      {p:24s} {100 * d['point']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]")
json.dump(strict(R), open(OUT / "results.json", "w"), indent=1, allow_nan=False)
