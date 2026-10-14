"""Item F: fused zero-Kermany-DME CV (fill-aware) vs fused fill-aware (exp. 4) and fill-aware LOSO, test split.
Helpers (read/pool/replicates/evaluate/views) are copied from .scratchpad/6fc182a_final-scoring/score.py (seed 20261014,
patient-level shared-draw bootstrap). Duke arms pooled out-of-fold. Run: uv run python data/paper/zero-dme/zero_dme.py
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from oct_classify.analysis.bootstrap import _WeightedAuroc, encode_groups, group_confusion, metrics_from_confusion

REPO = Path(".").resolve()
RUNS = REPO / "artifacts/runs"
OUT = Path(__file__).resolve().parent
SRC = ["kermany", "octdl", "paima", "duke"]
B, BA, SEED = 2000, 500, 20261014
CL = ["normal", "amd", "dme"]

SPLITS = {s: pd.read_json(REPO / f"artifacts/splits/{s}.jsonl", lines=True).set_index("path") for s in SRC}
for s in SRC: assert SPLITS[s].index.is_unique
TEST = {s: set(SPLITS[s].index[SPLITS[s].split == "test"]) if s != "duke" else set(SPLITS[s].index) for s in SRC}

def read(p):
    d = pd.read_csv(p).set_index("path"); assert d.index.is_unique, p; return d
def pool(ds):
    d = pd.concat(ds); assert d.index.is_unique; return d

def fused(s, pref):
    ds = [read(RUNS / f"fused/resnet50/{pref}{i}/predictions-test/{s}.csv") for i in range(1, 6)]
    return [pool(ds)] if s == "duke" else ds
def loso(s, tag):
    base = RUNS / "fused/resnet50"
    if s == "duke": return [read(base / f"loso-duke-local-{tag}/evaluations/duke-loso/predictions.csv")]
    return [read(base / f"loso-{s}-local-{tag}-fold{i}/evaluations/{s}-loso/predictions.csv") for i in range(1, 6)]
FILL_F, SQ_F = "exp4-fill-aware-58577-fold", "fused-square-crop-local-fold"
FILL_L, SQ_L = "20261008-174938", "3a-r2-square-crop"

def pct(x, q): 
    x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if len(x) else np.nan

def replicates(codes, G, fs, counts, names):
    K = len(names); pts, reps, ap, ar, confs = [], [], [], [], []
    for y, P in fs:
        conf = group_confusion(codes, y, P.argmax(1), G, K)
        pts.append(metrics_from_confusion(conf.sum(0))); reps.append(metrics_from_confusion(np.einsum("bg,gij->bij", counts, conf)))
        au = _WeightedAuroc(y, P); s = au(np.ones(len(y))); ap.append(np.nan if s is None else s)
        ar.append([np.nan if (s := au(counts[r][codes].astype(float))) is None else s for r in range(BA)])
        confs.append(conf.sum(0))
    ren = lambda n: f"recall_{names[int(n[7:])]}" if n.startswith("recall_") else n
    point, rep, sd = {}, {}, {}
    for n in pts[0]:
        v = np.array([float(p[n]) for p in pts]); point[ren(n)] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        sd[ren(n)] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
        rep[ren(n)] = np.stack([r[n] for r in reps]).mean(0)
    v = np.array(ap); point["macro_auroc"] = float(v.mean()); sd["macro_auroc"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    rep["macro_auroc"] = np.array(ar).mean(0)
    return point, rep, sd, np.mean(confs, axis=0)

def evaluate(arms, src, classes, pairs, rows=None):
    sets = [set(d.index) for dfs in arms.values() for d in dfs]
    paths = set.intersection(*sets) if rows is None else set.intersection(*sets, rows)
    first = next(iter(arms.values()))[0]
    paths = np.array(sorted(paths)); paths = paths[first.loc[paths, "target"].isin(classes).to_numpy()]
    if len(paths) == 0: return None
    for dfs in arms.values():
        for d in dfs: assert (d.loc[paths, "target"].to_numpy() == first.loc[paths, "target"].to_numpy()).all()
    codes, G = encode_groups(SPLITS[src].loc[paths, "group_id"].astype(str).to_numpy())
    idx = {c: i for i, c in enumerate(classes)}; cols = [f"probability_{c}" for c in classes]
    prep = {}
    for a, dfs in arms.items():
        L = []
        for d in dfs:
            x = d.loc[paths]; P = x[cols].to_numpy(float); L.append((x.target.map(idx).to_numpy(), P / P.sum(1, keepdims=True)))
        prep[a] = L
    rng = np.random.default_rng(SEED); counts = rng.multinomial(G, np.full(G, 1 / G), size=B)
    res = {a: replicates(codes, G, L, counts, classes) for a, L in prep.items()}
    out = dict(n_images=len(paths), n_groups=int(G), classes=classes, arms={}, deltas={})
    for a, (pt, rep, sd, conf) in res.items():
        out["arms"][a] = dict(n_models=len(arms[a]), confusion=conf.tolist(), metrics={
            m: dict(point=pt[m], lo=pct(rep[m], 2.5), hi=pct(rep[m], 97.5), fold_sd=sd[m]) for m in pt})
    for a, b in pairs:
        out["deltas"][f"{a} - {b}"] = {m: dict(point=res[a][0][m] - res[b][0][m],
            lo=pct(res[a][1][m] - res[b][1][m], 2.5), hi=pct(res[a][1][m] - res[b][1][m], 97.5)) for m in res[a][0]}
    return out

def views(dfs, src):
    allc = [c for c in CL if f"probability_{c}" in dfs[0].columns]
    return {"normal_amd": ["normal", "amd"]} if src == "paima" else {"all": allc, **({"normal_amd": ["normal", "amd"]} if len(allc) == 3 else {})}

ZERO_F = "fused-zero-dme-noval-local-fold"
R = {}
for s in SRC:
    arms = {"zero-dme": fused(s, ZERO_F), "fused-fill": fused(s, FILL_F)}
    pairs = [("zero-dme", "fused-fill")]
    if s != "duke":
        for d in [*arms["zero-dme"], *arms["fused-fill"]]: assert set(d.index) == TEST[s], s
    else:
        arms["loso-fill"] = loso(s, FILL_L); pairs.append(("loso-fill", "zero-dme"))
    if s == "kermany":
        arms["loso-fill"] = loso(s, FILL_L); pairs.append(("loso-fill", "zero-dme"))
    for v, cl in views(arms["zero-dme"], s).items():
        rows = TEST[s]
        r = evaluate(arms, s, cl, pairs, rows)
        R[f"{s}|{v}"] = r; print("done", s, v, r["n_images"], r["n_groups"], flush=True)

def strict(v):
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, dict): return {k: strict(x) for k, x in v.items()}
    if isinstance(v, list): return [strict(x) for x in v]
    return v
json.dump(strict(R), open(OUT / "results.json", "w"), indent=1, allow_nan=False)
for k, r in R.items():
    d = r["deltas"]["zero-dme - fused-fill"]
    print(k, "F1 zero %.1f fused %.1f delta %.1f [%.1f, %.1f]" % (100*r["arms"]["zero-dme"]["metrics"]["macro_f1"]["point"], 100*r["arms"]["fused-fill"]["metrics"]["macro_f1"]["point"], 100*d["macro_f1"]["point"], 100*d["macro_f1"]["lo"], 100*d["macro_f1"]["hi"]))
