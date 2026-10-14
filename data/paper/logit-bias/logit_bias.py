import math
"""Per-source logit-bias diagnostic on fill-aware LOSO predictions (seed 20260913 runs, tag 20261008-174938).
Variants: none / bias-crossfit (patient halves) / bias-oracle (in-sample). Views: all (3-class), normal_amd.
Scoring conventions follow .scratchpad/6fc182a_final-scoring/score.py (splits group_id, shared-draw patient bootstrap,
model-mean metrics). Also: grid-search macro-F1 upper bound.
Run: uv run python .scratchpad/9d08e53_logit-bias/logit_bias.py
"""
import json, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import f1_score
from oct_classify.analysis.bootstrap import _WeightedAuroc, encode_groups, group_confusion, metrics_from_confusion

REPO = Path(".").resolve()
RUNS = REPO / "artifacts/runs/fused/resnet50"
TAG = "20261008-174938"
SRC = ["kermany", "octdl", "paima", "duke"]
CL = ["normal", "amd", "dme"]
B, BA, SEED = 2000, 500, 20260913
RIDGE, FLOOR = 1e-3, 1e-12
FINAL = json.load(open(REPO / "data/final-scoring/results.json"))
SPLITS = {s: pd.read_json(REPO / f"artifacts/splits/{s}.jsonl", lines=True).set_index("path") for s in SRC}

def read(p):
    d = pd.read_csv(p).set_index("path"); assert d.index.is_unique, p; return d
def loso(s):
    if s == "duke": return [read(RUNS / f"loso-duke-local-{TAG}/evaluations/duke-loso/predictions.csv")]
    return [read(RUNS / f"loso-{s}-local-{TAG}-fold{i}/evaluations/{s}-loso/predictions.csv") for i in range(1, 6)]

def pct(x, q):
    x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if len(x) else np.nan

def replicates(codes, G, fs, counts, names):  # verbatim from score.py
    K = len(names); pts, reps, ap, ar, confs = [], [], [], [], []
    for y, P in fs:
        conf = group_confusion(codes, y, P.argmax(1), G, K)
        pts.append(metrics_from_confusion(conf.sum(0))); reps.append(metrics_from_confusion(np.einsum("bg,gij->bij", counts, conf)))
        au = _WeightedAuroc(y, P); s = au(np.ones(len(y))); ap.append(np.nan if s is None else s)
        ar.append([np.nan if (s := au(counts[r][codes].astype(float))) is None else s for r in range(BA)])
        confs.append(conf.sum(0))
    ren = lambda n: f"recall_{names[int(n[7:])]}" if n.startswith("recall_") else n
    point, rep = {}, {}
    for n in pts[0]:
        v = np.array([float(p[n]) for p in pts]); point[ren(n)] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rep[ren(n)] = np.stack([r[n] for r in reps]).mean(0)
    v = np.array(ap); point["macro_auroc"] = float(v.mean()); rep["macro_auroc"] = np.array(ar).mean(0)
    return point, rep, np.mean(confs, axis=0)

def fit_bias(logp, t):
    """Additive bias (last class 0) minimising class-balanced NLL + ridge."""
    k = logp.shape[1]; present = np.unique(t); w = np.zeros(len(t))
    for c in present: w[t == c] = 1.0 / ((t == c).sum() * len(present))
    def loss(free):
        b = np.append(free, 0.0)
        return float(-(w * log_softmax(logp + b, axis=1)[np.arange(len(t)), t]).sum() + RIDGE * (free ** 2).sum())
    return np.append(minimize(loss, np.zeros(k - 1), method="L-BFGS-B").x, 0.0)

def halves(groups, t):
    by = {}
    for r, g in enumerate(groups): by.setdefault(g, []).append(r)
    bl = {}
    for g in sorted(by):
        maj = max(sorted(Counter(t[by[g]]).items()), key=lambda kv: kv[1])[0]; bl.setdefault(int(maj), []).append(g)
    rng = random.Random(SEED); h = np.zeros(len(groups), int); off = 0
    for lab in sorted(bl):
        ks = bl[lab]; rng.shuffle(ks)
        for i, g in enumerate(ks): h[by[g]] = (off + i) % 2
        off += len(ks)
    return h

def run_view(src, view, classes, dfs):
    sets = [set(d.index) for d in dfs]; first = dfs[0]
    paths = np.array(sorted(set.intersection(*sets))); paths = paths[first.loc[paths, "target"].isin(classes).to_numpy()]
    groups = SPLITS[src].loc[paths, "group_id"].astype(str).to_numpy(); codes, G = encode_groups(groups)
    idx = {c: i for i, c in enumerate(classes)}; cols = [f"probability_{c}" for c in classes]
    t = first.loc[paths, "target"].map(idx).to_numpy(); h = halves(list(groups), t)
    arms = {"none": [], "bias-crossfit": [], "bias-oracle": []}; biases = {"bias-crossfit": [], "bias-oracle": []}
    for d in dfs:
        x = d.loc[paths]; assert (x.target.map(idx).to_numpy() == t).all()
        P = x[cols].to_numpy(float); P = P / P.sum(1, keepdims=True); lp = np.log(np.maximum(P, FLOOR))
        arms["none"].append((t, P)); cross = np.zeros_like(P)
        for k in (0, 1):
            b = fit_bias(lp[h != k], t[h != k]); biases["bias-crossfit"].append(b); cross[h == k] = np.exp(log_softmax(lp[h == k] + b, axis=1))
        arms["bias-crossfit"].append((t, cross))
        b = fit_bias(lp, t); biases["bias-oracle"].append(b); arms["bias-oracle"].append((t, np.exp(log_softmax(lp + b, axis=1))))
    rng = np.random.default_rng(SEED); counts = rng.multinomial(G, np.full(G, 1 / G), size=B)
    res = {a: replicates(codes, G, fs, counts, classes) for a, fs in arms.items()}
    seen = FINAL[f"seen|{src}|{view}"]["arms"]["fused-fill"]["metrics"]["macro_f1"]["point"]
    out = dict(n_images=len(paths), n_groups=int(G), classes=classes, n_models=len(dfs), seen_macro_f1=seen, arms={})
    none_f1 = res["none"][0]["macro_f1"]
    for a, (pt, rep, conf) in res.items():
        o = dict(metrics={m: dict(point=pt[m], lo=pct(rep[m], 2.5), hi=pct(rep[m], 97.5)) for m in pt})
        if a != "none":
            o["delta"] = {m: dict(point=pt[m] - res["none"][0][m], lo=pct(rep[m] - res["none"][1][m], 2.5), hi=pct(rep[m] - res["none"][1][m], 97.5)) for m in pt}
            bb = np.array(biases[a]); o["bias_mean"] = bb.mean(0)[:-1].tolist(); o["bias_sd"] = bb.std(0)[:-1].tolist()
            o["gap_closed"] = None if abs(seen - none_f1) < 0.01 else (pt["macro_f1"] - none_f1) / (seen - none_f1)
        out["arms"][a] = o
    # grid-search in-sample upper bound
    grid = np.arange(-4, 4.01, 0.25); gl = []
    for (y, P) in arms["none"]:
        lp = np.log(np.maximum(P, FLOOR))
        sub = np.random.default_rng(0).choice(len(y), 20000, replace=False) if len(y) > 20000 else np.arange(len(y))
        lp, yy = lp[sub], y[sub]; f0 = f1_score(yy, lp.argmax(1), average="macro")
        if len(classes) == 3: best = max((f1_score(yy, (lp + [bn, ba, 0]).argmax(1), average="macro"), bn, ba) for bn in grid for ba in grid)
        else: best = max((f1_score(yy, (lp + [bn, 0]).argmax(1), average="macro"), bn, np.nan) for bn in grid)
        gl.append((f0, *best))
    g = np.nanmean(gl, axis=0); out["grid"] = dict(none=g[0], best=g[1], bias_normal=g[2], bias_amd=g[3], subsampled=bool(len(t) > 20000))
    return out

R = {}
for s in SRC:
    dfs = loso(s)
    allc = [c for c in CL if f"probability_{c}" in dfs[0].columns]
    for v, cl in ({"normal_amd": ["normal", "amd"]} if s == "paima" else {"all": allc, "normal_amd": ["normal", "amd"]}).items():
        R[f"{s}|{v}"] = run_view(s, v, cl, dfs); r = R[f"{s}|{v}"]
        ref = FINAL[f"loso_full|{s}|{v}"]["arms"]["loso-fill"]["metrics"]; m = r["arms"]["none"]["metrics"]
        r["sanity"] = dict(f1_diff=m["macro_f1"]["point"] - ref["macro_f1"]["point"], auroc_diff=m["macro_auroc"]["point"] - ref["macro_auroc"]["point"])
        print(s, v, "none F1 %.4f (final %.4f) AUROC %.4f (final %.4f)" % (m["macro_f1"]["point"], ref["macro_f1"]["point"], m["macro_auroc"]["point"], ref["macro_auroc"]["point"]), flush=True)
OUT = Path(__file__).resolve().parent
def strict(v):  # NaN -> null, so the paper (Typst) can read the file
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, dict): return {k: strict(x) for k, x in v.items()}
    if isinstance(v, list): return [strict(x) for x in v]
    return v
json.dump(strict(R), open(OUT / "results.json", "w"), indent=1, allow_nan=False)
