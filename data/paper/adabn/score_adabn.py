"""AdaBN (all BN layers, patient cross-fit) vs stored predictions on fill-aware LOSO (tag 20261008-174938).
Arms: none (stored <src>-loso/predictions.csv), adabn-all (<src>-loso-adabn-all/predictions.csv).
Scoring as data/paper/logit-bias/logit_bias.py. Writes results.json and table.md next to this file.
Run (repo root, CPU): uv run python data/paper/adabn/score_adabn.py
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from oct_classify.analysis.bootstrap import _WeightedAuroc, encode_groups, group_confusion, metrics_from_confusion

REPO = Path(".").resolve()
RUNS = REPO / "artifacts/runs/fused/resnet50"
TAG = "20261008-174938"
SRC = ["kermany", "octdl", "paima", "duke"]
CL = ["normal", "amd", "dme"]
B, BA, SEED = 2000, 500, 20260913
FLOOR = 1e-12
FINAL = json.load(open(REPO / "data/final-scoring/results.json"))
SPLITS = {s: pd.read_json(REPO / f"artifacts/splits/{s}.jsonl", lines=True).set_index("path") for s in SRC}
ARMS = {"none": "{s}-loso", "adabn-all": "{s}-loso-adabn-all"}

def read(p):
    d = pd.read_csv(p).set_index("path"); assert d.index.is_unique, p; return d
def runs(s):
    return [RUNS / f"loso-duke-local-{TAG}"] if s == "duke" else [RUNS / f"loso-{s}-local-{TAG}-fold{i}" for i in range(1, 6)]
def load(s, arm): return [read(r / "evaluations" / ARMS[arm].format(s=s) / "predictions.csv") for r in runs(s)]

def pct(x, q):
    x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if len(x) else np.nan

def replicates(codes, G, fs, counts, names):
    K = len(names); pts, reps, ap, ar = [], [], [], []
    for y, P in fs:
        conf = group_confusion(codes, y, P.argmax(1), G, K)
        pts.append(metrics_from_confusion(conf.sum(0))); reps.append(metrics_from_confusion(np.einsum("bg,gij->bij", counts, conf)))
        au = _WeightedAuroc(y, P); s = au(np.ones(len(y))); ap.append(np.nan if s is None else s)
        ar.append([np.nan if (s := au(counts[r][codes].astype(float))) is None else s for r in range(BA)])
    ren = lambda n: f"recall_{names[int(n[7:])]}" if n.startswith("recall_") else n
    point, rep = {}, {}
    for n in pts[0]:
        v = np.array([float(p[n]) for p in pts]); point[ren(n)] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rep[ren(n)] = np.stack([r[n] for r in reps]).mean(0)
    point["macro_auroc"] = float(np.array(ap).mean()); rep["macro_auroc"] = np.array(ar).mean(0)
    return point, rep

def run_view(src, classes, dfs_by_arm):
    first = dfs_by_arm["none"][0]
    sets = [set(d.index) for ds in dfs_by_arm.values() for d in ds]
    paths = np.array(sorted(set.intersection(*sets))); paths = paths[first.loc[paths, "target"].isin(classes).to_numpy()]
    groups = SPLITS[src].loc[paths, "group_id"].astype(str).to_numpy(); codes, G = encode_groups(groups)
    idx = {c: i for i, c in enumerate(classes)}; cols = [f"probability_{c}" for c in classes]
    t = first.loc[paths, "target"].map(idx).to_numpy()
    arms = {}
    for a, dfs in dfs_by_arm.items():
        arms[a] = []
        for d in dfs:
            x = d.loc[paths]; assert (x.target.map(idx).to_numpy() == t).all()
            P = x[cols].to_numpy(float); P = P / P.sum(1, keepdims=True); arms[a].append((t, P))
    rng = np.random.default_rng(SEED); counts = rng.multinomial(G, np.full(G, 1 / G), size=B)  # one shared draw
    res = {a: replicates(codes, G, fs, counts, classes) for a, fs in arms.items()}
    out = dict(n_images=len(paths), n_groups=int(G), classes=classes, n_models=len(dfs_by_arm["none"]), arms={})
    for a, (pt, rep) in res.items():
        o = dict(metrics={m: dict(point=pt[m], lo=pct(rep[m], 2.5), hi=pct(rep[m], 97.5)) for m in pt})
        if a != "none":
            o["delta"] = {m: dict(point=pt[m] - res["none"][0][m], lo=pct(rep[m] - res["none"][1][m], 2.5), hi=pct(rep[m] - res["none"][1][m], 97.5)) for m in pt}
        out["arms"][a] = o
    return out

def strict(v):
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, dict): return {k: strict(x) for k, x in v.items()}
    if isinstance(v, list): return [strict(x) for x in v]
    return v

R = {}
for s in SRC:
    dfs = {a: load(s, a) for a in ARMS}
    allc = [c for c in CL if f"probability_{c}" in dfs["none"][0].columns]
    for v, cl in ({"normal_amd": ["normal", "amd"]} if s == "paima" else {"all": allc, "normal_amd": ["normal", "amd"]}).items():
        r = R[f"{s}|{v}"] = run_view(s, cl, dfs)
        ref = FINAL[f"loso_full|{s}|{v}"]["arms"]["loso-fill"]["metrics"]; m = r["arms"]["none"]["metrics"]
        r["sanity"] = dict(f1_diff=m["macro_f1"]["point"] - ref["macro_f1"]["point"], auroc_diff=m["macro_auroc"]["point"] - ref["macro_auroc"]["point"])
        print(s, v, "none F1 %.4f (final %.4f) AUROC %.4f (final %.4f)" % (m["macro_f1"]["point"], ref["macro_f1"]["point"], m["macro_auroc"]["point"], ref["macro_auroc"]["point"]), flush=True)
        assert abs(r["sanity"]["f1_diff"]) < 1e-4 and abs(r["sanity"]["auroc_diff"]) < 1e-4, "none arm does not reproduce data/final-scoring"

OUT = Path(__file__).resolve().parent
json.dump(strict(R), open(OUT / "results.json", "w"), indent=1, allow_nan=False)

N = {"kermany": "Kermany", "octdl": "OCTDL", "paima": "PAIMA", "duke": "Duke (1 model, eye-level groups)"}
pc = lambda m: f"{100*m['point']:.1f} [{100*m['lo']:.1f}, {100*m['hi']:.1f}]"
sg = lambda m: f"{100*m['point']:+.1f} [{100*m['lo']:+.1f}, {100*m['hi']:+.1f}]"
def tab(view, three):
    L = ["| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Δ AUROC | Recall " + ("N / A / D" if three else "N / A") + " |", "|---|---|---|---|---|---|---|"]
    for s in ["duke", "kermany", "octdl", "paima"]:
        k = f"{s}|{view}"
        if k not in R: continue
        r = R[k]
        for a in ARMS:
            o = r["arms"][a]; m = o["metrics"]
            rec = " / ".join(f"{100*m['recall_'+c]['point']:.1f}" for c in r["classes"])
            d1, d2 = ("—", "—") if a == "none" else (sg(o["delta"]["macro_f1"]), sg(o["delta"]["macro_auroc"]))
            L.append(f"| {N[s] if a=='none' else ''} | {a} | {pc(m['macro_f1'])} | {pc(m['macro_auroc'])} | {d1} | {d2} | {rec} |")
    return "\n".join(L)
head = """# AdaBN (all BatchNorm layers) on fill-aware LOSO checkpoints

Models: fused ResNet-50 fill-aware LOSO runs `artifacts/runs/fused/resnet50/loso-{kermany,octdl,paima}-local-20261008-174938-fold{1..5}`
(5 models each) and `loso-duke-local-20261008-174938` (1 model), each evaluated on its held-out source's full manifest. Weights are
not retrained.

Method (as the legacy 20261004 AdaBN report, `all` variant only): all 53 BatchNorm layers reset and refilled with cumulative-average
statistics (`momentum=None`) from the held-out source's unlabeled, unaugmented images, with patient two-fold cross-fit (default folds
and seed; no image is classified with statistics from its own patient). Command per run:
`uv run oct-classify train evaluate --checkpoint <run>/checkpoint-best.pt --source <held-out> --split-dir artifacts/splits --full-source --adapt-bn --adapt-bn-layers all --output <run>/evaluations/<held-out>-loso-adabn-all --device cuda`
(sweep script: `.scratchpad/adcaae9_adabn-eval/sweep.sh`; PAIMA with `--class-labels normal amd`).

Scoring as `data/paper/logit-bias/logit_bias.py`: splits `group_id` (Duke = eye) cluster bootstrap, 2000 draws (500 for AUROC), seed
20260913, one shared draw per replicate so deltas vs `none` are paired; model-mean metrics; 95% percentile CIs. `none` = stored
`<src>-loso/predictions.csv`; it reproduces the `loso_full` `loso-fill` values in `data/final-scoring/results.json` (checked to 1e-4 in the script).
Macro-F1 and macro-AUROC in percent [95% CI]; Δ is paired adabn-all minus none.

Reproduce (repo root, CPU only, after the sweep): `uv run python data/paper/adabn/score_adabn.py`.

"""
open(OUT / "table.md", "w").write(head + "## 3-class\n\n" + tab("all", True) + "\n\n## Binary Normal vs. AMD\n\n" + tab("normal_amd", False) + "\n")
