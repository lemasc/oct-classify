import json
from pathlib import Path
R = json.load(open(Path(__file__).parent / "results.json"))
N = {"kermany": "Kermany", "octdl": "OCTDL", "paima": "PAIMA", "duke": "Duke (1 model, eye-level groups)"}
pc = lambda m: f"{100*m['point']:.1f} [{100*m['lo']:.1f}, {100*m['hi']:.1f}]"
sg = lambda m: f"{100*m['point']:+.1f} [{100*m['lo']:+.1f}, {100*m['hi']:+.1f}]"
def tab(view, three):
    hdr = "| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Seen F1 | " + ("" if three else "Gap closed | ") + ("Bias (normal, AMD vs DME)" if three else "Bias (normal vs AMD)") + " | Recall " + ("N / A / D" if three else "N / A") + " |"
    L = [hdr, "|---|---|---|---|---|---:|" + ("" if three else "---:|") + "---|---|"]
    for s in ["duke", "kermany", "octdl", "paima"]:
        k = f"{s}|{view}"
        if k not in R: continue
        r = R[k]; cl = r["classes"]
        for a, lab in [("none", "none"), ("bias-crossfit", "cross-fit"), ("bias-oracle", "oracle")]:
            o = r["arms"][a]; m = o["metrics"]
            rec = " / ".join(f"{100*m['recall_'+c]['point']:.1f}" for c in cl)
            d = "—" if a == "none" else sg(o["delta"]["macro_f1"])
            gc = "" if three else " " + ("—" if a == "none" or o["gap_closed"] is None else f"{100*o['gap_closed']:.0f}%") + " |"
            b = "—" if a == "none" else ", ".join(f"{x:+.2f}" for x in o["bias_mean"])
            L.append(f"| {N[s] if a=='none' else ''} | {lab} | {pc(m['macro_f1'])} | {pc(m['macro_auroc'])} | {d} | {100*r['seen_macro_f1']:.1f} |{gc} {b} | {rec} |" if not three else
                     f"| {N[s] if a=='none' else ''} | {lab} | {pc(m['macro_f1'])} | {pc(m['macro_auroc'])} | {d} | {100*r['seen_macro_f1']:.1f} | {b} | {rec} |")
    return "\n".join(L)
def grid(view):
    L = ["| Held out | None | Best in-sample bias | Mean bias (normal" + (", AMD)" if view == "all" else ")") + " |", "|---|---:|---:|---|"]
    for s in ["duke", "kermany", "octdl", "paima"]:
        k = f"{s}|{view}"
        if k not in R: continue
        g = R[k]["grid"]; b = f"{g['bias_normal']:+.2f}" + (f", {g['bias_amd']:+.2f}" if view == "all" else "")
        L.append(f"| {N[s]}{' (20,000-row subsample per model)' if g['subsampled'] else ''} | {100*g['none']:.1f} | {100*g['best']:.1f} | {b} |")
    return "\n".join(L)
head = """# Per-source logit-bias diagnostic on fill-aware LOSO predictions

Models: fused ResNet-50 fill-aware LOSO runs `artifacts/runs/fused/resnet50/loso-{kermany,octdl,paima}-local-20261008-174938-fold{1..5}`
(5 models each) and `loso-duke-local-20261008-174938` (1 model), fp32 predictions of the full held-out source
(`evaluations/<source>-loso/predictions.csv`). No retraining or re-inference. Uses held-out labels: an upper bound on
recalibration, not a deployable method.

Method (as the legacy 20261004 report): per-class additive log-probability bias (last class fixed to 0) minimising
class-balanced cross-entropy + ridge 1e-3 (L-BFGS). `cross-fit`: fit on one half of the patients/eyes, apply to the other, swap
(whole groups, stratified by majority label, seed 20260913). `oracle`: fit and score on all rows. Normal/AMD view: rows
of those two classes with probabilities renormalised, 1 free parameter. Scoring follows `data/final-scoring` (`score.py`):
splits `group_id` (Duke = eye) cluster bootstrap, 2000 draws (500 for AUROC), seed 20260913, one shared draw per
replicate so deltas are paired with `none`; model-mean metrics; 95% percentile CIs. The `none` arm reproduces the
`loso_full` fill-aware values in `data/final-scoring/results.json` exactly (max |diff| 0 at 4 decimals). Seen F1 = fused
fill-aware (exp. 4) test-split macro-F1 from the same file. Gap closed = Δ / (seen - none), n/a when |gap| < 1 pt. Macro-F1 and
macro-AUROC in percent [95% CI]; Δ is the paired difference vs `none`; bias is the mean over models (and halves).

Reproduce: `uv run python .scratchpad/9d08e53_logit-bias/logit_bias.py && python .scratchpad/9d08e53_logit-bias/mk_table.py`
(scripts kept in `data/paper/logit-bias/`; run from the repo root, CPU only).

"""
open(Path(__file__).parent / "table.md", "w").write(head + "## 3-class\n\n" + tab("all", True) + "\n\nGrid-search in-sample macro-F1 upper bound (3-class bias, grid -4..4 step 0.25, mean over models):\n\n" + grid("all") +
  "\n\n## Binary Normal vs. AMD\n\n" + tab("normal_amd", False) + "\n\nGrid-search in-sample macro-F1 upper bound (normal bias, grid -4..4 step 0.25, mean over models):\n\n" + grid("normal_amd") + "\n")
