# Per-source logit-bias diagnostic on fill-aware LOSO predictions

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

Reproduce (repo root, CPU only): `uv run python data/paper/logit-bias/logit_bias.py && uv run python data/paper/logit-bias/mk_table.py`.

## 3-class

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Seen F1 | Bias (normal, AMD vs DME) | Recall N / A / D |
|---|---|---|---|---|---:|---|---|
| Duke (1 model, eye-level groups) | none | 89.7 [84.3, 93.6] | 97.8 [96.3, 99.0] | — | 95.2 | — | 97.7 / 93.6 / 75.8 |
|  | cross-fit | 90.3 [85.4, 93.8] | 97.5 [95.7, 98.9] | +0.6 [-1.5, +2.6] | 95.2 | -1.41, +0.06 | 93.9 / 93.9 / 82.5 |
|  | oracle | 90.5 [85.7, 94.0] | 97.6 [96.2, 99.0] | +0.8 [-1.1, +2.8] | 95.2 | -1.44, +0.07 | 94.2 / 94.1 / 82.6 |
| Kermany | none | 73.8 [72.6, 74.9] | 93.7 [93.1, 94.2] | — | 97.4 | — | 74.9 / 88.6 / 71.6 |
|  | cross-fit | 77.5 [76.2, 78.7] | 94.1 [93.6, 94.7] | +3.8 [+3.5, +4.0] | 97.4 | +0.53, -0.98 | 83.6 / 87.3 / 74.0 |
|  | oracle | 77.6 [76.2, 78.7] | 94.1 [93.6, 94.7] | +3.8 [+3.5, +4.0] | 97.4 | +0.52, -0.99 | 83.6 / 87.3 / 74.1 |
| OCTDL | none | 92.0 [90.0, 93.8] | 98.7 [97.9, 99.2] | — | 95.9 | — | 95.1 / 97.8 / 77.3 |
|  | cross-fit | 92.5 [90.8, 94.2] | 98.8 [98.1, 99.3] | +0.5 [-0.9, +1.9] | 95.9 | -0.56, -2.68 | 98.1 / 94.9 / 91.0 |
|  | oracle | 92.5 [90.8, 94.2] | 98.8 [98.1, 99.3] | +0.5 [-0.9, +1.8] | 95.9 | -0.57, -2.68 | 98.2 / 94.9 / 91.0 |

Grid-search in-sample macro-F1 upper bound (3-class bias, grid -4..4 step 0.25, mean over models):

| Held out | None | Best in-sample bias | Mean bias (normal, AMD) |
|---|---:|---:|---|
| Duke (1 model, eye-level groups) | 89.7 | 90.9 | -1.25, -1.50 |
| Kermany (20,000-row subsample per model) | 73.9 | 80.8 | +3.40, +1.50 |
| OCTDL | 92.0 | 93.8 | -0.40, -1.45 |

## Binary Normal vs. AMD

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Seen F1 | Gap closed | Bias (normal vs AMD) | Recall N / A |
|---|---|---|---|---|---:|---:|---|---|
| Duke (1 model, eye-level groups) | none | 97.1 [93.8, 99.5] | 99.3 [98.3, 100.0] | — | 98.5 | — | — | 98.6 / 94.9 |
|  | cross-fit | 96.9 [93.3, 99.5] | 99.0 [97.3, 99.9] | -0.1 [-1.1, +0.6] | 98.5 | -10% | -1.44 | 98.0 / 95.7 |
|  | oracle | 97.3 [94.1, 99.7] | 99.3 [98.3, 100.0] | +0.3 [-0.2, +0.8] | 98.5 | 19% | -1.58 | 98.4 / 96.0 |
| Kermany | none | 93.9 [93.6, 94.3] | 99.3 [99.2, 99.4] | — | 98.4 | — | — | 90.5 / 98.3 |
|  | cross-fit | 95.6 [95.3, 95.9] | 99.3 [99.2, 99.4] | +1.6 [+1.5, +1.8] | 98.4 | 37% | +0.72 | 94.1 / 97.6 |
|  | oracle | 95.5 [95.2, 95.8] | 99.3 [99.2, 99.4] | +1.6 [+1.4, +1.7] | 98.4 | 35% | +0.60 | 93.8 / 97.8 |
| OCTDL | none | 96.3 [95.0, 97.4] | 99.2 [98.8, 99.6] | — | 96.8 | — | — | 95.2 / 98.1 |
|  | cross-fit | 96.1 [94.5, 97.3] | 99.2 [98.8, 99.6] | -0.3 [-1.0, +0.4] | 96.8 | — | +1.55 | 98.5 / 96.9 |
|  | oracle | 96.2 [94.6, 97.3] | 99.2 [98.8, 99.6] | -0.2 [-0.9, +0.5] | 96.8 | — | +1.54 | 98.6 / 97.0 |
| PAIMA | none | 86.4 [85.0, 87.7] | 92.8 [91.6, 93.8] | — | 90.8 | — | — | 95.0 / 77.7 |
|  | cross-fit | 87.4 [86.1, 88.6] | 92.8 [91.6, 93.8] | +1.0 [+0.6, +1.5] | 90.8 | 23% | -1.33 | 91.4 / 83.4 |
|  | oracle | 87.5 [86.1, 88.7] | 92.8 [91.6, 93.8] | +1.1 [+0.6, +1.5] | 90.8 | 24% | -1.33 | 91.4 / 83.4 |

Grid-search in-sample macro-F1 upper bound (normal bias, grid -4..4 step 0.25, mean over models):

| Held out | None | Best in-sample bias | Mean bias (normal) |
|---|---:|---:|---|
| Duke (1 model, eye-level groups) | 97.1 | 97.3 | -1.50 |
| Kermany (20,000-row subsample per model) | 93.9 | 96.8 | +2.40 |
| OCTDL | 96.3 | 96.7 | +0.70 |
| PAIMA | 86.4 | 87.7 | -0.90 |
