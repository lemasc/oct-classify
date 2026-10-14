# AdaBN (all BatchNorm layers) on fill-aware LOSO checkpoints

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

## 3-class

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Δ AUROC | Recall N / A / D |
|---|---|---|---|---|---|---|
| Duke (1 model, eye-level groups) | none | 89.7 [84.3, 93.6] | 97.8 [96.3, 99.0] | — | — | 97.7 / 93.6 / 75.8 |
|  | adabn-all | 90.0 [84.8, 93.7] | 96.5 [94.1, 98.5] | +0.3 [-0.9, +1.5] | -1.3 [-2.8, -0.2] | 95.9 / 94.2 / 78.4 |
| Kermany | none | 73.8 [72.6, 74.9] | 93.7 [93.1, 94.2] | — | — | 74.9 / 88.6 / 71.6 |
|  | adabn-all | 76.5 [75.1, 77.7] | 94.0 [93.5, 94.5] | +2.7 [+2.2, +3.2] | +0.4 [+0.2, +0.5] | 82.1 / 82.1 / 81.6 |
| OCTDL | none | 92.0 [90.0, 93.8] | 98.7 [97.9, 99.2] | — | — | 95.1 / 97.8 / 77.3 |
|  | adabn-all | 84.5 [82.1, 86.7] | 98.0 [97.3, 98.6] | -7.5 [-9.9, -5.4] | -0.7 [-1.0, -0.3] | 91.6 / 89.0 / 95.4 |

## Binary Normal vs. AMD

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Δ AUROC | Recall N / A |
|---|---|---|---|---|---|---|
| Duke (1 model, eye-level groups) | none | 97.1 [93.8, 99.5] | 99.3 [98.3, 100.0] | — | — | 98.6 / 94.9 |
|  | adabn-all | 97.4 [94.3, 99.6] | 98.8 [97.1, 99.8] | +0.3 [+0.1, +0.7] | -0.5 [-1.4, +0.0] | 98.6 / 95.7 |
| Kermany | none | 93.9 [93.6, 94.3] | 99.3 [99.2, 99.4] | — | — | 90.5 / 98.3 |
|  | adabn-all | 95.7 [95.4, 95.9] | 99.3 [99.2, 99.3] | +1.7 [+1.6, +1.9] | -0.1 [-0.1, -0.0] | 94.2 / 97.6 |
| OCTDL | none | 96.3 [95.0, 97.4] | 99.2 [98.8, 99.6] | — | — | 95.2 / 98.1 |
|  | adabn-all | 96.6 [95.2, 97.7] | 99.3 [98.9, 99.6] | +0.2 [-0.3, +0.8] | +0.1 [-0.1, +0.2] | 98.3 / 97.5 |
| PAIMA | none | 86.4 [85.0, 87.7] | 92.8 [91.6, 93.8] | — | — | 95.0 / 77.7 |
|  | adabn-all | 85.4 [84.1, 86.6] | 92.8 [91.6, 93.8] | -1.0 [-1.8, -0.1] | -0.0 [-0.3, +0.2] | 84.1 / 86.8 |
