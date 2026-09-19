# ResNet-50 Training Results

## Scope and Provenance

This report covers the completed local ResNet-50 campaigns retained under `artifacts/runs/`: single-source baselines (16 September 2026), fused five-fold Duke-CV training (17 September), and leave-one-source-out (LOSO) fused training (18 September). Metrics are reported as **accuracy / balanced accuracy / macro-F1 / macro-AUROC**, in percent. All runs use pretrained ResNet-50 at 224 px with the configured stochastic augmentations; the baseline and fused campaigns use batch size 16, while LOSO uses batch size 12 to split evenly across three training sources.

Classes are ordered `normal, amd, dme`; PAIMA is structurally binary (`normal, amd`). Confusion-matrix rows are true classes and columns are predictions. Duke is evaluated at the B-scan/image level in the matrices below; its eye-level metrics are retained separately in the source JSON and should be preferred for clinical-unit claims.

### Reading Confusion Matrices

Matrices use `[normal, amd, dme]` for both rows and columns, except PAIMA, which uses `[normal, amd]`. Rows are the ground-truth class and columns are the predicted class. The diagonal therefore contains correct predictions; off-diagonal cells are errors. For example, the LOSO Duke row `[272, 60, 769]` is the true-DME row: 769 DME scans were correctly predicted as DME, while 272 were predicted as normal and 60 as AMD.

For a three-class source, each compact matrix in this report is written in LaTeX as

$$
C =
\begin{bmatrix}
C_{\mathrm{normal},\mathrm{normal}} & C_{\mathrm{normal},\mathrm{amd}} & C_{\mathrm{normal},\mathrm{dme}} \\
C_{\mathrm{amd},\mathrm{normal}} & C_{\mathrm{amd},\mathrm{amd}} & C_{\mathrm{amd},\mathrm{dme}} \\
C_{\mathrm{dme},\mathrm{normal}} & C_{\mathrm{dme},\mathrm{amd}} & C_{\mathrm{dme},\mathrm{dme}}
\end{bmatrix},
$$

where the first subscript is the true class and the second is the predicted class. For PAIMA, omit the `dme` row and column:

$$
C_{\mathrm{PAIMA}} =
\begin{bmatrix}
C_{\mathrm{normal},\mathrm{normal}} & C_{\mathrm{normal},\mathrm{amd}} \\
C_{\mathrm{amd},\mathrm{normal}} & C_{\mathrm{amd},\mathrm{amd}}
\end{bmatrix}.
$$

Durations are reconstructed from preserved filesystem timestamps. Baseline durations run from `config.json` creation to `history.jsonl` completion and are close estimates of training time. Fused and LOSO durations run from local-log creation to log completion; LOSO durations include the subsequent held-out-source evaluations.

## Training Selection

`Best epoch` maximizes validation macro-F1 (zero-based). `Epochs run` is the number of completed epochs, including epoch 0.

| Campaign | Run(s) | Duration | Best epoch(s) | Best validation macro-F1 | Epochs run |
|---|---|---|---|---|---|
| Baseline: duke | `local-20260916-duke-fold1` through `fold5` | 1m 29s, 1m 36s, 3m 40s, 1m 44s, 3m 20s | 0, 1, 16, 2, 12 | 92.02, 87.58, 93.28, 93.45, 100.00 | 11, 12, 27, 13, 23 |
| Baseline: kermany | `local-20260916-kermany` | 1h 52m 43s | 10 | 97.68 | 21 |
| Baseline: octdl | `local-20260916-octdl` | 3m 24s | 27 | 96.66 | 38 |
| Baseline: paima | `local-20260916-paima` | 12m 16s | 5 | 89.49 | 16 |
| Fused Duke-CV | `local-20260917-184617-fold1` through `fold5` | 45m 16s, 49m 38s, 51m 02s, 26m 05s, 39m 14s | 18, 22, 23, 6, 15 | 95.93, 96.02, 96.03, 95.67, 95.99 | 29, 33, 34, 17, 26 |
| LOSO hold out duke | `loso-duke-local-20260918-082742` | 53m 39s | 29 | 96.57 | 40 |
| LOSO hold out kermany | `loso-kermany-local-20260918-082742-fold1` through `fold5` | 2h 08m 16s | 7, 20, 6, 22, 17 | 91.13, 89.79, 90.18, 90.66, 92.03 | 18, 31, 17, 33, 28 |
| LOSO hold out octdl | `loso-octdl-local-20260918-082742-fold1` through `fold5` | 3h 12m 51s | 26, 22, 12, 10, 20 | 96.13, 96.17, 95.96, 95.73, 96.21 | 37, 33, 23, 21, 31 |
| LOSO hold out paima | `loso-paima-local-20260918-082742-fold1` through `fold5` | 3h 04m 51s | 13, 10, 22, 28, 17 | 97.13, 97.07, 97.19, 97.37, 97.25 | 24, 21, 33, 39, 28 |

## Evaluation Summary

### Baseline Cross-Source Matrix

Each cell is macro-F1 (%). Duke training has five outer-fold checkpoints; its cells are mean +/- SD across folds. The diagonal is the source's held-out test split; off-diagonals are the retained cross-source evaluations.

| Train source | Duke | Kermany | OCTDL | PAIMA |
|---|---:|---:|---:|---:|
| duke | 93.39 +/- 5.69 | 23.41 +/- 1.26 | 30.52 +/- 4.07 | 59.03 +/- 9.25 |
| kermany | 89.23 | 98.73 | 91.36 | 82.79 |
| octdl | 77.59 | 79.94 | 97.05 | 69.06 |
| paima | 75.07 | 79.41 | 89.94 | 89.98 |

### Fused Duke-CV Test Matrix

Values are mean +/- SD across the five fused checkpoints. The Kermany, OCTDL, and PAIMA test sets are evaluated once per fold, so their aggregate confusion matrices repeat their fixed test samples across checkpoints and are descriptive only.

| Test source | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC | Aggregate confusion matrix |
|---|---:|---:|---:|---:|---|
| duke | 94.11 +/- 5.48 | 93.76 +/- 5.62 | 94.07 +/- 5.50 | 99.38 +/- 1.08 | $\begin{bmatrix}1403 & 3 & 1 \\ 6 & 687 & 30 \\ 143 & 15 & 943\end{bmatrix}$ |
| kermany | 96.84 +/- 1.56 | 96.08 +/- 2.00 | 96.59 +/- 1.65 | 99.82 +/- 0.16 | $\begin{bmatrix}1142 & 77 & 31 \\ 14 & 2478 & 8 \\ 0 & 28 & 1222\end{bmatrix}$ |
| octdl | 97.34 +/- 0.43 | 95.88 +/- 0.73 | 96.20 +/- 0.75 | 99.62 +/- 0.05 | $\begin{bmatrix}245 & 5 & 0 \\ 19 & 900 & 1 \\ 1 & 8 & 101\end{bmatrix}$ |
| paima | 90.76 +/- 0.74 | 90.60 +/- 0.79 | 90.69 +/- 0.77 | 95.56 +/- 0.37 | $\begin{bmatrix}6163 & 237 \\ 912 & 5128\end{bmatrix}$ |

### LOSO Full-Source Evaluation

Each held-out source was never used for training. Duke has one checkpoint because the training sources exclude Duke; the other targets have five checkpoints because their training sources include Duke CV. Values are mean +/- SD across checkpoints where applicable. Matrices are sums over checkpoints and therefore repeat the held-out full source for the three five-checkpoint conditions.

| Held-out source | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC | Aggregate confusion matrix |
|---|---:|---:|---:|---:|---|
| duke | 87.56 +/- 0.00 | 87.30 +/- 0.00 | 87.45 +/- 0.00 | 97.24 +/- 0.00 | $\begin{bmatrix}1381 & 25 & 1 \\ 33 & 679 & 11 \\ 272 & 60 & 769\end{bmatrix}$ |
| kermany | 80.63 +/- 4.03 | 77.62 +/- 4.22 | 73.96 +/- 3.96 | 92.22 +/- 2.72 | $\begin{bmatrix}199009 & 16253 & 36053 \\ 4780 & 171509 & 22996 \\ 11194 & 6777 & 37514\end{bmatrix}$ |
| octdl | 94.70 +/- 0.64 | 88.06 +/- 2.22 | 90.20 +/- 1.68 | 98.92 +/- 0.33 | $\begin{bmatrix}1604 & 56 & 0 \\ 161 & 5970 & 19 \\ 36 & 181 & 518\end{bmatrix}$ |
| paima | 86.73 +/- 1.99 | 86.71 +/- 1.89 | 86.70 +/- 1.97 | 93.06 +/- 0.64 | $\begin{bmatrix}37361 & 5244 \\ 5768 & 34632\end{bmatrix}$ |

## Interpretation

- Single-source baselines are strongest in-domain: Kermany reaches 98.73% macro-F1, OCTDL 97.05%, Duke 93.39% mean across outer folds, and PAIMA 89.98%. Their off-diagonal macro-F1 values are substantially lower, demonstrating material source shift rather than a uniformly transferable classifier.
- Fused training maintains strong in-domain performance across sources: Kermany reaches 96.59% mean macro-F1, OCTDL 96.20%, Duke 94.07%, and PAIMA 90.69%. Relative to the corresponding single-source diagonal, fusion modestly improves Duke and PAIMA but lowers Kermany and OCTDL; this trade-off is consistent with optimizing a shared, source-balanced representation rather than each source independently.
- LOSO is the appropriate generalization stress test because no held-out-source samples participate in fitting or validation. It is lower than fused in-domain performance for every source: Duke 87.45%, Kermany 73.96% +/- 3.96%, OCTDL 90.20% +/- 1.68%, and PAIMA 86.70% +/- 1.97% macro-F1. The especially large Kermany drop indicates that its source-specific image distribution remains poorly covered by the other three datasets.
- LOSO Duke has 272 DME scans predicted as normal, plus 60 as AMD, which accounts for most of its reduced recall. LOSO PAIMA has 5,768 AMD scans predicted as normal across the five repeated full-source evaluations, so adaptation or calibration should focus on recovering AMD sensitivity without degrading normal specificity.
- Results should not be compared as if all matrices use the same unit: Duke image-level rows are correlated within eyes, and repeated evaluations in the five-checkpoint summaries are not independent observations. For final claims, retain per-fold values, report Duke eye-level metrics alongside image-level metrics, and use the one-checkpoint LOSO Duke result separately from the five-fold LOSO summaries.

## Source Artifacts

- Baselines: `artifacts/runs/{duke,kermany,octdl,paima}/resnet50/local-20260916-*`
- Fused CV: `artifacts/runs/fused/resnet50/local-20260917-184617-fold{1..5}`
- LOSO: `artifacts/runs/fused/resnet50/loso-*-local-20260918-082742*`
- Per-epoch selection data: each run's `history.jsonl`; test and full-source evaluation data: `metrics-test.json` and `evaluations/*/metrics.json`.
