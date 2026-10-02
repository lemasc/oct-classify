# ResNet-50 Training Results: Median Padding

## Scope and Provenance

This report covers the three ResNet-50 campaigns retrained with median letterbox padding (`data.padding = "median"`), and compares each with its black-padding counterpart from [20260918-resnet50-training.md](20260918-resnet50-training.md). Everything else is unchanged between the paired configs: pretrained ResNet-50 at 224 px with the same augmentations, and the same splits, sources and seeds. Batch size is 16 for the baseline and fused campaigns and 12 for LOSO, so it divides evenly across three training sources.

| Campaign | Date | Config | Runs |
|---|---|---|---|
| Baseline: duke | 27 Sep 2026 | `configs/training/resnet50-median-pad.toml` | `duke/resnet50/local-20260927-duke-median-pad-fold{1..5}` |
| Fused Duke-CV | 27-28 Sep | `configs/training/resnet50-fused-median-pad.toml` | `fused/resnet50/local-20260927-fused-median-pad-fold{1..5}` |
| LOSO | 1 Oct | `configs/training/resnet50-loso-median-pad.toml` | `fused/resnet50/loso-*-local-20261001-132356*` |

The "old" side of every comparison is the black-pad run set in the 18 September report: `local-20260916-duke-fold*`, `local-20260917-184617-fold*` and `loso-*-local-20260918-082742*`. The Kermany, OCTDL and PAIMA single-source baselines are not retrained; their black-pad results stand.

Metrics are **accuracy / balanced accuracy / macro-F1 / macro-AUROC**, in percent, mean +/- SD (sample SD, n-1) across checkpoints. Classes are ordered `normal, amd, dme`; PAIMA is binary (`normal, amd`). Matrix rows are true classes and columns are predictions. Duke is evaluated at the B-scan level in the matrices; eye-level metrics are given separately. Matrix conventions are explained in the earlier report.

Durations are from preserved filesystem timestamps, with the same method as before. Baseline durations run from `config.json` creation to `history.jsonl` completion. Fused and LOSO durations run from local-log creation to log completion, so LOSO includes the held-out-source evaluations. Wall-clock times depend on what else was running on the GPU and are not a controlled comparison.

## Training Selection

`Best epoch` maximizes validation macro-F1 (zero-based). Baseline runs select on the Duke validation split. Fused and LOSO runs select on the unweighted mean of per-source validation macro-F1. `Epochs run` is the number of completed epochs including epoch 0.

| Campaign | Condition | Duration | Best epoch(s) | Best validation macro-F1 | Epochs run |
|---|---|---|---|---|---|
| Baseline: duke (folds 1-5) | black | 1m 28s, 1m 36s, 3m 41s, 1m 44s, 3m 20s | 0, 1, 16, 2, 12 | 92.02, 87.58, 93.28, 93.45, 100.00 | 11, 12, 27, 13, 23 |
| | median | 2m 08s, 2m 08s, 2m 09s, 1m 57s, 2m 15s | 4, 4, 4, 3, 4 | 92.98, 90.97, 90.05, 89.22, 100.00 | 15, 15, 15, 14, 15 |
| Fused Duke-CV (folds 1-5) | black | 45m 17s, 49m 38s, 51m 02s, 26m 05s, 39m 14s | 18, 22, 23, 6, 15 | 95.93, 96.02, 96.03, 95.67, 95.99 | 29, 33, 34, 17, 26 |
| | median | 35m 42s, 49m 53s, 56m 58s, 36m 32s, 43m 08s | 10, 19, 23, 11, 15 | 95.89, 95.86, 95.75, 95.73, 96.02 | 21, 30, 34, 22, 26 |
| LOSO hold out duke | black | 53m 39s | 29 | 96.57 | 40 |
| | median | 28m 55s | 7 | 95.88 | 18 |
| LOSO hold out kermany (folds 1-5) | black | 2h 08m 15s | 7, 20, 6, 22, 17 | 91.13, 89.79, 90.18, 90.66, 92.03 | 18, 31, 17, 33, 28 |
| | median | 2h 15m 36s | 28, 15, 4, 2, 24 | 90.66, 90.24, 90.42, 90.46, 91.66 | 39, 26, 15, 13, 35 |
| LOSO hold out octdl (folds 1-5) | black | 3h 12m 50s | 26, 22, 12, 10, 20 | 96.13, 96.17, 95.96, 95.73, 96.21 | 37, 33, 23, 21, 31 |
| | median | 3h 31m 15s | 16, 16, 17, 18, 17 | 95.98, 96.08, 96.07, 95.91, 96.14 | 27, 27, 28, 29, 28 |
| LOSO hold out paima (folds 1-5) | black | 3h 04m 51s | 13, 10, 22, 28, 17 | 97.13, 97.07, 97.19, 97.37, 97.25 | 24, 21, 33, 39, 28 |
| | median | 4h 00m 06s | 13, 31, 31, 17, 21 | 97.04, 97.19, 97.15, 97.05, 97.45 | 24, 42, 42, 28, 32 |

Median-pad Duke baselines peak consistently at epoch 3-4 (black pad: 0-16). Their best validation F1 is lower for folds 2-4 and higher for fold 1. Fused and LOSO best validation scores are within 0.2 points of black pad everywhere except LOSO Duke (95.88 vs 96.57).

## Evaluation Summary

### Baseline: Duke-Only Cross-Source Matrix

Each cell is macro-F1 (%), mean +/- SD across the five Duke outer-fold checkpoints. The Duke column is the fold's own held-out test split; the other columns evaluate each checkpoint on that source's fixed test split. Single-source baselines for the other three sources were not retrained.

| Train source | Padding | Duke | Kermany | OCTDL | PAIMA |
|---|---|---:|---:|---:|---:|
| duke | black | 93.39 +/- 5.69 | 23.41 +/- 1.26 | 30.52 +/- 4.07 | 59.03 +/- 9.25 |
| duke | median | 91.34 +/- 7.55 | 28.96 +/- 7.03 | 41.74 +/- 12.28 | 65.45 +/- 9.69 |

Full median-pad Duke-baseline test and cross-source metrics:

| Evaluated on | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| duke (own test) | 91.83 +/- 6.82 | 92.21 +/- 6.68 | 91.34 +/- 7.55 | 98.74 +/- 1.52 |
| kermany | 52.02 +/- 2.94 | 36.93 +/- 4.28 | 28.96 +/- 7.03 | 81.19 +/- 5.89 |
| octdl | 72.03 +/- 2.15 | 43.52 +/- 10.63 | 41.74 +/- 12.28 | 78.21 +/- 6.33 |
| paima | 66.99 +/- 7.86 | 67.11 +/- 7.58 | 65.45 +/- 9.69 | 77.90 +/- 4.82 |

(Black-pad Duke baseline for comparison: Kermany 49.88 / 33.60 / 23.41 / 72.38, OCTDL 72.11 / 34.70 / 30.52 / 71.99, PAIMA 62.08 / 62.54 / 59.03 / 75.61.) Duke eye-level macro-F1 is 97.71 +/- 5.11 with median pad vs 95.43 +/- 6.26 with black pad. The image-level Duke drop (-2.05) is well within one fold SD and the eye-level metric moves the other way; with 6 patients per fold neither is a reliable signal. Patient-level CIs and the Grad-CAM padding analysis are in [20260927-resnet50-median-pad-comparison.md](20260927-resnet50-median-pad-comparison.md) §1-2.

### Fused Duke-CV Test Matrix

Values are mean +/- SD across the five fused checkpoints. Kermany, OCTDL and PAIMA are scored once per fold on fixed test splits, so their aggregate matrices repeat the same samples across checkpoints and are descriptive only.

| Test source | Padding | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---|---:|---:|---:|---:|
| duke | black | 94.11 +/- 5.48 | 93.76 +/- 5.62 | 94.07 +/- 5.50 | 99.38 +/- 1.08 |
| | median | 95.75 +/- 4.14 | 95.45 +/- 4.31 | 95.61 +/- 4.31 | 99.37 +/- 1.18 |
| kermany | black | 96.84 +/- 1.56 | 96.08 +/- 2.00 | 96.59 +/- 1.65 | 99.82 +/- 0.16 |
| | median | 96.34 +/- 2.53 | 95.44 +/- 3.16 | 95.57 +/- 3.39 | 99.74 +/- 0.20 |
| octdl | black | 97.34 +/- 0.43 | 95.88 +/- 0.73 | 96.20 +/- 0.75 | 99.62 +/- 0.05 |
| | median | 97.58 +/- 0.51 | 96.79 +/- 2.23 | 96.84 +/- 1.20 | 99.59 +/- 0.13 |
| paima | black | 90.76 +/- 0.74 | 90.60 +/- 0.79 | 90.69 +/- 0.77 | 95.56 +/- 0.37 |
| | median | 90.41 +/- 0.49 | 90.23 +/- 0.51 | 90.33 +/- 0.51 | 95.55 +/- 0.63 |

Patient-clustered pooled macro-F1 with 95% bootstrap CI (`data/analysis/fused-cv-median-pad/`, vs `data/analysis/fused-cv/`):

| Set | Level | Black | Median |
|---|---|---:|---:|
| Duke | image | 93.98 [89.58, 97.32] | 95.46 [92.38, 97.89] |
| Duke | eye | 95.55 [88.25, 100.00] | 100.00 [100.00, 100.00] |
| Kermany | image | 96.59 [95.80, 97.30] | 95.57 [94.57, 96.47] |
| OCTDL | image | 96.20 [92.59, 98.59] | 96.84 [93.74, 98.88] |
| PAIMA | image | 90.69 [87.06, 93.64] | 90.33 [86.57, 93.36] |

Aggregate median-pad confusion matrices:

$$
C_{\mathrm{duke}} =
\begin{bmatrix}
1393 & 11 & 3 \\
6 & 692 & 25 \\
77 & 21 & 1003
\end{bmatrix}
\quad
C_{\mathrm{kermany}} =
\begin{bmatrix}
1111 & 33 & 106 \\
9 & 2476 & 15 \\
2 & 18 & 1230
\end{bmatrix}
$$

$$
C_{\mathrm{octdl}} =
\begin{bmatrix}
245 & 5 & 0 \\
20 & 900 & 0 \\
0 & 6 & 104
\end{bmatrix}
\quad
C_{\mathrm{paima}} =
\begin{bmatrix}
6166 & 234 \\
959 & 5081
\end{bmatrix}
$$

### LOSO Full-Source Evaluation

Each held-out source was never used for training or validation. Duke has one checkpoint (the other three sources do not contain Duke CV). The other targets have five checkpoints, one per Duke-CV fold of the training data. Matrices are sums over checkpoints, so for the five-checkpoint targets they repeat the held-out full source five times.

| Held-out source | Padding | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---|---:|---:|---:|---:|
| duke | black | 87.56 | 87.30 | 87.45 | 97.24 |
| | median | 86.51 | 86.68 | 86.73 | 94.68 |
| kermany | black | 80.63 +/- 4.03 | 77.62 +/- 4.22 | 73.96 +/- 3.96 | 92.22 +/- 2.72 |
| | median | 79.32 +/- 7.73 | 76.20 +/- 4.85 | 73.16 +/- 6.85 | 91.53 +/- 3.90 |
| octdl | black | 94.70 +/- 0.64 | 88.06 +/- 2.22 | 90.20 +/- 1.68 | 98.92 +/- 0.33 |
| | median | 95.52 +/- 0.70 | 90.53 +/- 2.30 | 91.90 +/- 1.54 | 98.78 +/- 0.64 |
| paima | black | 86.73 +/- 1.99 | 86.71 +/- 1.89 | 86.70 +/- 1.97 | 93.06 +/- 0.64 |
| | median | 86.23 +/- 2.73 | 86.18 +/- 2.64 | 86.18 +/- 2.73 | 92.69 +/- 1.29 |

Held-out Duke eye-level macro-F1 is 93.27 for both conditions.

Aggregate median-pad confusion matrices:

$$
C_{\mathrm{duke}} =
\begin{bmatrix}
1296 & 19 & 92 \\
11 & 669 & 43 \\
219 & 52 & 830
\end{bmatrix}
\quad
C_{\mathrm{kermany}} =
\begin{bmatrix}
193756 & 20171 & 37388 \\
2411 & 171327 & 25547 \\
10277 & 8843 & 36365
\end{bmatrix}
$$

$$
C_{\mathrm{octdl}} =
\begin{bmatrix}
1605 & 52 & 3 \\
125 & 5987 & 38 \\
26 & 139 & 570
\end{bmatrix}
\quad
C_{\mathrm{paima}} =
\begin{bmatrix}
37472 & 5133 \\
6298 & 34102
\end{bmatrix}
$$

Paired per-fold macro-F1 differences (median minus black), for the five-checkpoint targets: Kermany -9.2, -1.7, -2.0, +7.6, +1.3 (2/5 improved); OCTDL -1.1, +4.2, +2.8, +4.6, -2.0 (3/5); PAIMA +0.1, -3.2, +1.3, +2.6, -3.5 (3/5). Per-class recall (mean over checkpoints, normal / amd / dme): Kermany 79.2 / 86.1 / 67.6 black vs 77.1 / 86.0 / 65.5 median; OCTDL 96.6 / 97.1 / 70.5 vs 96.7 / 97.3 / 77.6; PAIMA 87.7 / 85.7 vs 88.0 / 84.4. Held-out Duke recall is 98.2 / 93.9 / 69.8 black vs 92.1 / 92.5 / 75.4 median.

## Interpretation

- **Median padding helps the single-source Duke baseline across sources.** Cross-source macro-F1 rises on Kermany (23.4 to 29.0), OCTDL (30.5 to 41.7) and PAIMA (59.0 to 65.5). The intervals are wide (fold SDs of 7-12 points), but the direction is consistent with the Grad-CAM finding that the black-pad Duke model put 80-95% of its attention on letterbox padding for Kermany and OCTDL. Duke in-domain performance is unchanged within noise.
- **Fused Duke-CV is a wash.** Duke improves (94.07 to 95.61 image-level; eye-level 95.5 to 100) and OCTDL improves slightly (96.20 to 96.84), while Kermany (96.59 to 95.57) and PAIMA (90.69 to 90.33) dip slightly. Every difference is smaller than the fold SD or CI width. The Kermany change comes mostly from more normal scans predicted as DME (31 to 106 in the aggregate matrix), which is worth watching but is a single set of five related checkpoints.
- **LOSO shows no padding effect either.** Held-out macro-F1 changes by -0.7 (Duke), -0.8 (Kermany), +1.7 (OCTDL) and -0.5 (PAIMA), all within fold-to-fold noise. Kermany's fold SD rose from 4.0 to 6.9 and its worst fold dropped 9.2 points; the paired differences are inconsistent in sign. The only systematic shift is OCTDL DME recall (70.5% to 77.6%), which is suggestive but rests on five related checkpoints. Grad-CAM on the LOSO checkpoints (§4 of the comparison report) shows padding mass of 2-22% in both conditions, against canvas padding shares of 27-63%, so with three training sources the padding shortcut is already absent.
- **Absolute generalization is unchanged.** Held-out Kermany remains the hardest target (73-74% macro-F1) and held-out DME recall on Kermany and Duke (65-76%) is the main weakness in both conditions. LOSO PAIMA still predicts about 6,300 of its 40,400 AMD scans (aggregate over five checkpoints) as normal, so sensitivity on PAIMA is the same issue as before.
- **Recommendation.** Keep median padding for any Duke-only or Duke-heavy training, where it removes a real shortcut and improves transfer. For fused and LOSO training it is neutral on the evidence here, so the choice can follow whichever default is simpler; do not claim a fused-training benefit from these runs.

## Caveats

- The five folds of each campaign share training data, so their SDs are rough and the five-checkpoint summaries are not independent observations. None of the differences between conditions would hold up under a formal test, except possibly the Duke-baseline cross-source gains, and those have wide intervals.
- Duke has 6 patients per fold (45 eyes pooled). Image-level Duke rows are correlated within eyes. The median-pad fused eye-level interval [100.00, 100.00] is a degenerate bootstrap on 45 eyes with no errors at the eye level and should not be read as a precise estimate.
- Held-out Duke in LOSO is one checkpoint and has no measure of variation. Its AUROC drop (97.24 to 94.68) should not be generalized.
- Fused and LOSO checkpoint selection uses an unweighted mean of per-source validation macro-F1, which weights Duke and OCTDL (small) equal to Kermany and PAIMA (large). This adds checkpoint-selection noise (see the memory note `fused-cv-unweighted-mean-noise`) and likely contributes to Kermany's fold swings.
- Durations are not a controlled comparison (shared GPU, different days).

## Source Artifacts

- Baseline (median pad): `artifacts/runs/duke/resnet50/local-20260927-duke-median-pad-fold{1..5}`
- Fused CV (median pad): `artifacts/runs/fused/resnet50/local-20260927-fused-median-pad-fold{1..5}`
- LOSO (median pad): `artifacts/runs/fused/resnet50/loso-*-local-20261001-132356*`, with held-out evaluations in each run's `evaluations/<source>-loso/metrics.json`
- Black-pad references: see the source artifacts of [20260918-resnet50-training.md](20260918-resnet50-training.md)
- Logs: `artifacts/local/` (baseline `resnet50-duke-median-pad-fold*`, fused `fused-median-pad-fold*`, LOSO `loso-*-local-20261001-132356.out`)
- Campaign analysis: `data/analysis/baseline-duke-median-pad/`, `data/analysis/fused-cv-median-pad/`. LOSO Grad-CAM outputs are in each run's `analysis/eval-<source>-loso/gradcam/`.
- Follow-up analysis: [20260927-resnet50-median-pad-comparison.md](20260927-resnet50-median-pad-comparison.md) covers patient-level CIs and Grad-CAM padding mass for the Duke baselines and the LOSO checkpoints.
