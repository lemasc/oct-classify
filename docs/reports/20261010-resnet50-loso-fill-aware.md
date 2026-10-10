# ResNet-50 LOSO with Fill-Aware Padding, Patient-Level CIs

## Scope and Provenance

This report covers the leave-one-source-out (LOSO) campaign `local-20261008-174938`, trained with `data.padding = "fill_aware"` (`configs/training/resnet50-loso-fill-aware.toml`, 6 data-loader workers), and compares it with the two earlier LOSO campaigns:

| Campaign | Run id | Padding | Config |
|---|---|---|---|
| Black pad | `local-20260918-082742` | black letterbox | `resnet50-loso.toml` |
| Median pad | `local-20261001-132356` | median | `resnet50-loso-median-pad.toml` |
| Fill-aware | `local-20261008-174938` | fill-aware | `resnet50-loso-fill-aware.toml` |

The configs differ in `data.padding`, plus `num_workers` in the fill-aware config. Splits, sources and seed (20260913) are the same. Each held-out source is scored on its full manifest (`train evaluate --full-source`). Held-out Duke has one model. The other targets have 5 models, one per Duke-CV fold of the training data. Runs live in `artifacts/runs/fused/resnet50/loso-*-local-20261008-174938*`, with held-out results in each run's `evaluations/<source>-loso/`.

The campaign was run in three pieces. It stopped in OCTDL fold 3 on a validation NaN ([20261009-fp16-validation-nan.md](20261009-fp16-validation-nan.md)). After the fp32-evaluation fix, `.scratchpad/02da1c8_loso-continue/continue.sh` finished OCTDL folds 3-5, the OCTDL evaluations and the full PAIMA hold-out. Everything completed with no non-finite losses.

**Evaluation precision differs between campaigns.** The black-pad and median-pad numbers were computed with fp16 autocast. Fill-aware evaluation and validation ran in fp32. Training numerics are unchanged. The effect on metrics is probably small, but it is a confound for the comparison.

## Method

Group-level bootstrap, following the project requirement that error bars use independent units, not B-scans:

- Resampling unit is `group_id` from `artifacts/splits/<source>.jsonl`: eye for Duke (45), patient for Kermany (5,419) and OCTDL (637), volume for PAIMA (437). 2,000 draws, seed 20261010.
- Each draw resamples held-out groups once. The metric is then computed for every model and averaged over the models. The same draw is used for all three campaigns, so paired deltas share the held-out sampling noise.
- Macro-F1 and balanced accuracy use the predicted labels in `predictions.csv`, not recomputed thresholds. Point estimates match `metrics.json`.
- The CIs cover **held-out sampling only**. They do not cover training-seed or fold variation. The fold range is reported separately.

Script: `.scratchpad/02da1c8_loso-summary/loso_ci.py` (ad hoc, not in the package). Full output including accuracy: `.scratchpad/02da1c8_loso-summary/loso_ci.json`.

## 1. Held-Out Results Across Padding Conditions

Macro-F1 and balanced accuracy in percent: mean over models, with the group-bootstrap 95% CI. Fold range is the min-max macro-F1 across the 5 models.

| Held out | Groups (images) | Padding | Macro-F1 [95% CI] | Balanced acc [95% CI] | Fold range (macro-F1) |
|---|---:|---|---|---|---|
| Duke | 45 (3,231) | black | 87.4 [81.1, 92.1] | 87.3 [82.0, 91.7] | n/a |
| | | median | 86.7 [80.6, 91.0] | 86.7 [80.9, 91.3] | n/a |
| | | fill-aware | 89.7 [84.3, 93.7] | 89.1 [84.0, 93.1] | n/a |
| Kermany | 5,419 (101,217) | black | 74.0 [72.7, 75.2] | 77.6 [76.5, 78.8] | 67.3-77.0 |
| | | median | 73.2 [71.9, 74.4] | 76.2 [75.0, 77.4] | 65.5-81.1 |
| | | fill-aware | 73.8 [72.5, 75.0] | 78.4 [77.4, 79.3] | 66.4-78.7 |
| OCTDL | 637 (1,709) | black | 90.2 [88.0, 92.3] | 88.1 [85.5, 90.6] | 88.3-92.0 |
| | | median | 91.9 [90.0, 93.6] | 90.5 [88.2, 92.7] | 90.0-93.3 |
| | | fill-aware | 92.0 [90.1, 93.8] | 90.1 [87.7, 92.4] | 91.5-92.8 |
| PAIMA | 437 (16,601) | black | 86.7 [85.3, 88.0] | 86.7 [85.3, 88.0] | 83.6-88.5 |
| | | median | 86.2 [84.8, 87.4] | 86.2 [84.8, 87.4] | 82.9-89.4 |
| | | fill-aware | 86.4 [84.9, 87.7] | 86.3 [85.0, 87.6] | 83.8-88.7 |

Mean ± SD across the 5 models, all four metrics, for fill-aware (same format as the earlier training reports; macro-AUROC is image-level):

| Held out | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| duke | 89.35 | 89.07 | 89.66 | 97.78 |
| kermany | 79.94 ± 6.49 | 78.36 ± 1.50 | 73.77 ± 5.15 | 93.66 ± 0.93 |
| octdl | 95.49 ± 0.30 | 90.06 ± 0.53 | 92.02 ± 0.57 | 98.68 ± 0.53 |
| paima | 86.58 ± 1.86 | 86.35 ± 1.92 | 86.39 ± 1.97 | 92.81 ± 0.98 |

Held-out Duke eye-level macro-F1 is 93.27, identical to the black-pad and median-pad campaigns (45 eyes: 12 of 15 DME eyes correct, all normal and AMD eyes correct).

## 2. Paired Differences

Macro-F1 difference in percentage points, with the paired group-bootstrap 95% CI. Bold means the CI excludes 0.

| Held out | Median − black | Fill-aware − black | Fill-aware − median |
|---|---|---|---|
| Duke | −0.7 [−3.8, +2.2] | **+2.2 [+0.6, +4.2]** | **+2.9 [+0.1, +5.9]** |
| Kermany | **−0.8 [−1.1, −0.5]** | −0.2 [−0.5, +0.1] | **+0.6 [+0.3, +0.9]** |
| OCTDL | **+1.7 [+0.6, +2.7]** | **+1.8 [+0.7, +2.9]** | +0.1 [−0.7, +0.9] |
| PAIMA | **−0.5 [−1.0, −0.0]** | −0.3 [−1.1, +0.5] | +0.2 [−0.5, +0.9] |

Balanced-accuracy differences follow the same pattern, with one addition: on Kermany, fill-aware is +0.7 [+0.3, +1.2] over black pad and +2.2 [+1.8, +2.6] over median pad.

## 3. Comparison With Earlier Experiments

How the LOSO numbers sit against the other experiment families on the same sources. Macro-F1 in percent. The earlier rows are copied from [20261002-resnet50-training.md](20261002-resnet50-training.md) and [20260927-resnet50-median-pad-comparison.md](20260927-resnet50-median-pad-comparison.md). Fill-aware was only trained for LOSO. The Duke-only baseline and fused Duke-CV have no fill-aware runs.

| Target | Duke-only, black | Duke-only, median | Fused Duke-CV, black | Fused Duke-CV, median | LOSO, black | LOSO, median | LOSO, fill-aware |
|---|---:|---:|---:|---:|---:|---:|---:|
| Duke | 93.4 ± 5.7 | 91.3 ± 7.6 | 94.1 ± 5.5 | 95.6 ± 4.3 | 87.4 | 86.7 | 89.7 |
| Kermany | 23.4 ± 1.3 | 29.0 ± 7.0 | 96.6 ± 1.7 | 95.6 ± 3.4 | 74.0 ± 4.0 | 73.2 ± 6.8 | 73.8 ± 5.2 |
| OCTDL | 30.5 ± 4.1 | 41.7 ± 12.3 | 96.2 ± 0.8 | 96.8 ± 1.2 | 90.2 ± 1.7 | 91.9 ± 1.5 | 92.0 ± 0.6 |
| PAIMA | 59.0 ± 9.3 | 65.5 ± 9.7 | 90.7 ± 0.8 | 90.3 ± 0.5 | 86.7 ± 2.0 | 86.2 ± 2.7 | 86.4 ± 2.0 |

How to read the columns:

- **Duke-only** is cross-source for Kermany, OCTDL and PAIMA (model never saw the target). Duke is the model's own test split.
- **Fused Duke-CV** is in-domain for every column: all four sources are in training and the target is that source's own test split. It is an upper bound for the held-out columns.
- **LOSO** is out-of-source for every column. Duke's value comes from one model, the others are 5-model means.

What the table shows:

- Training on three sources lifts held-out macro-F1 far above the Duke-only baseline: Kermany 23-29 → 73-74, OCTDL 31-42 → 90-92, PAIMA 59-65 → 86-87. This is a much bigger effect than any padding choice, and none of the paddings changes the ordering.
- The remaining gap to in-domain fused training is about 22 points on Kermany, 4 on OCTDL and 4 on PAIMA. Kermany is the source where adding the others does not close the gap. Held-out Duke (87-90 vs 94-96) cannot be judged reliably from a single model and 45 eyes.
- PAIMA's 86-87 held-out vs 90-91 in-domain is a small gap and both sit near the label-noise ceiling noted in the earlier review (Duke is labeled per eye, PAIMA per slice).

## 4. Interpretation

- **Fill-aware padding gives a measurable gain on held-out Duke (+2.2 over black, +2.9 over median) and on OCTDL (+1.8 over black), and is neutral on Kermany and PAIMA.** OCTDL's gain is shared with median padding (fill-aware vs median +0.1), so it is not evidence for fill-aware specifically.
- **Duke's gain has the weakest support.** It is one model, so training-seed variance is not measured. The CI covers only which 45 eyes were sampled. The lower bound against median is +0.1 points. Treat it as suggestive.
- **Kermany's differences are statistically significant but small (≤0.8 F1 points), and the models' fold range is 10-16 points wide.** Training-seed or checkpoint-selection variance dominates, so the CI overstates how much these deltas mean. Fill-aware lifts Kermany balanced accuracy (+2.2 over median) while macro-F1 is flat, i.e. it shifts the errors between classes rather than reducing them. I have not broken this down per class.
- **Fill-aware narrows the fold spread on OCTDL (88.3-92.0 black, 90.0-93.3 median, 91.5-92.8 fill-aware) and on Kermany balanced accuracy (fold range 75.7-79.3).** Only five related models, so this is descriptive.
- **No padding condition changes the overall picture.** Held-out Kermany remains the hardest target at about 74 macro-F1. Held-out PAIMA stays near 86-87.

## 5. Caveats

- The CIs reflect held-out sampling only. The 5 models of a campaign share training data and use the same seed, so fold-to-fold variation is a lower bound on training variance, and none of the paired CIs include it.
- Duke has 45 eyes, labeled at eye level and propagated to B-scans. Its CIs are wide because of that, and it has one model per campaign. Duke and PAIMA use different labeling units (eye vs per-slice), so cross-source results partly measure that mismatch.
- fp16 vs fp32 evaluation, as noted at the top. The 6-worker change is also part of the fill-aware config. Neither should affect classification, but they are not separated from the padding change.
- Macro-F1 and balanced accuracy are computed over the labels present in the held-out source (PAIMA is binary). Group bootstrap draws containing a single class are skipped; I did not count how many were skipped for Duke.
- The earlier-experiment rows in §3 are copied from earlier reports and use their own aggregation (mean ± sample SD over folds), not the bootstrap. They are not directly comparable to the CI columns in §1.
- Grad-CAM padding mass was not recomputed for the fill-aware LOSO checkpoints. The earlier finding that LOSO models show no padding shortcut (comparison report §4.2) has not been rechecked for this campaign.

## Source Artifacts

- Runs: `artifacts/runs/fused/resnet50/loso-{duke,kermany,octdl,paima}-local-20261008-174938*`
- Held-out predictions and metrics: each run's `evaluations/<source>-loso/{metrics.json,predictions.csv}`
- Logs: `artifacts/local/loso-*-local-20261008-174938*.out`
- Continuation script: `.scratchpad/02da1c8_loso-continue/continue.sh`
- Bootstrap script and output: `.scratchpad/02da1c8_loso-summary/loso_ci.py`, `loso_ci.json`
