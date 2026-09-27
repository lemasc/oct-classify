# ResNet-50 Post-Training Analysis

## Scope and Provenance

This report analyzes the ResNet-50 runs covered by
[20260918-resnet50-training.md](20260918-resnet50-training.md):

- single-source baselines (16 September 2026)
- fused Duke-CV training (17 September)
- leave-one-source-out (LOSO) fused training (18 September)

It adds four things that report does not have:

1. confidence intervals over patients rather than images
2. calibration and decision-threshold analysis
3. error breakdowns by original label, cohort, duplicate cluster and patient
4. Grad-CAM attention

No model was retrained. Everything comes from each run's saved predictions and `checkpoint-best.pt`,
using `oct-classify analyze` at commit `4743768` with the analysis package uncommitted. Reading
instructions for every table are in [../analysis-guide.md](../analysis-guide.md).

Metrics are in percent. Intervals are 95% percentile cluster-bootstrap intervals (2000 replicates;
500 for AUROC; seed `20260927`) that resample `source:group_id` units: patients, or volumes for
Duke. Duke **eye-level** metrics average B-scan probabilities within each eye, then resample eyes.
Campaign aggregation uses one of two modes:

- **shared:** the checkpoints scored the same rows. One patient resample is applied to all of them,
  and the reported point is the mean across checkpoints.
- **pooled:** the checkpoints scored disjoint rows (the Duke CV outer folds), which are concatenated
  so each of the 45 Duke eyes counts once.

Recomputed point metrics match the saved `metrics-test.json` / `metrics.json` for all 136
prediction sets. The shared-mode means reproduce the training report's values exactly.

Reproduce with:

```bash
uv run oct-classify analyze run artifacts/runs/*/resnet50/*          # expand as an array in zsh
uv run oct-classify analyze campaign --name fused-cv artifacts/runs/fused/resnet50/local-20260917-184617-fold{1..5}
uv run oct-classify analyze gradcam <run>                             # per run, CUDA
```

Campaign outputs: `data/analysis/{baseline-duke,fused-cv,loso-duke,loso-kermany,loso-octdl,loso-paima}/`.

## Key Findings

1. **Duke is the least precisely measured source.** Even pooled over all five CV folds (45 eyes,
   3,231 B-scans), the fused model's Duke macro-F1 interval is 89.6–97.3 per B-scan and 88.3–100 per
   eye. Single-fold Duke intervals are far wider; for example, fused fold 1 is 91.5 [61.5, 98.8].
   Do not compare Duke folds or small Duke differences between models.
2. **The LOSO generalization gap is real and precisely estimated for the large sources.** Held-out
   Kermany is 73.96 [72.66, 75.20] macro-F1, against 96.59 [95.80, 97.30] in-domain for the fused
   model. The intervals are far apart.
3. **Much of the LOSO loss is miscalibration and threshold shift, not lost ranking.**
   - Held-out Kermany keeps 92.2 macro-AUROC but has 11.3 ECE.
   - Its disease-versus-normal specificity at the model's own decision is 79.2. A threshold tuned
     on the test labels (an upper bound, not a deployable result) reaches 90.2 with 88.8
     sensitivity.
   - Held-out Duke shows the opposite shift: sensitivity 83.3 at the model's decision versus 91.1
     at the tuned threshold.
4. **DME is the weakest class on every held-out source.** Held-out DME recall is 69.8 (Duke), 67.6
   (Kermany) and 70.5 (OCTDL), compared with 86–97 for AMD/CNV. The error direction depends on the
   source: Duke DME goes to normal, OCTDL DME goes to AMD.
5. **Within AMD, drusen is the problem.** PAIMA drusen B-scans have 79–80% recall against 95–98% for
   CNV.
6. **The Duke-only baselines attend to padding and fill on other sources.**
   - Evaluated on OCTDL and Kermany, they put 95% and 80% of their `layer4` Grad-CAM mass on the
     letterbox padding or fill.
   - That is well above those regions' 63% and 56% share of the image canvas, and far above the
     13–22% of models trained on those sources.
   - This matches the Duke baselines' cross-source collapse (Duke→Kermany macro-F1 23.41).
   - No other model or campaign shows this.
7. **Near-duplicates do not detectably inflate the scores.** Accuracy on images in exact or pHash
   duplicate clusters differs from the rest by at most about ±4 points, in both directions, with no
   consistent sign.

## 1. Patient-Level Uncertainty

### 1.1 In-domain test performance

For the fused runs, the Kermany, OCTDL and PAIMA rows are **shared** (five checkpoints on one fixed
test set). The Duke rows are **pooled** out-of-fold (45 eyes). `SD` is the checkpoint-to-checkpoint
standard deviation.

| Model | Test source | Units | Macro-F1 [95% CI] | SD | Balanced acc. [95% CI] | Macro-AUROC [95% CI] |
|---|---|---:|---|---:|---|---|
| Duke baseline (5 folds) | duke, B-scan | 45 | 93.30 [87.92, 97.57] | 5.69 | 93.06 [88.09, 97.60] | 98.82 [97.16, 99.78] |
| Duke baseline (5 folds) | duke, eye | 45 | 95.54 [88.19, 100.00] | 6.26 | 95.56 [88.89, 100.00] | 99.93 [99.60, 100.00] |
| Kermany baseline | kermany | 635 | 98.73 [98.00, 99.42] | — | — | 99.97 [99.94, 99.99] |
| OCTDL baseline | octdl | 99 | 97.05 [93.79, 99.19] | — | — | 99.48 [98.72, 99.93] |
| PAIMA baseline | paima | 65 | 89.98 [87.11, 92.37] | — | — | 95.42 [92.61, 97.16] |
| Fused CV | duke, B-scan | 45 | 93.98 [89.58, 97.32] | 5.50 | 93.46 [88.99, 97.15] | 99.45 [98.69, 99.91] |
| Fused CV | duke, eye | 45 | 95.55 [88.25, 100.00] | 10.08 | 95.56 [88.19, 100.00] | 99.93 [99.60, 100.00] |
| Fused CV | kermany | 635 | 96.59 [95.80, 97.30] | 1.65 | 96.08 [95.24, 96.87] | 99.82 [99.73, 99.90] |
| Fused CV | octdl | 99 | 96.20 [92.59, 98.59] | 0.75 | 95.88 [92.17, 98.75] | 99.62 [99.05, 99.98] |
| Fused CV | paima | 65 | 90.69 [87.06, 93.64] | 0.77 | 90.60 [87.20, 93.51] | 95.56 [92.95, 97.41] |

How to read this table:

- **Fused versus single-source:** the intervals overlap for Duke, OCTDL and PAIMA. Only Kermany's
  drop (98.73 → 96.59) is clearly outside both intervals.
- **Checkpoint variability:** for OCTDL and PAIMA the checkpoint SD (≈0.8) is much smaller than the
  patient-sampling interval. More checkpoints would not tighten those estimates; more test
  patients would.
- **Duke eye-level SD of 10.08:** this comes from fold 4, which has two wrong eyes out of nine
  (one AMD→DME, one DME→normal); the other folds classify all nine eyes correctly. The pooled
  estimate is the one to report.

### 1.2 Leave-one-source-out (held-out full source)

| Held-out source | Checkpoints | Units | Macro-F1 [95% CI] | SD | Balanced acc. [95% CI] | Macro-AUROC [95% CI] |
|---|---:|---:|---|---:|---|---|
| duke, B-scan | 1 | 45 | 87.45 [81.44, 92.23] | — | 87.30 [82.02, 91.79] | 97.24 [94.92, 98.92] |
| duke, eye | 1 | 45 | 93.27 [84.44, 100.00] | — | 93.33 [85.71, 100.00] | 100.00 [100.00, 100.00] |
| kermany | 5 | 5,419 | 73.96 [72.66, 75.20] | 3.96 | 77.62 [76.43, 78.69] | 92.22 [91.52, 92.83] |
| octdl | 5 | 637 | 90.20 [87.80, 92.22] | 1.68 | 88.06 [85.34, 90.61] | 98.92 [98.40, 99.35] |
| paima (normal/AMD) | 5 | 437 | 86.70 [85.23, 87.98] | 1.97 | 86.71 [85.25, 88.00] | 93.06 [91.80, 94.23] |

- **Kermany:** its checkpoint SD (3.96) is larger than the half-width of its patient interval (about
  1.3). Training variability, not the size of the test set, dominates its uncertainty.
- **Duke:** the eye-level AUROC of 100 means every eye is ranked correctly, yet eye-level macro-F1
  is only 93.27. The LOSO Duke errors are decision-threshold errors (see §2).
- **In-domain scores in the LOSO campaigns:** these stay within about 2 points of fused CV. For
  example, the Duke pooled eye-level result is 100.00 over 45 eyes in the LOSO-Kermany, OCTDL and
  PAIMA campaigns, and Kermany test macro-F1 is 94.65–97.20 against 96.59 for fused CV.

### 1.3 Cross-source baseline matrix with intervals

Macro-F1 [95% CI] of each single-source baseline on another source's test split.

- Duke rows and the Duke column pool the five fold-specific evaluations. They are therefore slightly
  different from the training report's per-fold means (89.23, 77.59, 75.07).
- The PAIMA-trained model is scored on normal/AMD only.

| Train \ Test | Duke (B-scan) | Duke (eye) | Kermany | OCTDL | PAIMA |
|---|---|---|---|---|---|
| duke | 93.30 [87.92, 97.57] | 95.54 [88.19, 100.00] | 23.41 [21.80, 24.97] | 30.52 [27.19, 33.22] | 59.03 [53.87, 64.02] |
| kermany | 89.43 [84.75, 93.20] | 95.54 [88.25, 100.00] | 98.73 [98.00, 99.42] | 91.36 [82.53, 97.98] | 82.79 [77.88, 87.21] |
| octdl | 74.07 [65.62, 80.61] | 86.35 [74.79, 95.50] | 79.94 [76.92, 82.77] | 97.05 [93.79, 99.19] | 69.06 [62.98, 75.18] |
| paima | 67.65 [54.02, 79.93] | 79.17 [62.96, 93.06] | 79.41 [75.97, 82.47] | 89.94 [83.16, 94.55] | 89.98 [87.11, 92.37] |

## 2. Calibration and Decision Thresholds

The columns are:

- **ECE:** top-label expected calibration error.
- **Conf., Acc.:** mean top-label confidence and accuracy.
- **Conf. wrong:** mean confidence of wrong predictions.
- **Sensitivity and specificity:** for disease versus normal, scored as `1 - p(normal)`. They are
  given at the model's own decision (argmax) and at a threshold tuned for F1 on the test labels.

Values are means across checkpoints. The tuned threshold is an optimistic bound used only to
measure threshold shift. **Spec. @ 95% sens.** is the specificity at the strictest threshold that
still reaches 95% sensitivity.

| Set | ECE | Conf. | Acc. | Conf. wrong | Sens. / spec. (argmax) | Sens. / spec. (tuned) | Tuned threshold | Spec. @ 95% sens. |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Fused CV, duke | 3.9 | 97.5 | 94.1 | 79.6 | 92.4 / 99.7 | 98.6 / 98.4 | 0.12 | 98.8 |
| Fused CV, kermany | 1.6 | 96.7 | 96.8 | 73.4 | 99.6 / 91.4 | 99.2 / 97.3 | 0.85 | 99.4 |
| Fused CV, octdl | 2.2 | 98.9 | 97.3 | 83.7 | 98.1 / 98.0 | 98.6 / 98.0 | 0.35 | 100.0 |
| Fused CV, paima | 3.4 | 93.5 | 90.8 | 81.5 | 84.9 / 96.3 | 86.8 / 95.2 | 0.40 | 68.3 |
| LOSO duke | 6.6 | 94.1 | 87.6 | 81.0 | 83.3 / 98.2 | 91.1 / 92.9 | 0.15 | 82.2 |
| LOSO kermany | 11.3 | 91.9 | 80.6 | 81.6 | 93.7 / 79.2 | 88.8 / 90.2 | 0.90 | 74.1 |
| LOSO octdl | 2.0 | 96.3 | 94.7 | 79.3 | 97.1 / 96.6 | 97.9 / 94.9 | 0.36 | 99.1 |
| LOSO paima | 4.0 | 90.3 | 86.7 | 78.3 | 85.7 / 87.7 | 84.1 / 90.7 | 0.59 | 52.9 |

- **Held-out Kermany is overconfident** (91.9 confidence against 80.6 accuracy). It calls too many
  normals diseased: specificity is 79.2, and 14.3% of normal B-scans go to DME (§3). The tuned
  threshold sits near 0.9, far from the implicit 0.5, which is a large prior or threshold shift.
- **Held-out Duke shifts the other way.** The tuned threshold is 0.15, and argmax misses disease
  (sensitivity 83.3), almost entirely DME called normal.
- **Wrong predictions are made with high confidence everywhere** (mean 73–84%). Temperature scaling
  alone will not turn them into abstentions.
- **PAIMA cannot reach high sensitivity cheaply.** At 95% sensitivity, specificity falls to 53–68%.
  This is a separability limit (drusen, §3), not only a threshold problem.
- **Fused Kermany already shows a mild in-domain version of the LOSO shift:** specificity 91.4 at
  argmax against 97.3 at the tuned threshold.

**Implication.** Fitting a temperature and a per-source threshold on held-out-source validation data
would recover part of the LOSO loss for Kermany and Duke. The runs do not save validation
predictions yet, so this cannot be evaluated honestly from the current artifacts.

## 3. Where the Errors Are

### 3.1 Recall by source-native label

Mean recall across checkpoints (fraction of the label predicted correctly), with the largest
misclassification destination.

| Set | Normal | AMD / CNV | Drusen | DME |
|---|---|---|---|---|
| Fused CV, duke | 99.7 | 95.2 | — | 86.4 (→ normal 12.3) |
| Fused CV, kermany | 91.4 (→ amd 6.2) | CNV 99.2 | 99.0 | 97.8 |
| Fused CV, octdl | 98.0 | 97.8 | — | 91.8 (→ amd 7.3) |
| Fused CV, paima | 96.3 | CNV 98.0 | 79.1 (→ normal 20.9) | — |
| LOSO duke | 98.2 | 93.9 | — | 69.8 (→ normal 24.7) |
| LOSO kermany | 79.2 (→ dme 14.3) | CNV 86.4 (→ dme 12.3) | 84.8 (→ dme 8.6) | 67.6 (→ normal 20.2) |
| LOSO octdl | 96.6 | 97.1 | — | 70.5 (→ amd 24.6) |
| LOSO paima | 87.7 (→ amd 12.3) | CNV 94.8 | 79.7 (→ normal 20.3) | — |

- **DME** is the least transferable class. Where its errors go depends on the source's DME
  appearance, which suggests the fused model's DME concept is anchored to the training sources.
- **Drusen** (early/dry AMD) is the dominant AMD failure in PAIMA, both in-domain and held out.
  Merging drusen and CNV into one `amd` class hides a roughly 15-point recall gap.
- **PAIMA cohort caveat:** PAIMA B-scans are labelled individually, so a `DRUSEN`-cohort eye
  legitimately contains normal B-scans. The drusen slice above uses the per-B-scan label, which is
  the right unit.

### 3.2 Error concentration across patients

Mean across checkpoints.

| Set | Errors | Groups with errors | Share held by worst 10% of groups | Worst 25% |
|---|---:|---:|---:|---:|
| Fused CV, duke | 39.6 | 4.0 / 9 | 0.49 | 0.92 |
| Fused CV, paima | 229.8 | 46.4 / 65 | 0.50 | 0.73 |
| LOSO duke | 402 | 27 / 45 | 0.50 | 0.87 |
| LOSO kermany | 19,611 | 3,594 / 5,419 | 0.53 | 0.78 |
| LOSO octdl | 90.6 | 68.6 / 637 | 0.94 | 1.00 |
| LOSO paima | 2,202 | 342 / 437 | 0.39 | 0.66 |

- **Kermany and PAIMA:** errors are spread across most patients (66–78% of groups have at least
  one). That points to a source-wide shift rather than a few bad patients.
- **OCTDL:** 94% of held-out errors sit in the worst 10% of patients, and most patients are
  error-free. Reviewing those patients' images and labels is the cheapest next step for OCTDL.
- **Fused Kermany and OCTDL test sets** (31.6 and 6.8 errors) are left out of the table. Fewer than
  10% of groups have any error, so the share is trivially 1.0.

### 3.3 Duplicate clusters

Accuracy on images that belong to an exact or pHash duplicate cluster (`artifacts/audits/`),
compared with those that do not:

| Set | In a duplicate cluster | Not in one |
|---|---:|---:|
| Fused CV, paima | 89.9 | 92.4 |
| LOSO paima | 86.3 | 87.6 |
| LOSO kermany | 82.9 | 79.5 |
| LOSO octdl | 97.2 | 94.2 |
| LOSO duke | 85.0 | 88.7 |
| Fused CV, duke | 95.2 | 93.6 |

The differences are small and change sign across sets. There is no evidence that retained
near-duplicates inflate the scores.

## 4. Grad-CAM Attention

Grad-CAM was computed for each set's gallery:

- **Errors:** up to 24 per true→predicted cell, the most confidently wrong first, at most 3 per
  patient.
- **Controls:** 24 random correct predictions per class.

Values below are mean shares of Grad-CAM mass for the predicted class, averaged over checkpoints.

- **Outside retina** means outside a heuristic band from the ILM to about 10 px below the RPE.
  Implausible masks are excluded.
- **Padding** means letterbox padding or detected fill wedges.

For reference, padding and fill make up 27% (Duke), 56% (Kermany), 63% (OCTDL) and 61% (PAIMA) of
the 224² canvas, so a model spreading attention evenly would put about that share there.

| Model on set | `layer4` outside retina, error / control | `layer3` outside retina, error / control | `layer4` padding, error / control | Randomization Spearman (`layer4`) |
|---|---|---|---|---:|
| Fused CV on duke | 51.5 / 46.4 | 46.7 / 33.4 | 6.5 / 2.2 | 0.13 |
| Fused CV on kermany | 65.1 / 51.2 | 69.1 / 39.7 | 13.3 / 4.2 | 0.07 |
| Fused CV on octdl | 50.1 / 46.4 | 50.7 / 40.4 | 12.2 / 13.1 | 0.17 |
| Fused CV on paima | 51.4 / 52.0 | 46.4 / 36.7 | 19.9 / 21.7 | 0.13 |
| LOSO on duke | 49.4 / 44.3 | 46.5 / 30.0 | 5.6 / 1.7 | 0.21 |
| LOSO on kermany | 60.7 / 45.6 | 52.3 / 38.8 | 13.3 / 4.0 | 0.09 |
| LOSO on octdl | 48.1 / 42.6 | 41.7 / 31.6 | 16.2 / 12.5 | 0.10 |
| LOSO on paima | 43.8 / 48.3 | 35.4 / 40.6 | 13.9 / 19.4 | 0.19 |
| Duke baseline on duke | 57.8 / 62.7 | 46.2 / 50.0 | 1.7 / 22.6 | 0.17 |
| Duke baseline on kermany | 93.5 / 90.4 | 82.9 / 75.2 | **79.8 / 71.5** | 0.23 |
| Duke baseline on octdl | 99.1 / 94.7 | 94.7 / 87.6 | **94.8 / 89.0** | −0.09 |
| Duke baseline on paima | 77.9 / 73.2 | 72.5 / 68.7 | 57.3 / 47.4 | 0.00 |
| Kermany baseline on other sources | 49.4 / 47.7 | 52.2 / 40.7 | 16.7 / 15.8 | 0.22 |
| OCTDL baseline on other sources | 58.1 / 56.2 | 51.9 / 50.3 | 15.3 / 13.6 | 0.27 |
| PAIMA baseline on other sources | 57.1 / 54.4 | 50.8 / 46.7 | 22.0 / 19.3 | 0.04 |

- **The Duke-baseline padding shortcut.**
  - On OCTDL and Kermany, the Duke-only models put 80–95% of attention on padding or fill, above
    those regions' 56–63% canvas share. Every other model puts 13–22% there on the same kinds of
    images.
  - The pattern holds for correct and wrong predictions, and at `layer3`.
  - On PAIMA the share (47–57%) is not above PAIMA's padding area. The shortcut is clearest for the
    sources whose letterbox geometry differs most from Duke's near-square 512×496 scans.
  - Duke's canvas has much less padding or fill (27% on average), so a Duke-only model rarely had
    to learn to ignore large padded regions. This is a plausible explanation, not a tested one.
- **Errors attend off-retina more than controls for held-out Kermany** (60.7 against 45.6 at
  `layer4`, 52.3 against 38.8 at `layer3`), and to a lesser degree for Duke and OCTDL. In the
  inspected Kermany galleries this is mostly attention in the choroid below the RPE. Held-out PAIMA
  is the exception: its errors are *more* on-retina than its controls. That fits drusen being a
  subtle RPE-level finding the model looks at but misjudges, but it has not been verified.
- **The sanity check passes.** Correlations between trained and top-randomized CAMs are low
  (medians −0.09 to 0.27), so the maps depend on the learned `layer4`/`fc` weights rather than on
  image edges.

## 5. Implications and Next Steps

1. **Report Duke as the pooled CV estimate**, both per B-scan and per eye, with intervals. Retire
   per-fold Duke comparisons.
2. **Save validation predictions during training**, then evaluate temperature scaling and
   per-source thresholds. §2 suggests this recovers part of the held-out Kermany and Duke loss;
   test-tuned numbers must not be reported as results.
3. **Change how padding is handled** before the ViT/SSL comparisons. Options: crop to content, pad
   with the image median, or randomize padding geometry during training. Then rerun
   `analyze gradcam` and check that the Duke-baseline padding share drops below the canvas share.
4. **Target DME transfer and drusen:**
   - report per-raw-label recall alongside macro-F1
   - consider DME-focused augmentation or class weighting
   - keep drusen versus CNV visible rather than merged in results tables
5. **Review the handful of OCTDL patients that hold almost all held-out OCTDL errors**, and check
   them for label or acquisition issues.

## 6. Caveats

- **Retina band:** it is a heuristic, checked visually on samples from all four sources. `layer4`
  maps are 7×7, so attention shares are coarse. Compare them between conditions rather than reading
  absolutes.
- **Grad-CAM sample size:** it covers at most 24 images per confusion cell and 24 controls per
  class for each checkpoint, not whole test sets.
- **Kermany and OCTDL** have no eye IDs, so their units are patients. PAIMA is labelled per
  B-scan, so its metrics are not eye diagnoses.
- **Shared-mode intervals** describe mean checkpoint performance on these test patients. They are
  not intervals for a new training run on new data.
- **Tuned thresholds** in §2 are fit on the evaluated labels and are upper bounds.

## Source Artifacts

- **Per-run bundles:** `artifacts/runs/*/resnet50/<run>/analysis/<set>/`, with Grad-CAM under
  `gradcam/`.
- **Campaign summaries:** `data/analysis/<campaign>/{summary.json,table.csv}`.
- **Interactive view:** `uv run --group notebook marimo edit notebooks/analysis_results.py`.
