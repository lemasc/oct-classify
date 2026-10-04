# Per-Source Logit Bias Diagnostic on ResNet-50 LOSO Predictions

## Scope and Provenance

This is experiment 7 in [cross-device-plan.md](../cross-device-plan.md). It asks whether the
leave-one-source-out (LOSO) gap is a decision-cutoff or label-prior shift, which a per-source class bias
would undo, or a feature problem, which it would not. No model was retrained or re-evaluated. It rescores
the saved predictions of the 16 median-padding LOSO checkpoints
(`fused/resnet50/loso-*-local-20261001-132356*`, `evaluations/<source>-loso/predictions.csv`).

```bash
uv run python .scratchpad/a7f8d54_logit-bias/logit_bias.py   # main table
uv run python .scratchpad/a7f8d54_logit-bias/f1_grid.py      # direct macro-F1 upper-bound check
```

The first script reuses `score_adabn.py` and `binary_rescore.py` for loading, views and the paired
bootstrap. Its full output is in `.scratchpad/a7f8d54_logit-bias/{logit_bias.json,table.md}`.

**This diagnostic uses the held-out source's labels.** It is an upper bound on what recalibration could
achieve, not a method that can be deployed on an unseen device.

## Method

| Variant | Bias fitted on | Scored on |
|---|---|---|
| `none` | n/a | All rows, original predictions |
| `bias-crossfit` | One half of the patients, then the other half | The other half (honest estimate) |
| `bias-oracle` | All rows | The same rows (in-sample upper bound) |

- **Bias.** An additive per-class shift of the log-probabilities, with the last class fixed to 0. The 3-class
  view has 2 free parameters (normal and AMD relative to DME). The binary view has 1 (normal relative
  to AMD), fitted on normal/AMD rows only. The shift minimises class-balanced cross-entropy with a small
  ridge penalty (L-BFGS), a smooth stand-in for macro-F1.
- **Cross-fit.** Whole patients, stratified by each patient's majority label, seed 20261004, the same
  scheme as the AdaBN experiment. One split is shared by all checkpoints. Each checkpoint gets its own
  bias per half.
- **Scoring.** Fold-averaged metrics with 95% patient-level cluster-bootstrap intervals, and a paired
  difference against the unadapted predictions (each replicate resamples patients once and scores both).
  "Gap closed" is the cross-fit gain divided by the seen-minus-held-out macro-F1 difference from
  [20261004-binary-rescore.md](20261004-binary-rescore.md).
- **Direct check.** `f1_grid.py` searches the 3-class bias on a grid (-4 to +4, step 0.25) for the
  best in-sample macro-F1 per checkpoint, because balanced cross-entropy is only a proxy. Kermany is
  subsampled to 20,000 rows per checkpoint for speed.
- **Numerical note.** Probabilities are stored in half precision. For the 3-class view they are floored at
  the smallest fp16 subnormal (6e-8) before taking logs. Only a few underflowed Kermany rows are affected.

## Results

Macro-F1 and macro-AUROC in percent, point estimate [95% CI]. Δ is the paired difference against `none`.
Recall is per class: normal / AMD / DME.

### 3-class

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Seen F1 | Bias (normal, AMD vs DME) | Recall N / A / D |
|---|---|---|---|---|---:|---|---|
| Duke (1 model) | none | 86.7 [80.8, 91.0] | 94.7 [90.8, 97.4] | — | 95.5 | — | 92.1 / 92.5 / 75.4 |
| | cross-fit | 84.9 [78.5, 89.5] | 93.7 [89.3, 96.8] | -1.8 [-3.1, -0.6] | | -0.09, -0.24 | 90.1 / 90.2 / 74.9 |
| | oracle | 86.5 [80.5, 90.7] | 94.7 [90.8, 97.4] | -0.3 [-0.9, +0.3] | | -0.15, -0.19 | 90.1 / 92.1 / 76.9 |
| Kermany | none | 73.2 [71.9, 74.3] | 91.5 [90.9, 92.2] | — | 95.6 | — | 77.1 / 86.0 / 65.5 |
| | cross-fit | 73.7 [72.4, 74.9] | 91.8 [91.1, 92.5] | +0.5 [+0.3, +0.8] | | +0.18, -0.77 | 79.7 / 84.3 / 68.5 |
| | oracle | 73.7 [72.5, 74.9] | 91.8 [91.1, 92.5] | +0.6 [+0.4, +0.8] | | +0.19, -0.76 | 79.7 / 84.4 / 68.5 |
| OCTDL | none | 91.9 [90.1, 93.6] | 98.8 [98.0, 99.3] | — | 96.8 | — | 96.7 / 97.3 / 77.6 |
| | cross-fit | 90.7 [88.9, 92.3] | 98.9 [98.2, 99.3] | -1.2 [-2.6, +0.1] | | -0.64, -2.37 | 98.3 / 93.9 / 89.0 |
| | oracle | 90.8 [89.0, 92.4] | 98.9 [98.2, 99.3] | -1.1 [-2.5, +0.2] | | -0.64, -2.37 | 98.5 / 93.9 / 88.8 |

Best in-sample macro-F1 from the direct grid search (`f1_grid.py`, mean over checkpoints):

| Held out | None | Best in-sample bias | Mean bias (normal, AMD) |
|---|---:|---:|---|
| Duke | 86.7 | 87.6 | +0.50, -1.00 |
| Kermany | 73.2 | 76.4 | +3.30, +1.50 |
| OCTDL | 91.9 | 93.4 | -0.70, -1.00 |

### Binary Normal vs. AMD

| Held out | Variant | Macro-F1 | Macro-AUROC | Δ macro-F1 | Seen F1 | Gap closed | Bias (normal vs AMD) | Recall N / A |
|---|---|---|---|---|---:|---:|---|---|
| Duke, B-scan (1 model) | none | 96.8 [93.8, 98.8] | 98.7 [96.8, 100.0] | — | 98.3 | — | — | 98.4 / 94.7 |
| | cross-fit | 95.3 [91.6, 97.9] | 97.3 [93.6, 99.9] | -1.5 [-2.6, -0.6] | | n/a | +0.42 | 97.2 / 93.1 |
| | oracle | 97.0 [94.0, 99.0] | 98.7 [96.8, 100.0] | +0.2 [+0.0, +0.4] | | n/a | +0.06 | 98.6 / 94.7 |
| Kermany | none | 93.2 [92.8, 93.6] | 99.0 [98.9, 99.1] | — | 98.4 | — | — | 89.3 / 98.2 |
| | cross-fit | 95.9 [95.6, 96.2] | 99.0 [98.9, 99.1] | +2.7 [+2.4, +3.0] | | 52% | +1.92 | 95.8 / 96.2 |
| | oracle | 95.9 [95.6, 96.2] | 99.0 [98.9, 99.1] | +2.7 [+2.5, +3.0] | | 52% | +1.92 | 95.8 / 96.2 |
| OCTDL | none | 96.6 [95.3, 97.7] | 99.4 [99.1, 99.7] | — | 96.9 | — | — | 96.8 / 97.9 |
| | cross-fit | 96.5 [95.0, 97.6] | 99.4 [99.1, 99.7] | -0.2 [-0.6, +0.3] | | n/a | +1.08 | 98.9 / 97.2 |
| | oracle | 96.4 [95.0, 97.6] | 99.4 [99.1, 99.7] | -0.2 [-0.7, +0.3] | | n/a | +1.08 | 98.9 / 97.2 |
| PAIMA | none | 86.2 [84.9, 87.4] | 92.7 [91.6, 93.8] | — | 90.3 | — | — | 88.0 / 84.4 |
| | cross-fit | 86.8 [85.4, 88.0] | 92.7 [91.6, 93.9] | +0.6 [+0.4, +0.8] | | 15% | -0.12 | 88.6 / 85.0 |
| | oracle | 86.8 [85.4, 88.0] | 92.7 [91.6, 93.9] | +0.6 [+0.4, +0.7] | | 14% | -0.12 | 88.5 / 85.0 |

"Gap closed" is n/a where the held-out score is already within about 1 point of the seen score (OCTDL) or
the denominator is unstable (Duke).

## Interpretation

- **The 3-class gap is not a cutoff or prior shift.** Even the macro-F1-optimal bias, fitted in-sample on the
  target's own labels, lifts Kermany only from 73.2 to 76.4 against a seen score of 95.6, and OCTDL and Duke
  by about 1 point. If recalibration could fix DME, an oracle bias would have found it. DME is a
  feature or representation problem on unseen devices. This agrees with the AdaBN result, where DME was the
  output that moved most and in the wrong direction.
- **Kermany's Normal-vs-AMD gap is about half cutoff shift.** A +1.9 shift on normal restores normal recall
  (89.3 to 95.8) and closes 52% of the binary gap (93.2 to 95.9 against 98.4 seen). AUROC stays at 99.0, so
  the model ranks Kermany well and places the threshold wrongly. The remaining half is not recoverable by
  a bias and is not explained by BatchNorm statistics (experiment 2).
- **PAIMA is mostly a ranking problem.** The bias closes 15% of the gap and the fitted shift is near zero.
  Its AUROC drop (95.6 seen, 92.7 held out) is untouched by any bias. Together with experiment 2, PAIMA is
  the one source where neither statistics nor cutoff explain the loss, which makes it the cleanest test of
  augmentation or pretraining.
- **OCTDL has no cutoff problem to fix.** Its binary score is already at the seen level. In 3-class the
  cross-fit bias raises DME recall (77.6 to 89.0) by trading away AMD recall (97.3 to 93.9) and does not
  improve macro-F1.
- **Priority.** The remaining plausible causes are data volume (experiment 4) and training-time changes
  (experiments 3 and 5). Label or calibration differences are not enough to explain the 3-class gap.

## Caveats

- Held-out Duke is a single model scored on 45 eyes (30 normal/AMD units). Its negative cross-fit deltas
  are within the noise expected from one model and from fitting on about half of 45 eyes.
- The bias uses target labels. It bounds what recalibration could do. It is not a deployable result.
- Kermany and OCTDL intervals average 5 checkpoints over many patients. They measure uncertainty from
  sampling patients, not from training randomness.
- Balanced cross-entropy is only a proxy for macro-F1. In-sample it made OCTDL's 3-class F1 slightly worse,
  which is why the direct grid search is reported. The grid search is in-sample and uses a Kermany subsample
  of 20,000 rows per checkpoint, so its Kermany number is approximate.
- A bias cannot change within-class ranking, so it leaves the binary AUROC unchanged by design. It can
  shift 3-class one-vs-rest AUROC slightly through the softmax.
