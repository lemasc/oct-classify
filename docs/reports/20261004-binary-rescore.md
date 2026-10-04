# Binary Normal-vs-AMD Rescoring of ResNet-50 Predictions

## Scope and Provenance

This is experiment 1 in [cross-device-plan.md](../cross-device-plan.md). It asks how much of the
leave-one-source-out (LOSO) gap reported in [20261002-resnet50-training.md](20261002-resnet50-training.md)
is specific to DME. No model was retrained or re-evaluated; it rescores the saved predictions of the
median-padding campaigns:

| Setting | Runs | Test rows |
|---|---|---|
| Fused CV (source seen in training) | `fused/resnet50/local-20260927-fused-median-pad-fold{1..5}` | Each source's test split (`predictions-test/<source>.csv`) |
| LOSO (source held out) | `fused/resnet50/loso-*-local-20261001-132356*` | The held-out source's full manifest (`evaluations/<source>-loso/predictions.csv`) |

```bash
uv run python .scratchpad/5fd59a7_binary-rescore/binary_rescore.py
```

The script reuses `oct_classify.analysis.loading.read_prediction_set` and
`oct_classify.analysis.bootstrap.bootstrap_metrics`. Its full output is in
`.scratchpad/5fd59a7_binary-rescore/{binary_rescore.json,table.md}`.

## Method

Each prediction set is scored in three views:

| View | Rows | Scoring |
|---|---|---|
| 3-class | All | Original 3-class argmax (reference; matches the earlier report) |
| 3-class, N/A only | Normal and AMD targets | Original 3-class argmax; a DME prediction counts as an error |
| Binary | Normal and AMD targets | `p(amd) / (p(normal) + p(amd))`, i.e. a softmax over the normal and AMD logits only |

PAIMA predictions are already Normal/AMD only, so all its views are identical. Intervals are 95%
patient/volume cluster bootstrap intervals (2,000 replicates; 500 for AUROC), averaged over the 5
checkpoints where they share test rows. Duke fused CV pools the 5 folds' own test volumes. Duke also
gets an eye-level row (mean B-scan probability per eye).

On held-out Kermany, the stored half-precision probabilities underflow to 0 for both normal and AMD on
0-246 normal/AMD images per checkpoint (at most 0.27%). These are scored as a 0.5 tie, which bounds
their effect at about 0.3 points.

## Results

### Binary vs. 3-class

Macro-F1, percent, point estimate [95% CI]. "Seen" is fused CV; "held out" is LOSO.

| Source | 3-class seen | 3-class held out | Binary seen | Binary held out |
|---|---|---|---|---|
| Duke, B-scan | 95.5 [92.4, 97.9] | 86.7 [80.7, 91.1] | 98.3 [95.7, 99.9] | 96.8 [94.0, 98.9] |
| Duke, eye | 100.0 [100.0, 100.0] | 93.3 [84.4, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Kermany | 95.6 [94.6, 96.5] | 73.2 [71.9, 74.3] | 98.4 [97.7, 99.0] | 93.2 [92.8, 93.6] |
| OCTDL | 96.8 [93.7, 98.9] | 91.9 [89.9, 93.6] | 96.9 [92.8, 99.3] | 96.6 [95.3, 97.6] |
| PAIMA | 90.3 [86.6, 93.4] | 86.2 [84.8, 87.4] | 90.3 [86.6, 93.4] | 86.2 [84.8, 87.4] |

### Binary detail (held out)

| Source | Units | Macro-AUROC | Recall normal | Recall AMD | Macro-AUROC seen |
|---|---:|---|---|---|---|
| Duke, B-scan | 30 | 98.7 [96.7, 100.0] | 98.4 [96.9, 99.6] | 94.7 [88.4, 99.6] | 99.8 [99.3, 100.0] |
| Kermany | 4,902 | 99.0 [98.8, 99.1] | 89.3 [88.9, 89.8] | 98.2 [97.9, 98.5] | 99.9 [99.9, 100.0] |
| OCTDL | 531 | 99.4 [99.1, 99.7] | 96.8 [95.1, 98.2] | 97.9 [97.0, 98.7] | 99.4 [98.5, 100.0] |
| PAIMA | 437 | 92.7 [91.5, 93.8] | 88.0 [86.1, 89.6] | 84.4 [82.2, 86.3] | 95.6 [92.9, 97.4] |

### Normal/AMD Images Lost to DME Predictions

Balanced accuracy on normal/AMD images. The gap between the two columns is the share of these images
that the 3-class model sends to DME.

| Source | 3-class, N/A only | Binary |
|---|---|---|
| Duke, B-scan, held out | 92.3 [87.4, 96.2] | 96.6 [93.4, 99.1] |
| Kermany, seen | 94.0 [92.7, 95.2] | 98.1 [97.3, 98.8] |
| Kermany, held out | 81.5 [80.9, 82.1] | 93.8 [93.5, 94.0] |
| OCTDL, held out | 97.0 [96.0, 97.8] | 97.4 [96.4, 98.2] |

## Interpretation

- **Most of the held-out-source gap is DME.** For OCTDL and Duke, Normal-vs-AMD performance on a held-out
  source is within the seen-source interval; their 3-class drop comes from DME. For Kermany, the
  macro-F1 gap shrinks from 22 points (3-class) to 5 (binary).
- **On held-out Kermany, the model over-predicts DME.** About 12 points of balanced accuracy on
  normal/AMD images are lost to DME predictions, against 4 points on the seen Kermany test split. With
  Kermany held out, the training DME comes from only 122 patients (15 Duke + 107 OCTDL), so this may
  be a data-volume effect as well as a device effect (experiment 4).
- **The remaining Kermany binary gap is a decision-cutoff problem, not a ranking problem.** AUROC stays at
  99.0, but normal recall falls from 96.7 to 89.3. A shift in intensity statistics or calibration
  would produce this pattern, and AdaBN (experiment 2) tests it directly.
- **PAIMA is the only source with a real Normal-vs-AMD shift.** Both AUROC and recall drop. Part of
  this is PAIMA's per-slice labeling (its "normal" slices can come from eyes with AMD).

## Caveats

- "Seen" and "held out" are scored on different rows: the fused CV test split vs. the held-out source's
  full manifest. The comparison is between settings, not paired on images.
- This rescores a model trained on 3 classes. A model trained only on Normal vs AMD might behave differently;
  experiment 6 would test that.
- Duke has 30 normal/AMD volumes and saturates at the eye level, so it supports these conclusions but cannot
  decide them on its own. Duke held out is a single model.
- The 3-class, N/A-only view does not report macro-F1, because DME has predictions but no targets
  there, which pins it near 64 regardless of model quality.
