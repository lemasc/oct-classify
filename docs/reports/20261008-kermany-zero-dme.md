# Fused CV with Zero Kermany DME in Training

## Scope and Provenance

This is experiment 4b in [cross-device-plan.md](../cross-device-plan.md). The DME-cut run
([20261005-kermany-dme-cut.md](20261005-kermany-dme-cut.md)) still trained on 122 Kermany DME patients, so it could
not separate an unseen device from unseen in-device DME. This run removes all Kermany DME from training and keeps
Kermany normal and AMD, so the Kermany device is seen but its DME never is.

- **The cut.** All Kermany train-split DME removed (571 patients, 9,220 images). Kermany normal and AMD, all
  validation and test splits, and the other three sources are unchanged (`.scratchpad/1e23970_kermany-dme0/make_splits.py`,
  splits in `artifacts/splits/duke-cv-kermany-dme0`). The only DME left in training is OCTDL + Duke (122 patients).
- **Training.** `configs/training/resnet50-fused-median-pad.toml`, 5-fold Duke CV, local RTX 3060
  (`artifacts/runs/fused/resnet50/kdme0-local-20261008-011217-fold{1..5}`, 01:12 to 02:58).
- **Reference.** The fused median-pad CV run, `local-20260927-fused-median-pad-fold{1..5}`.

```bash
uv run python .scratchpad/1e23970_kermany-dme0/bootstrap_dme0.py   # writes bootstrap_dme0.json
uv run python .scratchpad/1e23970_kermany-dme0/bias_dme0.py        # bias follow-up, writes table.md
```

## Method

Identical to the DME-cut report: saved test predictions only, paired patient-level cluster bootstrap (2,000
replicates, 500 for AUROC, seed 20261004), fold-mean metrics for Kermany/OCTDL/PAIMA, pooled out-of-fold for Duke.
Both scripts are the experiment 4 scripts with the arm swapped.

## Results

Percent; Δ = zero-DME minus reference with its 95% interval. **Bold** marks intervals that exclude 0.

### 3-class

| Source (units) | Metric | Reference | Zero DME | Δ |
|---|---|---:|---:|---|
| Kermany (635 patients) | Macro-F1 | 95.6 | 90.0 | **-5.6 [-6.7, -4.5]** |
| | Macro-AUROC | 99.7 | 98.9 | **-0.8 [-1.1, -0.5]** |
| | DME recall | 98.4 | 95.8 | **-2.6 [-4.2, -1.2]** |
| OCTDL (99) | Macro-F1 | 96.8 | 94.4 | -2.4 [-6.5, +0.4] |
| | DME recall | 94.5 | 98.2 | +3.6 [-2.1, +9.5] |
| Duke (45 eyes) | Macro-F1 | 95.5 | 89.7 | **-5.7 [-12.2, -0.5]** |
| | Macro-AUROC | 99.4 | 97.3 | **-2.1 [-4.2, -0.4]** |
| | DME recall | 91.1 | 74.5 | **-16.6 [-33.9, -0.3]** |

### Binary Normal vs. AMD

| Source (units) | Metric | Reference | Zero DME | Δ |
|---|---|---:|---:|---|
| Kermany (490) | Macro-F1 | 98.4 | 98.4 | +0.0 [-0.8, +0.8] |
| OCTDL (83) | Macro-F1 | 96.9 | 96.4 | -0.5 [-1.7, +0.7] |
| Duke (30) | Macro-F1 | 98.3 | 98.1 | -0.2 [-1.3, +1.2] |
| PAIMA (65) | Macro-F1 | 90.3 | 85.7 | **-4.7 [-6.8, -3.0]** |
| | Macro-AUROC | 95.6 | 93.2 | **-2.4 [-3.9, -1.3]** |

### Kermany against the LOSO gap

| Kermany 3-class macro-F1 | Kermany DME in training | Kermany normal/AMD in training | Score |
|---|---:|---|---:|
| Reference fused CV | 571 + 122 other | yes | 95.6 |
| DME cut (exp. 4) | 122 + 122 other | yes | 97.1 |
| **Zero DME (this run)** | 0 + 122 other | yes | **90.0** |
| LOSO held-out Kermany | 0 + 122 other | no | 73.2 |

Per-fold Kermany macro-F1: reference 97.3 / 98.1 / 96.9 / 89.7 / 95.9; zero DME 85.2 / 94.9 / 90.5 / 82.7 / 96.8.
Pooled confusion (row-normalised): DME recall 95.8 (reference 98.4), but normal is predicted DME 20.0% of the
time (reference 8.5%), so DME precision falls from 91.0 to 79.2. Normal recall is 79.8 (reference 88.9).

### Follow-up: per-source logit bias on both arms

Same diagnostic as the experiment 4 follow-up (uses target labels, so an upper bound only).

| Source | Variant | Reference | Zero DME | Δ macro-F1 |
|---|---|---:|---:|---|
| Kermany | none | 95.6 | 90.0 | **-5.6 [-6.7, -4.5]** |
| | bias cross-fit | 97.5 | 93.8 | **-3.6 [-4.5, -2.9]** |
| | bias oracle | 97.5 | 94.1 | **-3.4 [-4.2, -2.6]** |
| Duke | none | 95.5 | 89.7 | **-5.7 [-12.2, -0.5]** |
| | bias cross-fit | 94.8 | 89.6 | **-5.2 [-10.9, -0.4]** |
| | bias oracle | 96.5 | 91.1 | **-5.4 [-9.5, -1.8]** |
| OCTDL | none | 96.8 | 94.4 | -2.4 [-6.5, +0.4] |
| | bias cross-fit | 96.7 | 95.2 | -1.5 [-4.2, +0.3] |

Full table in `.scratchpad/1e23970_kermany-dme0/table.md`.

## Interpretation

- **With the device seen and no in-device DME, Kermany scores 90.0, not 73.2.** Removing every Kermany DME patient
  costs 5.6 points against the reference. Of the 22.4-point gap between seen (95.6) and held-out (73.2) Kermany,
  that accounts for about a quarter. By the rule set in the plan (stays near 90 or higher), the rest is
  device-wide: Kermany normal/AMD also being unseen, rather than DME appearance failing to transfer.
- **DME learned on other devices does transfer to Kermany.** DME recall is 95.8 with no Kermany DME in training.
  What breaks is the cutoff: normal is over-called as DME (20%). A per-source bias recovers only 3.8 of the
  5.6 points (cross-fit 93.8), so the Kermany DME contribution is more than a prior shift here, unlike the cut run.
- **Kermany DME matters for Duke.** Duke DME recall falls 91.1 to 74.5 (CI barely excludes 0) and AUROC falls
  2.1, and a bias does not fix it. The 122-patient cut gave -6.5 recall; zero gives -16.6. This supports the earlier
  weak lead that Duke DME benefits from Kermany's DME volume, but it is 45 eyes (roughly 11 DME).
- **Binary Normal-vs-AMD is unaffected on the three DME sources**, as in the cut run.
- **PAIMA binary drops 4.7 macro-F1 (AUROC -2.4) and is not explained.** PAIMA has no DME and the cut run did not move it
  (+0.3). A mechanism through DME is not obvious. See the caveats before reading anything into it.

## Implications for the plan

- Experiments 3 (augmentation) and 5 (domain pretraining) should target device shift in general and be judged on
  Kermany normal/AMD behaviour and PAIMA, not only on DME. Experiment 3 stays next.
- Per-source normalization during training remains a candidate for the same reason.

## Caveats

- **The LOSO comparison is still indirect.** LOSO-Kermany also lacks Kermany normal and AMD and trains on 3 sources
  instead of 4 with different folds. "The remaining gap is device-wide" is the reading by elimination, not a
  tested claim. A direct test would train on all Kermany DME-free but also drop Kermany normal/AMD, which is LOSO.
- **One training run per arm.** Intervals cover test-patient sampling only. Per-fold Kermany macro-F1 ranges 82.7 to
  96.8 for this arm, with fold 4 low in both arms (89.7 reference). The PAIMA drop could be partly training variance,
  since it has no DME link. This run has not been repeated with another seed.
- **Training composition changed more than DME.** Kermany now contributes only normal and AMD (about 6.1k images each),
  so Kermany's share of the fused batches and the class balance both shift.
- **Duke is 45 eyes**, with wide intervals. Duke DME recall rests on a few eyes.
