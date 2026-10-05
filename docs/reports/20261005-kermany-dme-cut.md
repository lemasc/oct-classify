# Fused CV with Kermany DME Cut to 122 Patients

## Scope and Provenance

This is experiment 4 in [cross-device-plan.md](../cross-device-plan.md). Held-out Kermany scores 73.2
macro-F1 in LOSO against 95.6 when Kermany is in training. Two things differ in the LOSO run: the model has
never seen the Kermany device, and it has seen much less DME. This experiment removes the second difference. It
trains the usual fused model with Kermany's device present but its DME cut to the amount LOSO-Kermany has.

- **The cut.** Kermany train-split DME is reduced to 122 patients (1,496 images, down from 571 patients and
  9,220 images). 122 is the DME patient count of OCTDL + Duke, which is what LOSO-Kermany trains on. The
  same patients are used in every fold (seed 20261004). Kermany normal and AMD, all validation and test
  splits, and the other three sources are unchanged (`.scratchpad/a5096ac_kermany-dme120/make_splits.py`,
  splits in `artifacts/splits/duke-cv-kermany-dme122`).
- **Training.** `configs/training/resnet50-fused-median-pad.toml`, 5-fold Duke CV, local RTX 3060
  (`artifacts/runs/fused/resnet50/kdme122-local-20261004-210330-fold{1..5}`, 21:03 to 01:10).
- **Reference.** The fused median-pad CV run, `local-20260927-fused-median-pad-fold{1..5}`, which is
  the "seen" arm of the LOSO comparison.

```bash
uv run python .scratchpad/a5096ac_kermany-dme120/bootstrap_dme_cut.py   # writes bootstrap_dme_cut.json
```

## Method

Saved test predictions only. A paired patient-level cluster bootstrap (2,000 replicates, 500 for AUROC, seed
20261004) with one patient resample per replicate scored on both arms, so Δ is a paired difference.

- **Kermany, OCTDL, PAIMA.** The test set is the same in every fold, so each arm's score is the mean over its 5
  checkpoints.
- **Duke.** The test eyes differ per fold. The out-of-fold predictions are pooled into one set, so each
  of the 45 eyes is scored once by the model that did not train on it.
- **Views.** 3-class macro-F1, macro-AUROC and DME recall, plus the binary Normal-vs-AMD view (DME rows
  dropped, probabilities renormalised). PAIMA has no DME and is scored binary only.

## Results

Percent, point estimate; Δ = cut minus reference with its 95% interval. **Bold** marks intervals that exclude 0.

### 3-class

| Source (units) | Metric | Reference | DME cut | Δ |
|---|---|---:|---:|---|
| Kermany (635 patients) | Macro-F1 | 95.6 | 97.1 | **+1.5 [+0.8, +2.2]** |
| | Macro-AUROC | 99.7 | 99.8 | +0.1 [-0.0, +0.2] |
| | DME recall | 98.4 | 97.0 | **-1.4 [-2.9, -0.1]** |
| OCTDL (99) | Macro-F1 | 96.8 | 96.8 | -0.0 [-1.2, +1.0] |
| | Macro-AUROC | 99.6 | 99.6 | +0.1 [-0.1, +0.2] |
| | DME recall | 94.5 | 97.3 | **+2.7 [+0.0, +6.2]** |
| Duke (45 eyes) | Macro-F1 | 95.5 | 93.3 | **-2.1 [-4.3, -0.1]** |
| | Macro-AUROC | 99.4 | 99.1 | -0.2 [-0.8, +0.1] |
| | DME recall | 91.1 | 84.6 | **-6.5 [-11.5, -1.6]** |

### Binary Normal vs. AMD

| Source (units) | Metric | Reference | DME cut | Δ |
|---|---|---:|---:|---|
| Kermany (490) | Macro-F1 | 98.4 | 98.7 | +0.3 [-0.1, +0.8] |
| | Macro-AUROC | 99.9 | 100.0 | +0.0 [-0.0, +0.1] |
| OCTDL (83) | Macro-F1 | 96.9 | 96.2 | -0.7 [-1.9, +0.3] |
| | Macro-AUROC | 99.4 | 99.5 | +0.1 [-0.1, +0.3] |
| Duke (30) | Macro-F1 | 98.3 | 98.4 | +0.0 [-0.9, +1.4] |
| | Macro-AUROC | 99.8 | 99.9 | +0.0 [-0.1, +0.3] |
| PAIMA (65) | Macro-F1 | 90.3 | 90.7 | +0.3 [-0.4, +1.1] |
| | Macro-AUROC | 95.6 | 95.8 | +0.2 [-0.2, +0.6] |

### Against the LOSO gap

| Kermany 3-class macro-F1 | DME patients in training | Kermany device in training | Score |
|---|---:|---|---:|
| Reference fused CV | 571 (Kermany) + 122 | yes | 95.6 |
| **DME cut (this run)** | 122 (Kermany) + 122 | yes | **97.1** |
| LOSO held-out Kermany | 122 | no | 73.2 |

Pooled over folds, the DME-cut model's per-class DME behaviour on Kermany is precision 96.0 and recall 97.0
(reference 91.0 and 98.4). Duke DME is predicted as normal 11.6% of the time against 7.0%.

## Interpretation

- **The held-out-Kermany drop is device shift, not DME volume.** Cutting Kermany DME by about 80% leaves
  Kermany at 97.1, so the LOSO 73.2 is not explained by having only 122 DME patients. What is missing there is
  the Kermany device itself. This is consistent with the logit-bias result that the gap is in the features.
- **The Kermany gain is probably a prior effect, not a better model.** With less DME in training, the model
  over-calls DME less on the balanced Kermany test set (DME precision 91.0 to 96.0, recall 98.4 to
  97.0). Macro-AUROC does not move (+0.1 [-0.0, +0.2]), so ranking is unchanged. I did not test this
  explanation directly.
- **Duke DME loses recall.** The cut raises the share of Duke DME predicted normal (7.0% to 11.6%) and
  macro-F1 falls 2.1 (CI barely excludes 0). Fewer and less varied DME training patients plausibly hurt a
  source whose DME looks different from Kermany's. This is a weak lead from 45 eyes, not an established effect.
- **OCTDL and PAIMA are unaffected** in macro-F1 and AUROC. OCTDL DME recall rises 2.7 with a CI touching 0.
- **The binary view shows no effect anywhere.** The cut matters only for the DME decision.

## Caveats

- **The intervals measure test-patient sampling only.** They do not include training randomness, and each arm
  is 5 checkpoints from one run. Per-fold Kermany Δ ranges from -2.4 to +8.6 (fold 4 is the outlier), so the
  fold-to-fold spread is larger than the bootstrap interval suggests. Do not read +1.5 as established. The
  safe claim is "no loss from the cut", which is the result that matters here.
- **The cut removes only DME.** Kermany is now imbalanced in training (about 6.1k normal, 6.1k AMD and 1.5k DME
  images). This tests less DME diversity, not less Kermany data overall. The plan's "50% data" variant is
  not covered.
- **The LOSO comparison is indirect.** The LOSO models use different training sources and different folds.
- **Duke is 45 eyes** (30 normal/AMD units), pooled out-of-fold, so its intervals are wide.
- **Epoch counts differ** (cut 29/20/18/41/39, reference 21/30/34/22/26). Checkpoint selection may add
  some of the fold-level variation. It was not examined.
