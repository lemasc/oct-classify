# Exp. 2: Seed Noise Floor

Experiment 2 of the [cross-device plan](../cross-device-plan.md). It measures how much the held-out score moves when
only the training seed changes, so that gains from exp. 3 and later can be judged against it. Fill-aware and square-crop ResNet-50
LOSO are each retrained with two new seeds, with Kermany and PAIMA held out, 5 models per seed and target.

> **Both arms are in.** The fill-aware arm is sections 1 to 4; the square-crop arm (the base config for 3b and 5)
> is section 5. The two arms do not share a noise floor: square crop is as steady as fill-aware on Kermany but far
> noisier on PAIMA. Thresholds from the fill-aware arm alone must not be applied to square-crop deltas.

## Summary

- **Macro-F1 and AUROC move little between seeds.** Across the three seeds (reference plus two new), the seed means
  of held-out macro-F1 range over 2.6 points on Kermany (73.8, 74.5, 76.4) and 1.3 on PAIMA (86.4, 87.7, 87.0).
  AUROC ranges over 1.2 (Kermany) and 0.4 (PAIMA). All are below the plan's provisional 5-point threshold.
- **Fold-to-fold spread is larger than seed-to-seed spread.** The 5 models of one campaign differ by Duke CV fold, and
  their SD is 4.6 macro-F1 points on Kermany and 1.3 on PAIMA (average over seeds), against a seed-mean SD of 1.3
  and 0.7. A comparison between two arms should therefore be paired by fold, and a single campaign mean is not a
  precise number on Kermany.
- **Pooled per-class recall moves much more than the aggregate metrics.** On Kermany, AMD recall spans 9.3 points
  (82.9 to 92.2), DME recall 7.5 (67.9 to 75.4) and normal → DME 5.0 (13.3 to 18.3). On PAIMA, AMD recall spans 4.5
  and normal recall 3.2. A recall-only gain under about 8 points on Kermany is not separable from seed noise.
- **Square-crop arm (section 5):** Kermany is as steady as fill-aware (seed-mean range 1.8 macro-F1, 0.6 AUROC) and
  its gain over fill-aware is +8 to +13 at every seed. PAIMA is not: seed means of 85.2, 80.8 and 75.6 (AUROC 91.0, 83.6,
  82.7), a 9.7-point range. Working delta for PAIMA square-crop comparisons is about 10 points.
- **The 3a PAIMA deficit looked like noise but is not, for square crop.** Exp. 3a found both geometry arms about
  1.2 points below the PAIMA reference macro-F1 (85.0 and 85.2 vs 86.4), inside the fill-aware seed span of 1.3.
  The new square-crop seeds score 80.8 and 75.6, so the 3a figure was the best of three and square crop does cost
  PAIMA performance (section 5). The 3a Kermany gains (+6.0 mirror-pad, +12.9 square-crop macro-F1 on the full source) are
  2 to 5 times the seed range and stand.
- **Recommended working delta (fill-aware arm, 3 seeds; square-crop deltas are in section 5):** about 3 points for Kermany macro-F1, 1.5 for AUROC,
  1.5 for PAIMA macro-F1, and about 10 for a single class's recall. The plan's flat 5 points is conservative for
  the aggregate metrics and too tight for per-class recall. With three seeds, these are rough ranges, not intervals.

## Setup

- **Configs:** `configs/training/resnet50-loso-fill-aware-seed{20260914,20260915}.toml`. Identical to the reference
  fill-aware config except `seed`. The reference and exp. 3a used seed 20260913.
- **Training:** ResNet-50, 224 px, source-class-balanced sampling, up to 100 epochs, early stopping with patience 10.
  Held-out source trained on the other three; one model per Duke CV fold (5 per campaign), so the fold is a second
  source of variation besides the seed.
- **Evaluation:** `checkpoint-best.pt` scored on the held-out source's full manifest (`--full-source`).
  Held-out Kermany is 101,217 images and PAIMA 16,601. Metrics are computed per model, then averaged over the 5 models.
- **Reference:** `loso-*-local-20261008-174938-fold{1..5}` in `artifacts/runs/fused/resnet50`.
- **Run:** `scripts/exp2-seeds-local.sh --workers 6` on the RTX 3060, driven through
  `.scratchpad/049579f_exp2-seeds/run.sh`. Runs and logs: `.scratchpad/049579f_exp2-seeds/{runs,local}`. Campaigns
  for the fill-aware arm finished 2026-10-12 (kermany s20260914 01:18, paima s20260914 06:59, kermany s20260915
  09:36, paima s20260915 about 16:50).
- **Statistics:** none beyond means, SDs and ranges. No patient bootstrap here, so none of these numbers are
  confidence intervals. Three seeds give a rough range only.
- **Checks passed:** all 20 folds have a held-out evaluation; each is full-source with the right target; every
  confusion-matrix total equals the prediction-file row count; each checkpoint path names the right fold and seed.
  Three kermany s20260915 evaluations (folds 3 to 5) were rerun with `--full-source` by `run.sh` after a gap in the
  original run, and kermany s20260914 and paima s20260914 are listed in the output root as `artifacts/...` in the
  evaluation records because those runs were moved to the scratchpad afterwards. Names are correct.

## 1. Held-out Kermany

Mean over the 5 fold models (SD across folds in brackets).

| Seed | Macro-F1 | Balanced acc | AUROC |
|---|---|---|---|
| 20260913 (reference) | 73.8 (5.2) | 78.4 (1.5) | 93.7 (0.9) |
| 20260914 | 74.5 (7.4) | 79.1 (4.9) | 92.8 (2.5) |
| 20260915 | 76.4 (1.1) | 79.7 (2.5) | 94.0 (1.0) |
| **Seed-mean range** | **2.6** | **1.4** | **1.2** |
| Fold SD (avg over seeds) | 4.6 | 3.0 | 1.5 |

Per-fold macro-F1 by seed (folds 1 to 5):

| Seed | F1 | F2 | F3 | F4 | F5 |
|---|---:|---:|---:|---:|---:|
| 20260913 | 78.7 | 70.6 | 66.4 | 77.3 | 75.9 |
| 20260914 | 72.8 | 76.6 | 64.1 | 84.7 | 74.5 |
| 20260915 | 76.9 | 74.8 | 76.4 | 76.1 | 77.6 |

Fold 3 is the weakest under two of three seeds, but fold 4 swings 77.3, 84.7 and 76.1. One campaign's mean is
dominated by a few folds, and seed 20260914 is the noisiest campaign (SD 7.4).

## 2. Held-out PAIMA

| Seed | Macro-F1 | Balanced acc | AUROC |
|---|---|---|---|
| 20260913 (reference) | 86.4 (2.0) | 86.3 (1.9) | 92.8 (1.0) |
| 20260914 | 87.7 (0.8) | 87.6 (0.8) | 93.3 (0.7) |
| 20260915 | 87.0 (1.1) | 87.0 (1.1) | 92.9 (0.9) |
| **Seed-mean range** | **1.3** | **1.3** | **0.4** |
| Fold SD (avg over seeds) | 1.3 | 1.3 | 0.9 |

PAIMA is steadier than Kermany on every metric. The reference was the lowest of the three seeds on macro-F1 and
AUROC, so the 3a comparisons against it start from the unlucky end of the range.

## 3. Pooled per-class recall

Row-normalized percent, pooled over the 5 fold models of each seed (the method of
`.scratchpad/02da1c8_fill-vs-plan/recall.py`). The bold rows are the plan's watch items.

**Kermany**

| | Ref 20260913 | 20260914 | 20260915 | Range |
|---|---:|---:|---:|---:|
| AMD recall | 88.6 | 82.9 | 92.2 | 9.3 |
| **DME recall** | **71.6** | **75.4** | **67.9** | **7.5** |
| Normal recall | 74.9 | 79.2 | 79.1 | 4.3 |
| **Normal predicted as DME** | **18.3** | **16.5** | **13.3** | **5.0** |
| DME predicted as AMD | 15.0 | 8.7 | 16.4 | 7.7 |

**PAIMA**

| | Ref 20260913 | 20260914 | 20260915 | Range |
|---|---:|---:|---:|---:|
| **AMD recall** | **77.7** | **81.9** | **77.4** | **4.5** |
| **Normal recall** | **95.0** | **93.4** | **96.6** | **3.2** |

Per-model recall varies far more than the pooled value: Kermany AMD recall runs from 59.7 to 97.0 across the five
s20260914 models, and DME recall from 50.3 to 83.0 across s20260915. The plan's earlier note that PAIMA recall moved
7 points under fill-aware (median-pad vs fill-aware) is larger than any seed spread here (4.5 and 3.2), so that
change is probably more than seed noise, but this was not tested directly.

## 4. Reading against the plan

- **Plan question answered:** a gain of 2 to 3 points in aggregate macro-F1 on Kermany cannot be told from seed
  noise, and a PAIMA macro-F1 change under about 1.5 cannot either. A gain over 5 points on either clears it.
- **Exp. 3a re-read:** Kermany gains (mirror-pad +6.0, square-crop +12.9 full-source macro-F1) are well above the
  seed range. The PAIMA result (about 1 point below the reference for both arms) was within the fill-aware noise, but the
  square-crop seeds in section 5 show a real PAIMA loss for that arm (mean −6.9 and −11.5 on the new seeds, paired
  by fold). Neither arm helps PAIMA. The statement that PAIMA's AUROC falls from 92.8 to 91.0 for
  square-crop (1.8 points) is above the AUROC seed range of 0.4, but the 3a reference sits at the low end of the
  seed spread, so the drop could be somewhat smaller against the seed mean (93.0).
- **Size-cue evidence on normal → DME (exp. 1 and 3a):** the reference's 18.3% sits at the high end of the 13.3 to
  18.3 seed range. Quoted drops of a few points in this rate are not large against it. The by-size breakdowns
  (36.6% → 4.5% at 768×496) are much larger than this noise.
- **Not covered:** these numbers measure training-seed variance on the full-source held-out score. They say nothing
  about test-patient sampling, which the paired patient bootstrap in 3a covers. The 3a intervals describe the second
  and not the first, so the two should be combined only roughly, as the plan already states.

## 5. Square-crop arm

Same method as sections 1 to 3. Seed 20260913 is the exp. 3a run (`loso-*-local-3a-r2-square-crop`); the two new
seeds are `loso-{kermany,paima}-exp2-fill-aware-square-crop-s{20260914,20260915}` (all 20 folds evaluated full-source,
finished 2026-10-13: kermany s20260915 04:00, paima s20260915 09:50). Table values are the mean over the 5 fold models
(SD across folds in brackets) and were produced by `.scratchpad/049579f_exp2-seeds/compare.py`.

**Held-out Kermany**

| Seed | Macro-F1 | Balanced acc | AUROC |
|---|---|---|---|
| 20260913 (3a) | 86.6 (3.1) | 88.7 | 97.6 (0.4) |
| 20260914 | 85.9 (2.0) | 87.2 | 97.2 (0.6) |
| 20260915 | 84.8 (4.9) | 87.3 | 97.0 (1.2) |
| **Seed-mean range** | **1.8** | **1.5** | **0.6** |

**Held-out PAIMA**

| Seed | Macro-F1 | Balanced acc | AUROC |
|---|---|---|---|
| 20260913 (3a) | 85.2 (5.0) | 85.2 | 91.0 (3.9) |
| 20260914 | 80.8 (4.8) | 81.0 | 83.6 (8.4) |
| 20260915 | 75.6 (10.2) | 76.8 | 82.7 (11.2) |
| **Seed-mean range** | **9.7** | **8.4** | **8.3** |

Per-fold macro-F1 on PAIMA (folds 1 to 5): s913 88.9, 84.0, 77.0, 88.5, 87.8; s914 78.2, 86.9, 84.4, 79.0, 75.3;
s915 62.5, 80.9, 79.2, 87.6, 67.6. Some folds collapse (62.5 and 67.6 in s915), and the fold that is worst changes
with the seed, so the instability is not tied to one Duke fold.

**Pooled per-class recall**

| Kermany | s913 | s914 | s915 | Range |
|---|---:|---:|---:|---:|
| AMD recall | 95.8 | 94.5 | 93.6 | 2.2 |
| **DME recall** | **82.3** | **77.8** | **80.6** | **4.5** |
| Normal recall | 88.0 | 89.2 | 87.7 | 1.5 |
| **Normal predicted as DME** | **6.7** | **4.7** | **7.5** | **2.8** |

| PAIMA | s913 | s914 | s915 | Range |
|---|---:|---:|---:|---:|
| **AMD recall** | **83.3** | **74.8** | **79.3** | **8.5** |
| **Normal recall** | **87.2** | **87.2** | **74.4** | **12.8** |

**Square crop minus fill-aware, paired by fold and seed (macro-F1 points; mean over 5 folds, SD in brackets)**

| Seed | Kermany | PAIMA |
|---|---|---|
| 20260913 | +12.9 (8.1) | −1.2 (6.6) |
| 20260914 | +11.3 (8.6) | −6.9 (4.0) |
| 20260915 | +8.4 (5.1) | −11.5 (9.6) |

Findings:

- **Kermany: the square-crop gain is confirmed.** It is +8.4 to +12.9 macro-F1 at every seed, 5 to 7 times the
  square-crop seed range (1.8). Normal → DME falls to 4.7 to 7.5% (fill-aware 13.3 to 18.3%) and DME recall rises to
  78 to 82%. The seed range is a little smaller than fill-aware's (2.6). Fold-to-fold SD is still 5 to 9 on the
  paired difference, so the size of the gain is not precise.
- **PAIMA: square crop is not noise-equivalent to fill-aware.** The 3a result (−1.2) was the best of three seeds.
  The paired loss grows to −6.9 and −11.5 with the new seeds, and AUROC drops from 91.0 to 83.6 and 82.7. Seed range is
  9.7 macro-F1 and 8.3 AUROC, against 1.3 and 0.4 for fill-aware. The "no demonstrated loss on PAIMA" conclusion of 3a
  does not stand; with three seeds, square crop looks worse than fill-aware on PAIMA by about 6 points on average,
  though the seeds disagree so strongly that this mean is itself uncertain. Normal recall on s915 (74.4) is the
  main driver of that seed's loss.
- **Working deltas for the square-crop arm:** about 3 macro-F1 and 1 AUROC points on Kermany, and about 10
  macro-F1 and 8 AUROC points on PAIMA. A PAIMA change under about 10 points cannot be told from seed noise.
  A 3b change should be compared against the square-crop seeds, not the fill-aware ones.
- **Cause of the PAIMA instability is not known.** Likely candidates, none tested: the random square window falls on
  different parts of 1024- and 1536-wide scans, so a model may see little tissue; or the model reads the scale of the
  tissue. This is the same unexplained PAIMA gap noted in the plan for exp. 1d and 3a.

## 6. Limits

- Three seeds per arm (two new plus the 3a run). Ranges, not intervals. The 3a square-crop run was trained and
  evaluated in the earlier session, not in this batch, so it differs only by seed.
- No patient bootstrap and no test of the PAIMA paired difference. The mean difference of −6.9 comes from three seeds
  of 5 fold pairs each, and the pairs within a seed share data.
- OCTDL and Duke as held-out targets were not run.
