# Exp. 2: Seed Noise Floor (Fill-Aware Arm)

Experiment 2 of the [cross-device plan](../cross-device-plan.md). It measures how much the held-out score moves when
only the training seed changes, so that gains from exp. 3 and later can be judged against it. Fill-aware ResNet-50
LOSO is retrained with two new seeds, with Kermany and PAIMA held out, 5 models per seed and target.

> **Pending: square-crop arm.** The plan reruns the square-crop config (the arm carried into 3b and 5) with the same
> two seeds. Those four campaigns (`kermany`/`paima` × seeds 20260914/20260915) were still queued or running when
> this report was written, so **no square-crop seed spread exists yet**. Every threshold below comes from the
> fill-aware arm only and must not be applied to square-crop deltas until that arm is in. Section 5 lists what to add.

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
- **The 3a PAIMA deficit is inside seed noise.** Exp. 3a found both geometry arms about 1.2 points below the PAIMA
  reference macro-F1 (85.0 and 85.2 vs 86.4). The fill-aware seeds alone span 1.3 points, so that deficit is not
  evidence of a cost. The 3a Kermany gains (+6.0 mirror-pad, +12.9 square-crop macro-F1 on the full source) are
  2 to 5 times the seed range and stand.
- **Recommended working delta (fill-aware arm, 3 seeds):** about 3 points for Kermany macro-F1, 1.5 for AUROC,
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
  seed range. The PAIMA result (about 1 point below the reference for both arms) is within noise, consistent with the
  plan's statement that neither arm helps PAIMA. The statement that PAIMA's AUROC falls from 92.8 to 91.0 for
  square-crop (1.8 points) is above the AUROC seed range of 0.4, but the 3a reference sits at the low end of the
  seed spread, so the drop could be somewhat smaller against the seed mean (93.0).
- **Size-cue evidence on normal → DME (exp. 1 and 3a):** the reference's 18.3% sits at the high end of the 13.3 to
  18.3 seed range. Quoted drops of a few points in this rate are not large against it. The by-size breakdowns
  (36.6% → 4.5% at 768×496) are much larger than this noise.
- **Not covered:** these numbers measure training-seed variance on the full-source held-out score. They say nothing
  about test-patient sampling, which the paired patient bootstrap in 3a covers. The 3a intervals describe the second
  and not the first, so the two should be combined only roughly, as the plan already states.

## 5. Pending: square-crop arm

Not yet run to completion: `loso-{kermany,paima}-exp2-fill-aware-square-crop-s{20260914,20260915}` (4 campaigns,
about 5 to 6 hours on the RTX 3060, started after the fill-aware arm finished on 2026-10-12). When they finish:

1. Repeat sections 1 to 3 for the square-crop arm, including the per-class recall tables.
2. Compare each seed against the square-crop 3a run (seed 20260913), since the 3a square-crop PAIMA SD (5.0) was
   already large and a seed spread there could change the "no demonstrated loss on PAIMA" conclusion.
3. Replace the working deltas above with the larger of the two arms' spreads, and update the measurement rules in
   the plan.
4. Optionally compute a fold-matched comparison and a patient bootstrap on the new seeds, so the spread is an
   interval and not a range of three points.
