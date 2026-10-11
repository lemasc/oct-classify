# Exp. 3a: Removing the Size Cue in Training (Mirror Padding vs Square Crop)

Experiment 3a of the [cross-device plan](../cross-device-plan.md). Fill-aware ResNet-50 models are trained with
the image geometry changed so that an image's shape no longer carries its class, then scored on the held-out
source. Two arms, Kermany and PAIMA held out, 5 models per arm and target.

## Summary

- **Kermany: both arms raise held-out macro-F1, square-crop most.** On the full source it is 73.8 for the
  fill-aware reference, 79.8 with mirror padding and **86.6 with square crops**. These are full-source numbers, not
  the test-split numbers in the plan's gap table (reference 83 vs seen 95.6), so how much of the gap closes is not
  yet shown (the test split is in section 5). Square-crop beats mirror-pad by
  +6.9 [+6.3, +7.5] and the reference by +12.9 [+12.3, +13.5] (paired patient bootstrap, P(>0) = 1.000 for every
  Kermany comparison).
- **By image size, the square-crop gain is attributed to the size cue; mirror-pad's is not (section 5).**
  Square-crop improves macro-F1 at every Kermany size, most at 768×496 (58.7 → 81.3, normal → DME 36.6% → 4.5%,
  the exp. 1 cue). On the test split it scores 88.8 vs 82.6 for the reference (+6.3 [+4.7, +7.8]; seen 95.6), so
  about half of the ~13-point gap closes. Mirror-pad is worse than the reference on the test split (−4.4
  [−6.1, −2.7]) and at 1024×496 and 1536×496, because it sends more normal images to DME there.
- **Mirror-pad shrinks the size cue but does not remove it (section 6).** Padding Kermany's 512×496 images to 768
  now moves normal → DME by +6.7 points (reference: 12% → 60%), but heavy mirroring still drops PAIMA's AUROC by
  7.2 (768 → 1536). The planned check that mirror training makes predictions invariant to padding fails.
- **Multi-window pooling adds nothing (section 4).** Mean pooling over overlapping square windows moves the trained
  square-crop models by +0.1 on full-source Kermany and on PAIMA AUROC. The single center window was not hiding a
  loss. On the Kermany test split, square-crop scores 88.8 against 82.6 for the reference (seen 95.6), so roughly
  half of the ~13-point gap closes, not all of it.
- **PAIMA: neither arm helps.** Both arms are about 1 point below the reference (85.0 and 85.2 vs 86.4; the
  intervals exclude zero) and indistinguishable from each other (+0.2 [−0.4, +0.9]). Square-crop's fold-to-fold SD
  on PAIMA is 5.0 points, four times larger than the drop, so the small deficit may be seed noise rather than a cost.
- **Square-crop is the better candidate.** It gives the large Kermany gain at every size and no demonstrated loss
  on PAIMA. Mirror-pad is worse on the test split and leaves a residual cue, so only square-crop carries forward.
  It also fixes tissue scale, which mirror-pad does not, so the Kermany gap between the arms points at tissue scale
  as well as shape.
- **PAIMA's ranking drop is not explained by shape alone.** Training without the shape cue leaves PAIMA's held-out
  AUROC at 92.4 (mirror-pad) and 91.0 (square-crop), against 92.8 for the reference. Exp. 1d's test-time windows
  reached 95.6, and neither training for it nor pooling windows on the trained models reproduces that (section 4).

## Setup

- **Arms:** `configs/training/resnet50-loso-fill-aware-{mirror-pad,square-crop}.toml`, identical to the fill-aware
  LOSO config except `augmentation.geometry` (code in commit `dd14f36`). Both apply the geometry change to the raw
  image before preprocessing.
  - **mirror-pad:** each training image's tissue is mirrored sideways to a random wider width (uncapped), then
    letterboxed. Test images are unchanged.
  - **square-crop:** a random height × height window in training; a **single centered window** at evaluation.
- **Training:** ResNet-50, 224 px, batch 12, source-class-balanced sampling, seed 20260913, up to 100 epochs with
  early stopping (patience 10). Held-out source trained on the other three.
- **Evaluation:** each checkpoint (`checkpoint-best.pt`) scored on the held-out source's full manifest. 5 folds per
  target because Duke is a training source (one model per Duke CV fold).
- **Reference:** fill-aware LOSO `loso-*-local-20261008-174938-fold{1..5}`.
- **Statistics:** bootstrap over patients (`group_id`) with 2,000 draws, seed 20261011. Metrics are computed per
  model and averaged over the 5 models. The deltas are paired (same draws for every arm). Held-out Kermany has 5,419
  patients (101,217 images), PAIMA 437 patients (16,601 images).
- Runs: `artifacts/runs/fused/resnet50/loso-{kermany,paima}-local-3a-r2-{mirror-pad,square-crop}-fold{1..5}`
  (`r2` because the first launch left a stub run directory). Logs: `artifacts/local/loso-*-local-3a-r2-*.out`.
  Per-arm campaign tables: `data/analysis/3a-{kermany,paima}-{mirror-pad,square-crop}/`.
  Scripts: `.scratchpad/59f97d3_exp3a/{run-arms.sh,paired.py}` (output in `paired.log`).

## 1. Held-out Kermany

| Model | Macro-F1 [95% CI] | Balanced acc [95% CI] | AUROC [95% CI] |
|---|---|---|---|
| Reference (fill-aware) | 73.8 [72.5, 75.0] | 78.4 [77.4, 79.4] | 93.7 |
| Mirror-pad | 79.8 [78.5, 81.0] | 83.0 [82.0, 84.0] | 95.3 [94.8, 95.7] |
| **Square-crop** | **86.6** [85.5, 87.6] | **88.7** [87.8, 89.5] | **97.6** [97.2, 97.9] |

| Paired delta | Macro-F1 | Balanced acc |
|---|---|---|
| Square-crop − mirror-pad | +6.9 [+6.3, +7.5] | +5.7 [+5.0, +6.3] |
| Square-crop − reference | +12.9 [+12.3, +13.5] | +10.3 [+9.7, +11.0] |
| Mirror-pad − reference | +6.0 [+5.6, +6.3] | +4.6 [+4.3, +5.0] |

Fold macro-F1 (across the 5 models): mirror-pad 0.842, 0.810, 0.747, 0.805, 0.783 (SD 3.5); square-crop 0.824,
0.879, 0.906, 0.850, 0.874 (SD 3.1). Square-crop is better in 4 of 5 folds. The gap between arms (6.9) is about
twice the fold SD, so it is not a single-seed artifact.

## 2. Held-out PAIMA

| Model | Macro-F1 [95% CI] | Balanced acc [95% CI] | AUROC [95% CI] |
|---|---|---|---|
| Reference (fill-aware) | 86.4 [84.9, 87.7] | 86.3 [84.9, 87.6] | 92.8 |
| Mirror-pad | 85.0 [83.5, 86.3] | 85.0 [83.5, 86.3] | 92.4 [91.2, 93.6] |
| Square-crop | 85.2 [83.9, 86.4] | 85.2 [83.9, 86.4] | 91.0 [89.7, 92.3] |

| Paired delta | Macro-F1 | Balanced acc |
|---|---|---|
| Square-crop − mirror-pad | +0.2 [−0.4, +0.9] | +0.2 [−0.4, +0.9] |
| Square-crop − reference | −1.2 [−1.9, −0.4] | −1.1 [−1.9, −0.4] |
| Mirror-pad − reference | −1.4 [−2.0, −0.9] | −1.4 [−1.9, −0.8] |

Fold macro-F1: mirror-pad 0.849, 0.851, 0.854, 0.862, 0.833 (SD 1.1); square-crop 0.889, 0.840, 0.770, 0.885,
0.878 (SD 5.0). PAIMA is labelled per B-scan, so these are image-level numbers only, not eye diagnoses.

## 3. Reading the result

- **Kermany's gain is real and large.** The test-patient uncertainty is a fraction of the effect size, and the
  fold spread is smaller than the gap between arms. The reference's Kermany errors clustered at the sizes that
  predict class in Duke (exp. 1). For square-crop, the by-size breakdown (section 5) shows the gain landing
  where the cue was: 768×496 normal → DME falls from 36.6% to 4.5%. For mirror-pad it does not hold up: the
  full-source gain comes mostly from 768×496 and the test split gets worse.
- **Why square-crop beats mirror-pad on Kermany.** Mirror-pad hides shape but keeps every pixel at the letterboxed
  scale, and the pad test (section 6) shows it only partly hides shape. Square-crop fixes both shape and tissue scale for every source. The 6.9-point gap is evidence that scale
  matters here, but the two arms also differ in what the model sees at test time (unchanged letterbox vs a cropped
  window), so scale is not isolated.
- **PAIMA does not respond.** The plan expected PAIMA's ranking drop to shrink if shape was the cause. It did not:
  AUROC is flat or slightly lower. The drop therefore stays under device appearance (exp. 3b) or unmeasured factors;
  shape and tissue scale are not enough in training. Exp. 1d's test-time gain (95.6) was not reproduced by training.
- **PAIMA's small deficit is unresolved.** The −1 point difference is statistically clear over test patients, but
  the training-seed variation (SD up to 5.0 for square-crop) is larger, and the bootstrap holds the 5 trained models
  fixed. Treat "no gain" as established and "a loss" as unproven.

## 4. Multi-window pooling on the trained square-crop models

Same method as exp. 1d (`.scratchpad/463d89d_exp3a-multiwindow/rescore.py`, `compare.py`): each image is cut into
overlapping square windows (side = height, half-window steps), and the window probabilities are pooled per image.
The baseline is the stored single centered-window evaluation. Same 5 models per target, same patient bootstrap
(2,000 draws, paired). The center window of the loop reproduces the stored PAIMA predictions exactly.

| Set | Stored | Mean pool | Max pool |
|---|---:|---:|---:|
| Kermany, full source (macro-F1) | 86.6 | 86.7 (+0.1 [0.0, +0.1]) | 86.5 (−0.2) |
| Kermany, test split, 1,000 images | 88.8 | 89.0 (+0.1 [−0.7, +0.9]) | 87.6 (−1.3) |
| Kermany 512×512 / 512×496 | 89.8 / 77.8 | same (one window) | same |
| Kermany 768×496 | 81.3 | 81.9 (+0.6 [+0.5, +0.8]) | 81.6 (+0.4) |
| Kermany 1024×496 | 89.3 | 89.9 (+0.6 [−0.1, +1.2]) | 89.0 (−0.4) |
| Kermany 1536×496 | 82.9 | 82.5 (−0.4 [−1.5, +0.7]) | 80.4 (−2.5) |
| PAIMA AUROC | 91.0 | 91.1 (+0.1 [0.0, +0.2]) | 91.1 (+0.1) |
| PAIMA macro-F1 | 85.2 | 85.1 (−0.1 [−0.2, +0.1]) | 85.0 (−0.2) |

- **Pooling gives nothing worth carrying forward.** The only interval that excludes zero in the right direction is
  768×496 (+0.6), well under the noise floor. Max pooling is slightly worse, most at 1536×496.
- **The center window was not hiding a loss.** Even 1536-wide scans, where it sees about a third of the tissue, lose
  nothing measurable. This agrees with exp. 1d: wide-scan losses came from unfamiliar tissue scale, not from missing
  lesions.
- **The 1d PAIMA gain (92.8 → 95.6) is not reproduced.** It was a property of the reference models, where windows
  corrected a shape and scale mismatch at test time. Models trained on crops with pooled windows reach 91.1. What
  gave the reference models that gain is unexplained.
- **Test-split gap.** Square-crop scores 88.8 on Kermany's test split. The reference scores 82.6 there (exp. 1d; the
  plan's gap table rounds it to 83) and seen Kermany scores 95.6. About half of the ~13-point gap closes, leaving
  ~7. The paired interval for square-crop vs reference on the test split is in section 5 (+6.3 [+4.7, +7.8]); the
  test split has 1,000 images from 635 patients.
- **By size,** square-crop is weakest on 512×496 (77.8, 53% of the images) and 1536×496 (82.9), and strongest on
  512×512 (89.8) and 1024×496 (89.3). The comparison with the reference is in section 5.

## 5. Kermany by image size

Stored predictions only, no inference (`.scratchpad/5091ebb_exp3a-followups/breakdown.py`, output in
`breakdown.log`). Macro-F1 over the classes present in each size, averaged over the 5 models. Paired patient
bootstrap, 2,000 draws, seed 20261011. Same caveat as above: test-patient sampling only.

| Set (images) | Reference | Mirror-pad | Square-crop | Square − ref | Mirror − ref |
|---|---:|---:|---:|---|---|
| Full source (101,217) | 73.8 | 79.8 | 86.6 | +12.9 [+12.3, +13.5] | +6.0 [+5.6, +6.3] |
| Test split (1,000) | 82.6 | 78.2 | **88.8** | +6.3 [+4.7, +7.8] | −4.4 [−6.1, −2.7] |
| 512×512 (8,205) | 80.9 | 83.4 | **89.8** | +8.9 [+7.2, +10.7] | +2.4 [+1.6, +3.4] |
| 512×496 (53,788) | 71.2 | 73.2 | **77.8** | +6.6 [+5.6, +7.6] | +2.0 [+1.5, +2.5] |
| 768×496 (27,377) | 58.7 | 72.9 | **81.3** | +22.5 [+21.1, +23.6] | +14.1 [+13.2, +14.9] |
| 1024×496 (1,290) | 84.1 | 79.5 | **89.3** | +5.3 [+3.8, +6.6] | −4.5 [−6.4, −2.8] |
| 1536×496 (10,534) | 77.6 | 68.0 | **82.9** | +5.4 [+3.0, +7.6] | −9.6 [−11.4, −7.5] |

Normal images called DME (the size-cue error), %:

| Set | Reference | Mirror-pad | Square-crop |
|---|---:|---:|---:|
| Full source | 18.3 | 12.5 | 6.7 |
| Test split | 10.9 | 36.0 | 8.5 |
| 768×496 | 36.6 | 13.6 | 4.5 |
| 1024×496 | 9.1 | 29.4 | 6.9 |
| 1536×496 | 3.4 | 19.5 | 3.7 |

- **Square-crop gains at every size,** and all seven intervals exclude zero. The largest gain is at 768×496, where
  the exp. 1 size cue acted. 1024×496 and 1536×496 gain 5 points each.
- **The test-split gap closes about half.** Reference 82.6, square-crop 88.8, seen 95.6: 6.2 of 13.0 points.
- **Mirror-pad's gain is concentrated at 768×496 and 512 px images.** It is worse than the reference at 1024×496
  and 1536×496 (and so on the test split, which is mostly 1024×496), where normal → DME rises from 9% to 29% and
  from 3% to 20%. A plausible reading is that mirrored wide images resemble DME-labeled wide images in training,
  but that is not tested here.
- **Still open:** the DME recall rows (dme → normal and dme → amd) move by size in both directions and were not
  analysed beyond the paired intervals in `breakdown.log`.

## 6. Pad test on the mirror-pad models

The 1b/1c pad test, rerun on the mirror-pad models (`.scratchpad/5091ebb_exp3a-followups/{rescore,compare}.py`).
Only the image width changes, by mirror padding; the stored predictions are the baseline. A 1,000-image check
reproduces the stored predictions exactly for both sources. Reference numbers are those of exp. 1b and 1c.

| Test | Reference | Mirror-pad models |
|---|---|---|
| Kermany 512×496 → 768, normal → DME | 12% → 60% | 10.7% → 17.4% (+6.7 [+6.4, +7.0]) |
| Kermany 512×496 → 1024, normal → AMD | 7% → 32% | 3.6% → 4.6% (+1.1 [+0.8, +1.3]) |
| Kermany 512×496 → 1024, normal → DME | not reported | 10.7% → 14.2% (+3.5 [+3.1, +3.9]) |
| PAIMA 768 → 1024, AUROC | 92.8 → 88.5 | 92.4 → 90.3 (−2.1 [−2.6, −1.7]) |
| PAIMA 768 → 1536, AUROC | not reported | 92.4 → 85.2 (−7.2 [−8.2, −6.1]) |
| PAIMA 768 → 1536, AMD → normal | not reported | 19.9% → 34.4% (+14.5 [+12.8, +16.1]) |

- **The cue is mostly, not fully, gone.** Kermany's pad effects fall by roughly 85-95% against the reference models.
- **The plan's invariance check fails.** The plan said training with mirroring in every class should make heavy
  mirroring harmless. PAIMA's AUROC still drops 7 points at 1536 wide, and the models miss many more AMD images.
  Uncapped mirror padding did not make the models invariant to mirrored width. The reference rows come from the
  1b/1c reports, and the 1c reference had the PAIMA 768 → 1536 effect as normal → AMD (5% → 33%), a different
  direction from the AMD recall loss here.
- **Not run:** a pad test on the square-crop models, since a centered square crop undoes any sideways padding.

## 7. Caveats

- **Intervals cover test-patient sampling only,** not training variance. The 5 models per arm share a
  configuration and seed, differ only by Duke CV fold, and do not amount to independent seeds.
- **Square-crop is evaluated on one centered window** in sections 1-2. Section 4 tests multi-window pooling and
  finds no meaningful change.
- **Not checked:** whether either arm changes Duke's own cross-validated performance; Grad-CAM on padding (a known shortcut for Duke-only models);
  label noise from crops that miss the lesion.

## 8. Next steps

1. Add training seeds for PAIMA and Kermany square-crop (exp. 2). All intervals here omit training variance.
2. Carry square-crop (not mirror-pad) into exp. 3b and exp. 5, and add OCTDL and Duke as held-out targets.
3. Mirror-pad is dropped as a candidate, and multi-window pooling is dropped for the trained arm.
4. The by-size breakdown and the pad test are done (sections 5 and 6).
