# Size Test on Held-Out PAIMA, Duke and OCTDL

Experiment 1c of the [cross-device plan](../cross-device-plan.md). No training: the reference fill-aware LOSO models
for PAIMA, Duke and OCTDL are rescored after changing only the size of each image, as in
[experiment 1b](20261010-kermany-size-test.md) for Kermany.

## Summary

- **Every held-out source responds to image geometry.** The size cue is not a Kermany-only problem.
- **Duke: the clearest evidence that the model reads the padding bands.** Mirror-padding Duke's 512×496 images by
  16 rows to 512×512 (Kermany's DME-only size) sends 99% of normal and 98% of AMD images to DME. After letterboxing,
  512×496 leaves 3-4 px bands and 512×512 leaves none, so band presence alone flips the prediction.
- **PAIMA: its ranking problem depends on geometry.** Cropping PAIMA's 768×496 images to 512 wide raises AUROC from
  92.8 to 95.2, close to the seen score of 95.6, and reduces both kinds of error. Padding to 1024 lowers AUROC to
  88.5. Cropping also enlarges the tissue on screen, so this does not yet separate a shape cue from tissue scale.
- **OCTDL: geometry explains part of its DME → AMD confusion.** Cropping to a narrower 1.4:1 ratio lowers DME → AMD
  from 19.7% to 12.8%; padding to 3.1:1 raises it to 25.0%. The breakdown of existing predictions shows no trend with
  OCTDL's own ratios, which vary little (most are 2.7-2.8:1).
- **Duke's 768×496 DME → normal (32%) is not a size effect.** Cropping those images to 512 wide leaves it unchanged.
  The difference from 512×496 (13%) comes from which eyes have which size.
- **Correction to experiment 1b.** Doubling an image's width with mirrored tissue pushes predictions toward AMD on
  its own: PAIMA padded from 768 to 1536 (planned as a no-change control) sends 33% of normal images to AMD. The 1b
  control (Kermany 512 → 1024) also doubled the width, so it is weaker evidence than reported. The 1b conclusion
  still stands on the arms without mirroring: cropping removed the 768 DME pull and halved the 1024 AMD pull, and
  padding by only 1.5× (to 768) pushed toward DME, not AMD.

## Setup

- **Models:** fill-aware LOSO reference, `loso-{paima,octdl}-local-20261008-174938-fold{1..5}` (5 models each) and
  `loso-duke-local-20261008-174938` (1 model), `checkpoint-best.pt`, each with its own preprocessing and
  normalization, fp32.
- **Arms** (applied to the raw image before the usual fill-aware letterbox; mirror padding is centered):

  | Held out | Arm | Images |
  |---|---|---|
  | PAIMA | mirror-pad 768×496 → 1024×496 | all 16,601 |
  | PAIMA | center-crop 768×496 → 512×496 | all 16,601 |
  | PAIMA | mirror-pad 768×496 → 1536×496 (planned control) | all 16,601 |
  | Duke | center-crop 768×496 and 1024×496 → 512 wide | 885 |
  | Duke | mirror-pad 512×496 → 512×512 (8 rows top and bottom) | 2,346 |
  | OCTDL | center-crop to 1.4:1 width-to-height | all 1,709 |
  | OCTDL | mirror-pad to 3.1:1 width-to-height | all 1,709 |

- **PAIMA scoring:** softmax over the normal and AMD outputs only, as in its evaluation. AUROC is AMD vs normal.
- **Check:** rescoring 1,000 images per source without changes reproduces the stored predictions (100% of labels,
  probabilities within 0.006).
- **Statistics:** percent of each true class predicted as each class, per model, averaged over models. Paired
  bootstrap over groups (PAIMA volumes, Duke eyes, OCTDL patients), 2,000 draws (500 for AUROC), seed 20261010.
- Scripts: `.scratchpad/12cc908_exp1c-size/{breakdown.py,rescore.py,compare.py}`. Summary:
  `data/analysis/heldout-size-test/size-test.json`. Rescored predictions: `artifacts/analysis/heldout-size-test/`.

## 1. PAIMA

| Arm | Normal → AMD | AMD → normal | AUROC |
|---|---|---|---|
| Unchanged | 5.0 | 22.3 | 92.8 |
| Crop to 512 | 2.8 (−2.3 [−2.8, −1.8]) | 18.6 (−3.6 [−4.6, −2.7]) | **95.2** (+2.4 [+2.0, +2.9]) |
| Pad to 1024 | 11.2 (+6.1 [+5.2, +7.1]) | 24.7 (+2.4 [+1.5, +3.4]) | 88.5 (−4.3 [−4.9, −3.7]) |
| Pad to 1536 | 32.9 (+27.9 [+25.9, +30.1]) | 24.9 (+2.6 [+1.1, +4.2]) | 81.0 (−11.8 [−13.2, −10.6]) |

Held-out PAIMA trains on Duke (1024×496 = AMD only), Kermany and OCTDL. Padding to 1024 pushes normal images toward
AMD, as the Duke link predicts. Cropping to 512 does the opposite and improves the ranking; seen PAIMA scores 95.6
AUROC, so the crop recovers most of the held-out drop. Crop and padding also change on-screen tissue size, so the
result shows that PAIMA's ranking depends on geometry, not which part of geometry.

The 1536 arm was planned as a control, because Kermany's 1536×496 images are balanced between normal and AMD. It moved
more than any other arm. See §4.

## 2. Duke (one model, 45 eyes)

| Arm | Size | True → predicted | Unchanged | Changed | Change |
|---|---|---|---:|---:|---|
| Pad to 512×512 | 512×496 | normal → DME | 0.9 | 99.4 | +98.4 [+97.9, +99.0] |
| | | AMD → DME | 4.2 | 98.0 | +93.8 [+88.7, +98.8] |
| | | DME → DME | 85.2 | 100.0 | +14.8 [+2.7, +28.1] |
| Crop to 512 | 768×496 | DME → normal | 32.2 | 33.3 | +1.2 [−4.2, +7.8] |
| | 1024×496 | AMD → normal | 1.1 | 9.0 | +7.9 [+0.2, +18.0] |

Held-out Duke trains on Kermany, where every 512×512 image is DME. Adding 16 rows changes the letterboxed input only
by removing two 3-4 px bands, and the model calls almost everything DME. Mirroring 8 rows adds little tissue, so this
is the most direct evidence that the model reads the bands themselves.

Duke's 768×496 DME images are called normal more often than its 512×496 DME images (32% vs 13%, from the breakdown
of existing predictions). Cropping does not change that, so it is a difference between eyes, not a size effect.

## 3. OCTDL

| Arm | DME → AMD | DME recall | Normal → AMD |
|---|---|---|---|
| Unchanged | 19.7 | 77.3 | 4.8 |
| Crop to 1.4:1 | 12.8 (−6.9 [−11.8, −2.0]) | 85.0 (+7.8 [+2.6, +12.8]) | 3.9 (−0.9 [−2.5, +0.5]) |
| Pad to 3.1:1 | 25.0 (+5.3 [+3.2, +7.4]) | — | 7.3 (+2.5 [+1.1, +4.0]) |

OCTDL shares no image size with any training source, yet a narrower image is called DME more often and a wider one
AMD more often. The breakdown of existing predictions shows no trend with OCTDL's own ratios, but those cover a narrow
range (10th-90th percentile 2.67-2.93:1). Geometry accounts for roughly a third of OCTDL's DME → AMD confusion in the
narrowing direction. As for PAIMA, crop and pad also change tissue scale.

## 4. Mirroring by Large Amounts Pushes Toward AMD

Padding PAIMA from 768 to 1536 doubles the width with two mirrored copies of the tissue and halves its on-screen
size. A third of normal images become AMD. Native Kermany 1536×496 images do not behave like this (held-out Kermany
calls 8.9% of its 1536×496 normal images AMD), so the shift comes from the heavy mirroring, not from the 1536 size.

Two consequences:

- **Experiment 1b's control is weaker than reported.** Its 1024 arm also doubled the width (512 → 1024) and pushed
  toward AMD. Part of that shift may be the mirroring rather than Duke's 1024 = AMD link. The 1b conclusion rests on
  the other arms: cropping without mirroring removed the 768 DME pull (normal → DME 37% → 8%) and halved the 1024 AMD
  pull (23% → 10%), and padding by 1.5× pushed toward DME.
- **Experiment 3a's mirror padding needs a limit.** The planned range (up to 3.2× the height) can mirror more than the
  image's own width. Trained on all classes, mirrored copies should not become a label cue, but current models treat
  heavily mirrored images as unusual. Cap the added width (for example at 50% of the image's own width) and compare
  with the square-crop arm.

## 5. What This Changes

- **Cause tracker:** image size is a confirmed cause on Kermany, Duke (512×512) and PAIMA, and a likely one on OCTDL.
  PAIMA's ranking drop moves from `open` to `partial`: geometry accounts for most of it at test time.
- **3a holds out PAIMA as well as Kermany**, and is judged on PAIMA's AUROC.
- **3a-pad caps the mirrored width**, and checks whether heavy mirroring still moves predictions after training.
- **Separating shape from tissue scale** is the open question for PAIMA and OCTDL. 3a-pad changes the shape while
  shrinking the tissue; 3a-crop keeps tissue scale fixed. Comparing the two arms on PAIMA and OCTDL answers it.

## Caveats

- Crop and pad change both the image's shape and the tissue's on-screen size; neither arm isolates one of them.
- Mirror padding adds mirrored tissue and seams. §4 shows that large amounts change predictions on their own.
- Held-out Duke is one model and 45 eyes, so its CIs are wide and its results are supporting evidence.
- Bootstrap CIs cover the sampling of held-out groups only, not training seeds.
