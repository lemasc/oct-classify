# Does the Model Read Image Size? Test-Time Crop and Pad on Held-Out Kermany

Experiment 1b of the [cross-device plan](../cross-device-plan.md). No training: the reference LOSO-Kermany models
are rescored after changing only the size of each image.

## Summary

**Yes. The held-out models use image shape as a label cue, learned from Duke.** In Duke, every 768×496 image is
DME and every 1024×496 image is AMD ([exp. 1](20261010-kermany-canvas-breakdown.md)). Changing a Kermany image's
size moves its prediction toward the label Duke attaches to that size:

| Change to the image | Normal images called DME | Normal images called AMD |
|---|---|---|
| 512×496, padded to 768 wide (Duke: DME only) | 12.4% → **59.8%** | 7.1% → 5.7% |
| 512×496, padded to 1024 wide (Duke: AMD only) | 12.4% → 3.4% | 7.1% → **32.0%** |
| 768×496, cropped to 512 wide | 36.6% → **8.3%** | 5.0% → 5.8% |
| 1024×496, cropped to 512 wide | 9.1% → 19.9% | 22.5% → **10.0%** |

> **Correction (exp. 1c, [report](20261010-heldout-size-test.md)):** doubling an image's width with mirrored tissue
> pushes predictions toward AMD on its own. The pad-to-1024 arm below doubles the width, so part of its AMD shift may
> come from the mirroring rather than Duke's 1024 = AMD link, and it is a weaker control than the next paragraph says.
> The conclusion still stands on the arms without heavy mirroring: cropping 768×496 images removes their DME pull,
> cropping 1024×496 images halves their AMD pull, and padding by 1.5× (to 768) pushes toward DME, not AMD.

The two padding arms are the cleanest test. They use the same images, the same kind of mirrored padding and the
same tissue, and differ only in the final width. One sends normal images to DME, the other sends them to AMD,
matching Duke's two single-label sizes. Neither the padding seam nor the smaller tissue can explain that: the
1024-wide arm has more of both, yet it moves predictions away from DME.

Image size moves from a lead to a confirmed cause. How much of the held-out gap it accounts for is not measured
here: cropping also removes tissue, so it cannot stand in for a model trained without the cue. Experiment 3a
measures that.

## Setup

- **Models:** fill-aware LOSO-Kermany reference, `loso-kermany-local-20261008-174938-fold{1..5}/checkpoint-best.pt`,
  with each model's own preprocessing and normalization, in fp32.
- **Arms**, applied to the raw image before the usual preprocessing (fill-aware letterbox to 224 px):
  - **crop:** center-crop images wider than 512 px to 512 wide, and 512×512 images to 512×496. Applied to all
    Kermany 768×496, 1024×496, 1536×496 and 512×512 images (47,406 images, 5,181 patients).
  - **pad:** reflect-pad 512×496 images to 768×496, half on each side (53,788 images, 4,365 patients).
  - **pad1024 (control):** the same images reflect-padded to 1024×496.
- **Baseline:** the stored held-out predictions for the same images.
- **Check:** rescoring 1,000 random images without changes reproduces the stored predictions (99.8-100% of
  predicted labels agree, probabilities within 0.018).
- **Statistics:** percent of each true class predicted as each class, per model, averaged over the 5 models.
  Paired patient-level bootstrap of the change (2,000 draws, seed 20261010); the same draw is applied to baseline
  and arm.
- Scripts and output: `.scratchpad/954003c_exp1b-size/{rescore.py,compare.py,compare.json}`.

## 1. Padding 512×496 Images

Percent of each true class predicted as each class. Change with paired 95% CI.

| True → predicted | Unchanged | Padded to 768 | Change | Padded to 1024 | Change |
|---|---:|---:|---|---:|---|
| normal → normal | 80.5 | 34.6 | −45.9 [−46.6, −45.3] | 64.6 | −15.9 [−16.7, −15.2] |
| normal → AMD | 7.1 | 5.7 | −1.4 [−1.8, −1.1] | 32.0 | **+24.9** [+24.2, +25.7] |
| normal → DME | 12.4 | 59.8 | **+47.3** [+46.6, +48.0] | 3.4 | −9.1 [−9.4, −8.7] |
| AMD → AMD | 91.4 | 74.2 | −17.2 [−18.0, −16.5] | 98.2 | +6.8 [+5.9, +7.7] |
| AMD → DME | 7.4 | 25.8 | **+18.4** [+17.6, +19.2] | 1.4 | −5.9 [−6.8, −5.1] |
| DME → AMD | 10.8 | 3.2 | −7.6 [−10.2, −5.3] | 24.2 | **+13.4** [+10.2, +17.2] |
| DME → DME | 77.8 | 95.4 | +17.6 [+13.9, +21.4] | 70.8 | −6.9 [−11.1, −3.1] |

Padding to 768 pushes every class toward DME. Padding to 1024 pushes every class toward AMD.

## 2. Cropping Wider Images to 512

| Size | True → predicted | Unchanged | Cropped | Change |
|---|---|---:|---:|---|
| 768×496 | normal → DME | 36.6 | 8.3 | **−28.4** [−29.1, −27.6] |
| | normal → normal | 58.4 | 85.9 | +27.5 [+26.8, +28.2] |
| | AMD → DME | 19.1 | 7.9 | −11.3 [−12.1, −10.6] |
| | DME → DME | 89.0 | 71.2 | −17.8 [−21.8, −14.2] |
| 1024×496 | normal → AMD | 22.5 | 10.0 | **−12.5** [−14.8, −10.2] |
| | normal → DME | 9.1 | 19.9 | +10.9 [+9.0, +12.8] |
| | DME → DME | 84.4 | 69.4 | −14.9 [−18.5, −11.2] |
| 1536×496 | normal → DME | 3.4 | 16.0 | +12.5 [+11.7, +13.4] |
| | AMD → AMD | 95.1 | 84.5 | −10.7 [−12.2, −9.2] |
| | DME → DME | 55.9 | 64.8 | +8.9 [+1.9, +16.6] |
| 512×512 | DME → DME | 69.1 | 70.4 | +1.4 [+1.1, +1.7] |

- **768×496:** cropping removes the DME pull. After cropping, these images behave like native 512×496 images
  (normal → DME 12.4, DME recall 77.8). DME recall falls because the size cue was doing part of the work.
- **1024×496:** cropping halves the push of normal images to AMD, as predicted, but adds new DME calls and loses DME
  recall.
- **1536×496:** no source has this size, so there is no cue to remove. The changes here show the cost of cropping
  itself: two thirds of the scan width is cut away and the tissue is drawn about three times larger.
- **512×512:** removing 16 rows barely matters; 512×512 is square in both cases.

Cropping is therefore not a usable fix on its own. It removes the cue but also changes content and tissue scale.

## 3. Why the Shape Survives Preprocessing

Images are letterboxed to a square: scaled so the width fits 224 px, then padded above and below. A 512×496 image
fills the square; a 768×496 image leaves bands of padding and a flatter retina; a 1024×496 image leaves wider bands.
Median and fill-aware padding make the bands the same color as the tissue background, but the band height and the
retina's on-screen size still reveal the original shape.

Tissue scale alone does not explain the result. On-screen tissue size shrinks steadily from 512 to 768 to 1024
wide, but predictions do not move steadily: they jump to DME at 768 and to AMD at 1024, the two sizes where Duke
has a single label.

## 4. Why Training Did Not Unlearn It

The LOSO-Kermany models train on Duke, OCTDL and PAIMA. PAIMA is also 768×496, with normal and AMD images, which
could teach the model that 768×496 is not always DME. But PAIMA has no DME labels, so the masked loss never trains
the DME output on PAIMA images. Nothing in training contradicts "768×496 means DME". OCTDL uses its own sizes and
does not help either.

## 5. What This Changes

- **Cause tracker:** image size becomes a confirmed cause (`partial`, causal) of held-out Kermany errors. Its share of
  the gap is not yet known.
- **Experiment 3a moves up to the next training run.** It should remove the shape cue during training, for example by
  randomly cropping each training image's width so that any label can appear at any shape. Success is judged on
  held-out Kermany with the full images, by per-size recall (exp. 1 script) and the test-split gap.
- **The other held-out sources may be affected too.** Held-out Duke trains on Kermany, where every 512×512 image is
  DME. Held-out PAIMA is all 768×496 and trains on Duke. PAIMA is scored on normal and AMD only, so a DME pull would
  not show up as DME predictions there. Repeat this test on held-out PAIMA and Duke (exp. 1c, no training).
- **Seen scores may also use the cue.** Fused training contains the same single-label sizes. The Kermany test split is
  almost all 1024×496, Duke's AMD-only size, and seen normal recall there is the lowest of the three classes (88.9).
  Experiment 4 should be read with this in mind.

## Caveats

- Reflect padding mirrors real tissue, so the padded strips contain retina, not blank background. The 1024 control
  has the same property, so this does not explain the opposite shifts.
- The test uses one training seed per fold and only the fill-aware reference. The median-padding models were not
  rescored.
- Bootstrap CIs cover patient sampling only, and are narrow because each patient contributes many images.
- Rescoring agrees with the stored predictions on 99.8-100% of labels, not exactly. GPU arithmetic and batch size
  differ slightly from the original evaluation.
