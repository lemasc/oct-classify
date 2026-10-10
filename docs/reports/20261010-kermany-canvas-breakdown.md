# Held-Out Kermany Errors by Image Size

Experiment 1 of the [cross-device plan](../cross-device-plan.md). No training: existing predictions are split by
image size (canvas, width × height) and rescored.

## Summary

- **Errors follow image size, and the pattern matches Duke's size-to-label link.** In Duke, every 768×496 image is
  DME and every 1024×496 image is AMD. Held-out Kermany models (trained on Duke, OCTDL and PAIMA) call 37% of
  Kermany's 768×496 normal images DME, and push 23% of its 1024×496 normal images to AMD. At other sizes these
  confusions are 3-12% and 7-9%.
- **The ~22-point Kermany gap compared different image sets.** The seen score (95.6) is on Kermany's supplied test
  split, which is 94% 1024×496. On that same test split, held-out models score 82.6 (fill-aware) and 83.7 (median).
  The like-for-like gap is about **12-13 points**. The rest comes from the held-out score also covering the train
  split, which has a different size mix.
- **White fill does not explain the size effect.** Within one size, images with and without fill behave alike
  (except a small 768×496 group with fill, which is worse in both paddings).
- Image size is therefore a strong lead, but not yet proven to be causal: sizes also stand for different scan
  protocols. Experiment 1b tests it directly.

## Setup

- **Held out:** LOSO-Kermany, fill-aware (`loso-kermany-local-20261008-174938-fold*`) and median
  (`loso-kermany-local-20261001-132356-fold*`), 5 models each, scored on all 101,217 Kermany images.
- **Seen:** fused Duke-CV, median padding (`local-20260927-fused-median-pad-fold*`), 5 models, scored on Kermany's
  supplied test split (1,000 images). No fused fill-aware run exists yet.
- Image size and fill come from `artifacts/eda/features.csv`. "Fill" means more than 0.1% white rotation fill.
- Patient-level bootstrap (2,000 draws, seed 20261010). Each metric is computed per model, then averaged over the 5
  models. CIs cover patient sampling only.
- Scripts and output: `.scratchpad/c038c5a_exp1-width/{width.py,pred_mix.py,width.json}`.

## 1. Image Sizes in Each Source

Number of images per size and label (sizes with at least 2% of a source):

| Size | Duke | Kermany train | Kermany test | PAIMA |
|---|---|---|---|---|
| 512×512 | — | 8,205 (all DME) | — | — |
| 512×496 | 1,407 N, 357 A, 582 D | 30,559 N, 21,773 A, 1,456 D | — | — |
| 768×496 | **519 D only** | 14,095 N, 12,470 A, 808 D | 4 | 8,521 N, 8,080 A |
| 1024×496 | **366 A only** | 170 N, 164 A, 14 D | 942 | — |
| 1536×496 | — | 5,183 N, 4,935 A, 364 D | 52 | — |

OCTDL uses its own sizes (for example 882×322), shared with no other source. Duke's 768×496 and 1024×496 images
each carry a single label. PAIMA is also 768×496 but has no DME labels, and the masked loss skips the DME output
for PAIMA images, so PAIMA never teaches the model that a 768×496 image can be non-DME on the DME output.

The images are letterboxed to a square before resizing, so a wider image gets thicker padding bands and a smaller
retina. The model can read the size from the input even with median or fill-aware padding.

## 2. Held-Out Kermany: Where Each Class Goes, by Size

Fill-aware LOSO, percent of each true class predicted as each class (pooled over 5 models). Median padding gives
the same pattern.

| True class | Size | → AMD | → DME | → normal |
|---|---|---:|---:|---:|
| normal | 512×496 | 7.1 | 12.4 | **80.5** |
| | 768×496 | 5.0 | **36.6** | 58.4 |
| | 1024×496 | **22.5** | 9.1 | 68.5 |
| | 1536×496 | 8.9 | 3.4 | **87.7** |
| AMD | 512×496 | **91.4** | 7.4 | 1.2 |
| | 768×496 | 80.6 | **19.1** | 0.3 |
| | 1024×496 | **98.7** | 1.1 | 0.2 |
| | 1536×496 | **95.1** | 3.0 | 1.9 |
| DME | 512×512 | 16.3 | **69.1** | 14.6 |
| | 512×496 | 10.8 | **77.8** | 11.4 |
| | 768×496 | 6.3 | **89.0** | 4.7 |
| | 1024×496 | 14.8 | **84.4** | 0.9 |
| | 1536×496 | 20.7 | **55.9** | 23.4 |

How this lines up with Duke:

- **768×496 (Duke: DME only):** this is the only size where AMD and normal images are called DME often (19% and 37%).
  DME recall is also highest here (89%).
- **1024×496 (Duke: AMD only):** normal images are pushed to AMD most often (23%) and AMD recall is highest (98.7%).
- **512×496 (Duke: mostly normal):** normal recall is good (80%).
- **1536×496 (no training source has this size):** DME is missed most (recall 56%, 23% called normal), and DME is
  almost never predicted.

## 3. Seen vs Held Out on the Same Images

Macro-F1 with patient-level 95% CI. Recall in percent.

| Model | Images scored | Macro-F1 | Normal recall | AMD recall | DME recall |
|---|---|---|---:|---:|---:|
| Seen (fused CV, median) | test split (1,000) | 95.6 [94.6, 96.4] | 88.9 | 99.0 | 98.4 |
| Held out, median | test split (1,000) | 83.7 [82.1, 85.3] | 61.2 | 98.8 | 86.2 |
| Held out, fill-aware | test split (1,000) | 82.6 [80.9, 84.2] | 63.0 | 98.4 | 83.4 |
| Held out, median | all images (101,217) | 73.2 [71.9, 74.4] | 77.1 | 86.0 | 65.5 |
| Held out, fill-aware | all images (101,217) | 73.8 [72.5, 74.9] | 74.9 | 88.6 | 71.6 |

The test split is almost all 1024×496, where held-out normal images are pushed to AMD. That is why normal
recall, not DME recall, is the main loss on the test split.

The seen models are only scored on 1024×496 and 1536×496. How a seen model does on Kermany's 512×496, 512×512 and
768×496 images is unknown, so the gap at those sizes cannot be measured with existing runs.

## 4. Fill Within One Size

| Size | Fill | Patients | Macro-F1 (fill-aware) | Macro-F1 (median) |
|---|---|---:|---|---|
| 512×496 | no | 471 | 67.0 [61.5, 73.4] | 66.1 [60.9, 71.5] |
| | yes | 4,364 | 71.3 [69.3, 73.1] | 73.0 [70.8, 75.0] |
| 768×496 | no | 3,918 | 60.8 [59.3, 62.5] | 58.8 [57.0, 60.6] |
| | yes | 540 | 41.4 [39.4, 43.4] | 42.8 [40.6, 45.0] |
| 1536×496 | no | 2,220 | 77.4 [74.3, 80.2] | 81.6 [78.2, 84.6] |
| | yes | 296 | 77.5 [72.5, 81.5] | 74.6 [68.8, 80.9] |

Differences between sizes (59 to 84) are much larger than differences between fill and no fill within a size. The
exception is 768×496 with fill, where 55% of non-DME images are called DME. Fill-aware padding does not fix it.
The "images without fill are harder" result in the fill-aware report was mostly a size effect: most images without
fill are 768×496 or 1536×496.

## 5. What This Changes

- **Cause tracker:** "image size predicts the class" moves from `open` to `partial`. Errors follow Duke's
  size-to-label link closely, but size also stands for scan protocol, so the link is not yet causal.
- **Gap table:** use the test-split comparison (about 12-13 points) for Kermany. The full-source held-out number
  mixes in a different set of image sizes.
- **Next, experiment 1b (no training):** rescore held-out Kermany after making image sizes uniform at test time, for
  example center-cropping 768, 1024 and 1536-wide images to 512×496. If 768×496 normal images stop being called DME
  and 1024×496 normal images stop going to AMD, size is a cause. Cropping keeps the tissue scale, so it separates
  the size cue from scan content.
- **Experiment 3a** should remove the size cue during training: crop or resize to a common aspect ratio, so that
  Duke's 768×496 and 1024×496 images no longer carry their label in their shape.

## Caveats

- Size is confounded with scan protocol (scan width, resolution, possibly site and time). This analysis shows
  where errors cluster. It does not show that the model reads the size.
- Patients appear at several sizes (4,183 of 5,419 Kermany patients), so per-size CIs are not independent of each
  other.
- Class mix differs by size, so macro-F1 is not comparable across sizes. Use the per-class rows in §2.
- Seen-model numbers use median padding; there is no fused fill-aware run yet.
- One training seed per campaign. CIs cover patient sampling only.
