# Cross-Device Experiment Plan

**Started:** 2026-10-10 at `02da1c8`. Follows [20261010-resnet50-loso-fill-aware.md](reports/20261010-resnet50-loso-fill-aware.md).
The overall project plan is [full-project-plan.md](full-project-plan.md).

## Question

A model scores well on a device (source) it was trained on, and worse on a device it has never seen. Why, and
what closes the gap?

The gap, in macro-F1 (seen = fused training with that source included; held out = LOSO). Both columns are scored on
the source's test split, so they compare the same images:

| Source | Seen | Held out | Gap |
|---|---:|---:|---:|
| Kermany | 95.6 | 83 | ~13 |
| OCTDL | ~97 | 92 | ~5 |
| PAIMA | ~90 | 86 | ~4 |

Kermany is the main problem. Earlier reports quote ~22 points for Kermany: that held-out number (74) is scored on
all Kermany images, which have a different mix of image sizes than the test split (exp. 1). OCTDL and PAIMA
held-out numbers are still full-source; their test-split values have not been computed yet.

## How to Use This Plan

- The **cause tracker** lists every suspected cause and whether it is ruled out, partly supported or still open.
  Update it whenever an experiment finishes.
- The **experiment table** is in run order. Each experiment says what result would change the plan.
- The **decision log** at the bottom records what was decided and why, newest last. Add an entry after every result.

## Cause Tracker

Status values: `ruled out`, `partial` (explains part of the gap), `open`, `known limit` (real but not fixable
with the current data).

| Suspect | Status | Evidence | Next test |
|---|---|---|---|
| Checkpoint selection favours the larger sources | ruled out | Fixed in `f6d1315` (size-weighted selection), before all LOSO runs | — |
| Decision cutoff or class-prior shift | ruled out | The best per-source bias lifts held-out Kermany only 73.2 to 76.4 (*) | — |
| Brightness/contrast statistics (BatchNorm) | ruled out | Re-estimating BatchNorm on the target gives Kermany +1.1 and costs OCTDL ~20 (*). Fill-aware normalization also leaves the gap unchanged | — |
| Less training data for the held-out device | ruled out | Seen Kermany with DME cut to 122 patients still scores 97.1 (*) | — |
| White rotation fill, padding, skewed normalization | ruled out | Fill-aware LOSO moves macro-F1 by at most 0.6 vs median on Kermany, OCTDL and PAIMA. Kermany gains just as much on images that have no fill | — |
| DME seen only on other devices | partial | Removing Kermany's DME from training (device otherwise seen) costs ~6 points (*), about half of the ~13-point test-split gap | exp. 3, 5 |
| Image size predicts the class | partial, confirmed cause | In Duke, every 768×496 image is DME and every 1024×496 image is AMD. Held-out Kermany errors cluster at those sizes (exp. 1). Padding the same 512×496 images to 768 wide sends 60% of normal images to DME; padding them to 1024 wide sends 32% to AMD instead (exp. 1b). Share of the gap not yet measured | exp. 3a, 1c |
| Device appearance (speckle, contrast, resolution, texture) | open | Whatever remains after missing DME and image size | exp. 3b, 5 |
| PAIMA ranking drop | open | AUROC 95.6 seen vs 92.7 held out; a cutoff shift doesn't fix it (*) | exp. 3, 5 |
| Label units differ (Duke per eye, PAIMA per B-scan) | known limit | Caps how well Duke and PAIMA can agree. Cannot be fixed without relabeling | — |

(*) From experiments on the `docs/cross-device-plan` branch: binary rescoring, AdaBN, logit bias, Kermany DME cut
and Kermany zero-DME. Their reports live on that branch only.

## Measurement Rules

- **Reference:** fill-aware ResNet-50 LOSO, `loso-*-local-20261008-174938*`. New experiments train with
  `data.padding = "fill_aware"` and are compared with it. (The earlier plan used the median-padding runs.)
- **Primary sets:** held-out Kermany and PAIMA. OCTDL is checked after them. Duke (45 eyes, one model) is supporting
  evidence only.
- **Metrics:** macro-F1 and macro-AUROC with patient-level bootstrap CIs, and paired deltas against the reference.
  Score PAIMA on Normal/AMD only. Show the binary Normal-vs-AMD view next to the 3-class result.
- **Per-class recall** for every held-out source. The DME rows are the main signal. For Kermany, also split it by
  image size (script from exp. 1), since errors differ sharply between sizes.
- **Seen vs held out on the same images:** compare against the seen score on the test split. The full-source
  held-out score is still reported, but it covers image sizes the seen models were never scored on.
- **When a change counts:** the paired CI excludes 0 **and** the delta is larger than the seed spread from exp. 2.
  Before exp. 2 is done, treat any delta under ~5 points as unconfirmed.
- Same splits, seed scheme, training budget and evaluation precision (fp32) as the reference.

Reference per-class recall (pooled over models; from `.scratchpad/02da1c8_fill-vs-plan/recall.py`):

| Held out | What to watch | Reference |
|---|---|---:|
| Kermany | DME recall | 71.6 |
| Kermany | normal predicted as DME | 18.3 |
| OCTDL | DME predicted as AMD | 19.7 |
| Duke | DME predicted as normal | 22.3 |
| PAIMA | AMD recall / normal recall | 77.7 / 95.0 |

## Experiments

Status values: `todo`, `running`, `done`, `dropped` (record the reason in the decision log).

| # | Experiment | Question | Cost | Status | Report |
|---|---|---|---|---|---|
| 1 | Image-size breakdown of existing predictions | Do errors depend on image size? | No training | done | [20261010-kermany-canvas-breakdown.md](reports/20261010-kermany-canvas-breakdown.md) |
| 1b | Rescore held-out Kermany after cropping or padding images to other sizes | Does the model read image size? | No training | done | [20261010-kermany-size-test.md](reports/20261010-kermany-size-test.md) |
| 1c | Same test on held-out PAIMA and Duke | Does the size cue affect the other held-out sources? | No training | todo | |
| 2 | Seed noise floor: 2 extra seeds of the reference, held-out Kermany and PAIMA | How big must a change be to count? | ~10 LOSO models | todo | |
| 3a | Geometry augmentation: random pad, crop, rescale; uniform image size | Does hiding image size and resolution close the gap? | LOSO, Kermany + PAIMA first | todo | |
| 3b | Appearance augmentation: speckle, gamma/contrast, blur | Does varying how images look close the gap? | LOSO, Kermany + PAIMA first | todo | |
| 4 | Fused CV with fill-aware padding | Is the seen-device score inflated by fill? | Fused CV (config exists) | todo | |
| 5a | OCT foundation model, fine-tuned under LOSO | Do OCT-specific features transfer across devices better than ImageNet ones? | LOSO | todo | |
| 5b | MAE pretraining that includes the held-out device's unlabeled images | Does adapting to the target device without labels help? | Pretraining + LOSO | todo | |
| 6 | Grad-CAM on fill-aware LOSO models | Do models look at image borders or size cues? | No training | dropped | |
| 7 | Binary runs comparable to the papers | How do our numbers compare to OCT-SelfNet? | Several runs | optional | |

### 1. Image-size breakdown of existing predictions (done)

**Why:** image size was the only acquisition artifact not yet tested. Held-out Kermany images without white fill
were much harder than images with fill, and fill-aware did not change that.

**What:** split held-out Kermany predictions by image size (width × height), for both paddings, and compare with
seen Kermany from the fused CV runs.

**Result:** errors follow Duke's size-to-label link. Duke's 768×496 images are all DME and its 1024×496 images are
all AMD. Held-out Kermany calls 37% of its 768×496 normal images DME and pushes 23% of its 1024×496 normal images to
AMD. Kermany's 1536×496 size appears in no training source, and that is where DME is missed most (recall 56%). Fill
within a size matters little. The seen score is measured almost only on 1024×496 test images; on those same images
the held-out score is 83, not 74. See [the report](reports/20261010-kermany-canvas-breakdown.md).

### 1b. Change image size at test time (done)

**Why:** exp. 1 shows where errors cluster, but size also stands for scan protocol. Changing only the size of the
same images separates the two, without any training.

**What:** rescore the reference LOSO-Kermany models three ways:

- **Crop:** center-crop 768, 1024 and 1536-wide images (and 512×512) to 512×496.
- **Pad to 768:** mirror-pad 512×496 images to 768×496, Duke's DME-only size.
- **Pad to 1024 (control):** the same images mirror-padded to 1024×496, Duke's AMD-only size. Same kind of padding,
  smaller tissue, opposite label in Duke.

**Result:** the model reads image shape. Padding to 768 sends normal images to DME (12% → 60%); padding the same
images to 1024 sends them to AMD instead (7% → 32%). Cropping 768×496 images to 512 removes most of their DME pull
(normal → DME 37% → 8%). Cropping also cuts away tissue and enlarges it on screen, so it is not a usable fix.
PAIMA's 768×496 images cannot counter the cue in training because the masked loss never trains the DME output on
PAIMA. See [the report](reports/20261010-kermany-size-test.md).

### 1c. Size test on held-out PAIMA and Duke

**Why:** held-out PAIMA is all 768×496 and trains on Duke (768×496 = DME). Held-out Duke trains on Kermany
(512×512 = DME). PAIMA is scored on normal and AMD only, so a DME pull would not show up as DME predictions there,
but it could still distort the normal/AMD ranking, which is PAIMA's open problem.

**What:** rerun the exp. 1b script on the held-out PAIMA models (crop 768×496 to 512×496) and the held-out Duke
model (pad 512×496 Duke images to 512×512 and to other sizes).

**Reading the result:** if PAIMA's AUROC or normal/AMD balance moves when the size changes, the size cue is part of
PAIMA's drop and 3a should be judged on PAIMA too. If nothing moves, the cue is a Kermany problem only.

### 2. Seed noise floor

**Why:** within one campaign, Kermany's 5 models already span 10-16 macro-F1 points. PAIMA's recall moved 7 points
under fill-aware although PAIMA's images were unchanged. Without knowing seed spread, a 2-3 point gain from
exp. 3 cannot be judged.

**What:** rerun the reference config with 2 new seeds, holding out Kermany and PAIMA. Report the spread of macro-F1,
AUROC and the per-class recall rows above.

**Reading the result:** sets the minimum delta in the measurement rules. Run it alongside exp. 3.

### 3. Device-mimicking augmentation, in two arms

**Why:** the leading suspect is how each device renders the image. Augmentation is the cheapest way to test it.
The two arms run separately so a gain can be attributed to one kind of difference. That answer shapes exp. 5.

- **3a geometry:** remove the size cue during training, so Duke's 768×496 and 1024×496 images no longer carry
  their label in their shape. For example, crop every image to a common aspect ratio, or randomly crop and pad so
  any label can appear at any size; add random rescale for resolution. Tests image size and resolution. Shape the
  details after 1b.
- **3b appearance:** speckle noise, gamma and contrast changes, blur. Tests device rendering.

**What:** LOSO holding out Kermany and PAIMA first, then OCTDL and Duke if either arm helps.

**Reading the result:** judge on Kermany DME and normal recall and on PAIMA AUROC (PAIMA's problem is ranking,
so a cutoff shift doesn't count). An arm that clears the noise floor moves its suspect to `partial`. If neither
helps, exp. 5 becomes the main bet.

### 4. Fused CV with fill-aware padding

**Why:** the seen-device scores (~96 on Kermany) come from median-padding runs. When every source is in training,
white fill tells the model which device an image came from, and that can inflate the seen score. If so, the gap
is smaller than measured. It also gives a fill-aware seen reference for the gap table.

**What:** run `configs/training/resnet50-fused-fill-aware.toml` and compare against the median fused CV.

**Reading the result:** a lower seen score shrinks the gap and updates the table at the top. It does not close the
gap, so it can run on the cluster in parallel with exps. 2 and 3.

**Limit:** Kermany's test split has almost only 1024×496 images, so neither fused run says how a seen model does on
Kermany's other sizes. Measuring that would need a patient-grouped Kermany test set drawn from the train split.

### 5. OCT-specific pretraining

**Why:** every diagnostic so far points to the features themselves, not cutoffs, statistics or data volume.
Pretraining is the direct way to change the features. It is the most expensive step, so design it while
exp. 3 runs.

- **5a:** fine-tune an existing OCT foundation model under the same LOSO protocol. Cheaper; run first.
- **5b:** MAE pretraining on unlabeled images from all sources, including the held-out one, then LOSO
  fine-tuning. Run only if 5a shows a gain or exp. 3 leaves most of the gap.

**Reading the result:** compare with the reference and with the best exp. 3 arm. A gain from 5b but not 5a means
seeing the target device matters more than OCT pretraining in general.

### 6. Grad-CAM on fill-aware LOSO models

**Why:** padding is already ruled out. Exp. 1 implicates image size, but 1b tests that more directly. Run this only
if 1b is unclear, to check whether models attend to the padding bands that reveal an image's size.

### 7. Binary runs comparable to the papers

Optional. Useful only for the write-up, not for finding the cause.

## Decision Log

- 2026-10-10: Fill-aware LOSO done ([report](reports/20261010-resnet50-loso-fill-aware.md)). Against median it
  moves held-out macro-F1 by -0.2 to +0.6 on Kermany, OCTDL and PAIMA, and leaves OCTDL's 19% DME-to-AMD confusion
  unchanged. On Kermany, DME recall rises 65.5 to 71.6 but more normal images are called DME (14.9 to 18.3), and
  images without fill gain as much as images with fill. PAIMA images are unchanged by the fix, yet its recall shifts
  7 points between classes at the same AUROC. White fill and skewed normalization are ruled out as the main cause.
- 2026-10-10: Fill-aware becomes the reference and the default padding for new runs. It is never worse than median and
  removes a known artifact.
- 2026-10-10: Run order set: 1 (no training) decides how to weight 3a and 3b; 2 and 4 run alongside 3; 5 is designed
  in parallel and run after 3. Grad-CAM (6) only runs if image size is implicated.
- 2026-10-10 (exp. 1): Held-out Kermany errors follow Duke's size-to-label link: 37% of 768×496 normal images are
  called DME (Duke's 768×496 images are all DME) and 23% of 1024×496 normal images go to AMD (Duke's 1024×496 images
  are all AMD). Image size moves to `partial`. PAIMA is also 768×496, but the masked loss never trains the DME output
  on PAIMA images, so it does not counter the link. The Kermany gap is ~13 points on the test split, not ~22; the gap
  table now compares the same images. Next: add 1b (test-time crop and pad, no training) before 3a, and shape 3a to
  remove the size cue.
- 2026-10-10 (exp. 1b): The held-out Kermany models read image shape as a label cue learned from Duke. Padding 512×496
  images to 768 wide sends normal images to DME (12% → 60%); padding them to 1024 wide sends them to AMD (7% → 32%).
  The control rules out the padding seam and tissue scale as explanations. Image size becomes a confirmed cause.
  Next: 3a is the next training run, run alongside 2. Add 1c (same test on held-out PAIMA and Duke, no training).
  Grad-CAM (6) is dropped: 1b answered its question directly.
