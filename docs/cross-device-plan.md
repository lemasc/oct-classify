# Cross-Device Experiment Plan

**Started:** 2026-10-10 at `02da1c8`. Follows [20261010-resnet50-loso-fill-aware.md](reports/20261010-resnet50-loso-fill-aware.md).
The overall project plan is [full-project-plan.md](full-project-plan.md).

## Question

A model scores well on a device (source) it was trained on, and worse on a device it has never seen. Why, and
what closes the gap?

The gap, in macro-F1 (seen = fused training with that source included; held out = LOSO):

| Source | Seen | Held out | Gap |
|---|---:|---:|---:|
| Kermany | ~96 | 74 | ~22 |
| OCTDL | ~97 | 92 | ~5 |
| PAIMA | ~90 | 86 | ~4 |

Kermany is the main problem. Most of the lost accuracy is DME being confused with other classes.

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
| DME seen only on other devices | partial | Removing Kermany's DME from training (device otherwise seen) costs ~6 of the ~22 points (*) | exp. 3, 5 |
| Image size predicts the class | open | 74% of Kermany DME is 512×512; Duke width alone gives 0.66 balanced accuracy. Held-out Kermany images without fill are much harder (DME recall 58 vs 70) | exp. 1, 3a |
| Device appearance (speckle, contrast, resolution, texture) | open, leading | What remains after everything above: ~3/4 of the Kermany gap | exp. 3b, 5 |
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
- **Per-class recall** for every held-out source. The DME rows are the main signal.
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
| 1 | Width breakdown of existing predictions | Do errors depend on image size? | No training | todo | |
| 2 | Seed noise floor: 2 extra seeds of the reference, held-out Kermany and PAIMA | How big must a change be to count? | ~10 LOSO models | todo | |
| 3a | Geometry augmentation: random pad, crop, rescale; uniform image size | Does hiding image size and resolution close the gap? | LOSO, Kermany + PAIMA first | todo | |
| 3b | Appearance augmentation: speckle, gamma/contrast, blur | Does varying how images look close the gap? | LOSO, Kermany + PAIMA first | todo | |
| 4 | Fused CV with fill-aware padding | Is the seen-device score inflated by fill? | Fused CV (config exists) | todo | |
| 5a | OCT foundation model, fine-tuned under LOSO | Do OCT-specific features transfer across devices better than ImageNet ones? | LOSO | todo | |
| 5b | MAE pretraining that includes the held-out device's unlabeled images | Does adapting to the target device without labels help? | Pretraining + LOSO | todo | |
| 6 | Grad-CAM on fill-aware LOSO models | Do models look at image borders or size cues? | No training | todo | |
| 7 | Binary runs comparable to the papers | How do our numbers compare to OCT-SelfNet? | Several runs | optional | |

### 1. Width breakdown of existing predictions

**Why:** image size is the only acquisition artifact not yet tested. Held-out Kermany images without white fill
are much harder than images with fill, and fill-aware did not change that. Fill tracks image width in Kermany, so
the real factor may be width or resolution.

**What:** split held-out Kermany recall by width (512, 768, 1024, 1536) for the reference. Do the same for seen
Kermany in the fused CV runs.

**Reading the result:** if errors cluster by width in the held-out runs, size or resolution is part of the device
difference. Run 3a first and give it the most weight. If they don't, mark image size as ruled out at test time
(it can still leak during training, which 3a covers).

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

- **3a geometry:** random padding and cropping, random rescale, uniform image size. Tests image size and
  resolution.
- **3b appearance:** speckle noise, gamma and contrast changes, blur. Tests device rendering.

**What:** LOSO holding out Kermany and PAIMA first, then OCTDL and Duke if either arm helps.

**Reading the result:** judge on Kermany DME and normal recall and on PAIMA AUROC (PAIMA's problem is ranking,
so a cutoff shift doesn't count). An arm that clears the noise floor moves its suspect to `partial`. If neither
helps, exp. 5 becomes the main bet.

### 4. Fused CV with fill-aware padding

**Why:** the seen-device scores (~96 on Kermany) come from median-padding runs. When every source is in training,
white fill tells the model which device an image came from, and that can inflate the seen score. If so, the gap
is smaller than measured.

**What:** run `configs/training/resnet50-fused-fill-aware.toml` and compare against the median fused CV.

**Reading the result:** a lower seen score shrinks the gap and updates the table at the top. It does not close the
gap, so it can run on the cluster in parallel with exps. 2 and 3.

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

**Why:** padding is already ruled out, so this has low value now. Run it only if exp. 1 or 3a implicates image size,
to check whether models attend to image borders.

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
