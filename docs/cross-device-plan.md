# Cross-Device Experiment Plan

**Started:** 2026-10-04 at `98be7ad`. Follows the ResNet-50 fused and LOSO campaigns in
[20261002-resnet50-training.md](reports/20261002-resnet50-training.md). The overall project plan is
[full-project-plan.md](full-project-plan.md).

## Question

Why does performance drop when testing on a device (source) not seen in training, and which changes close the gap?

LOSO evidence so far: macro-F1 is 96–100 when the test source was in training, but drops for unseen
sources. Held-out Kermany falls to 73 macro-F1. DME recall drops most (median-padding reference, checkpoints
pooled): in held-out Duke, 20% of DME is predicted normal; in held-out OCTDL, 19% of DME is predicted AMD;
in held-out Kermany, 19% goes to normal and 16% to AMD.

## Measurement Rules

- **Primary sets:** LOSO held-out Kermany, OCTDL, and PAIMA. Duke (45 eyes) is supporting evidence only.
- Report macro-F1 and macro-AUROC with patient-level bootstrap CIs and paired deltas against the
  ResNet-50 LOSO reference (`loso-*-local-20261001-132356*`, median padding).
- Report the 3-class results as the main result, with the binary Normal-vs-AMD view alongside. Score PAIMA on Normal/AMD only.
- Keep per-class recall by source-native label (the DME rows are the main signal).

## Prerequisite

- [x] Checkpoint selection weights sources by validation size. Already done in `f6d1315` (2026-09-17),
      before both LOSO campaigns, so the reference runs already use it.

## Experiments

| # | Experiment | Question | Training needed? | Status | Report |
|---|---|---|---|---|---|
| 1 | Binary rescoring of existing LOSO predictions (normal/AMD images and logits only) | How much of the gap is specific to DME? | No | done | [20261004-binary-rescore.md](reports/20261004-binary-rescore.md) |
| 2 | AdaBN: re-estimate BatchNorm statistics on the held-out source's unlabeled images | Do intensity and contrast differences between devices explain the gap? | No | done | [20261004-adabn.md](reports/20261004-adabn.md) |
| 3 | Augmentation that mimics device differences (speckle, gamma/contrast, blur, random padding/crop) | Does making training images look more varied close the gap? | 4 LOSO runs | todo | |
| 4 | Data-volume control: fused training with Kermany DME cut down to about 120 patients, or 50% data | How much of the held-out-Kermany drop is less data rather than a new device? | Fused CV | done | [20261005-kermany-dme-cut.md](reports/20261005-kermany-dme-cut.md) |
| 5 | Domain-specific pretraining: OCT foundation model, or MAE pretraining that includes the held-out source's images without labels | Does OCT pretraining, or adapting to the target without labels, help? | Pretraining + LOSO | todo | |
| 6 | Binary ResNet-50 runs comparable to the papers: Kermany↔Duke, plus binary LOSO | How do our numbers compare to OCT-SelfNet's ResNet-50 cross-dataset numbers? | Several runs | optional | |
| 7 | Per-source logit bias diagnostic: fit a class-bias shift on the held-out source's labels (patient cross-fit) and rescore existing LOSO predictions | Is the gap a decision-cutoff or label-prior shift rather than a feature problem? | No | done | [20261004-logit-bias.md](reports/20261004-logit-bias.md) |

Status values: `todo`, `running`, `done`, `dropped` (record the reason below).

## Decisions

- 2026-10-04: Keep DME. Report the binary Normal-vs-AMD view as a secondary result, not as a replacement task.
- 2026-10-04: Deprioritize more padding or hyperparameter tuning of the fused model, and a ResNet-vs-ViT
  comparison that doesn't change the pretraining.
- 2026-10-04 (exp. 1): Most of the held-out-source gap is DME. Binary macro-F1, fused CV vs LOSO:
  Kermany 98.4 vs 93.2 (3-class: 95.6 vs 73.2), OCTDL 96.9 vs 96.6, Duke B-scan 98.3 vs 96.8.
  PAIMA has no DME and still drops (90.3 vs 86.2, AUROC 95.6 vs 92.7). Held-out Kermany keeps binary
  AUROC 99.0 but normal recall falls to 89.3. Next: run experiment 2 (AdaBN) and check whether it fixes
  Kermany's normal recall and over-calling of DME; make PAIMA the main binary test; give experiment 4
  higher priority; experiment 6 is now optional.
- 2026-10-04 (exp. 2): AdaBN does not close the gap. Re-estimating all BatchNorm layers gives held-out
  Kermany +1.1 macro-F1 (DME recall 65.5 to 78.6), moves PAIMA by less than 1 point, and costs OCTDL
  about 20 points by pushing predictions into DME. Adapting only the stem and `layer1` is no better.
  Kermany's binary normal recall stays at 89, so its decision-cutoff shift is not a BatchNorm-statistics
  effect. Drop test-time BatchNorm adaptation. Next: experiments 3 (augmentation) and 4 (data volume). Keep
  per-source normalization during training (project plan §6) as a candidate.
- 2026-10-04 (exp. 7): Added the logit bias diagnostic and run it before experiments 3 and 4. It needs no
  training and decides how much they are worth: if an oracle per-source bias closes most of the Kermany
  and OCTDL gap, the cause is calibration or labeling. It uses target labels, so report it as an upper
  bound on a diagnostic, not a deployable method.
- 2026-10-04 (exp. 7 result): A per-source bias does not close the 3-class gap. The best in-sample bias for
  macro-F1 lifts held-out Kermany only 73.2 to 76.4 (seen 95.6) and OCTDL and Duke by about 1 point, so
  the DME problem is in the features, not the cutoff. The bias closes 52% of Kermany's binary gap (93.2
  to 95.9; normal recall 89 to 96, AUROC unchanged) but only 15% of PAIMA's, whose AUROC drop is a ranking
  problem. Calibration and label-prior shift are ruled out as the main cause. Next: experiment 4 (data
  volume) first, then 3 (augmentation), judged mainly on PAIMA.
- 2026-10-05 (exp. 4): Cutting Kermany train DME to 122 patients (the amount LOSO-Kermany has) leaves seen
  Kermany at 97.1 macro-F1 (reference 95.6, Δ +1.5 [+0.8, +2.2]), against 73.2 when the device is held out.
  Data volume is therefore not the cause of the held-out-Kermany drop; the device is. OCTDL and PAIMA do not
  move. Duke macro-F1 falls 2.1 (CI [-4.3, -0.1], DME recall 91.1 to 84.6), a weak lead from 45 eyes. The
  intervals cover test-patient sampling only; per-fold Kermany Δ ranges from -2.4 to +8.6. The cut removes only
  DME, so it does not cover the 50%-data variant. Next: experiment 3 (augmentation), judged mainly on PAIMA.
