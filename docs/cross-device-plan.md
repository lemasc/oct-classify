# Cross-Device Experiment Plan

**Started:** 2026-10-04 at `98be7ad`. Follows the ResNet-50 fused and LOSO campaigns in
[20261002-resnet50-training.md](reports/20261002-resnet50-training.md). The overall project plan is
[full-project-plan.md](full-project-plan.md).

## Question

Why does performance drop when testing on a device (source) not seen in training, and which changes close the gap?

LOSO evidence so far: macro-F1 is 96–100 when the test source was in training, but drops for unseen
sources. Held-out Kermany falls to 73 macro-F1. DME recall drops most: in held-out Duke, 25% of DME is
predicted normal; in held-out OCTDL, 25% of DME is predicted AMD.

## Measurement Rules

- **Primary sets:** LOSO held-out Kermany, OCTDL, and PAIMA. Duke (45 eyes) is supporting evidence only.
- Report macro-F1 and macro-AUROC with patient-level bootstrap CIs and paired deltas against the
  ResNet-50 LOSO reference (`loso-*-local-20261001-132356*`, median padding).
- Report the 3-class results as the main result, with the binary Normal-vs-AMD view alongside. Score PAIMA on Normal/AMD only.
- Keep per-class recall by source-native label (the DME rows are the main signal).

## Prerequisite

- [ ] Replace the unweighted `validation_mean_macro_f1` checkpoint-selection metric with a size-weighted
      or Kermany/PAIMA-based metric before any new training run. Do not change the reference runs.

## Experiments

| # | Experiment | Question | Training needed? | Status | Report |
|---|---|---|---|---|---|
| 1 | Binary rescoring of existing LOSO predictions (normal/AMD images and logits only) | How much of the gap is specific to DME? | No | todo | |
| 2 | AdaBN: re-estimate BatchNorm statistics on the held-out source's unlabeled images | Do intensity and contrast differences between devices explain the gap? | No | todo | |
| 3 | Augmentation that mimics device differences (speckle, gamma/contrast, blur, random padding/crop) | Does making training images look more varied close the gap? | 4 LOSO runs | todo | |
| 4 | Data-volume control: fused training with Kermany DME cut down to about 120 patients, or 50% data | How much of the held-out-Kermany drop is less data rather than a new device? | Fused CV | todo | |
| 5 | Domain-specific pretraining: OCT foundation model, or MAE pretraining that includes the held-out source's images without labels | Does OCT pretraining, or adapting to the target without labels, help? | Pretraining + LOSO | todo | |
| 6 | Binary ResNet-50 runs comparable to the papers: Kermany↔Duke, plus binary LOSO | How do our numbers compare to OCT-SelfNet's ResNet-50 cross-dataset numbers? | Several runs | optional | |

Status values: `todo`, `running`, `done`, `dropped` (record the reason below).

## Decisions

- 2026-10-04: Keep DME. Report the binary Normal-vs-AMD view as a secondary result, not as a replacement task.
- 2026-10-04: Deprioritize more padding or hyperparameter tuning of the fused model, and a ResNet-vs-ViT
  comparison that doesn't change the pretraining.
