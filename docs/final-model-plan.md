# Final Model Plan

**Started:** 2026-10-13 at `83347e9`. Closes out [cross-device-plan.md](cross-device-plan.md), which holds the full
history of the cross-device investigation (cause tracker, experiments 1-7, decision log). This plan covers only what
goes into the final results: which models exist, what they show, and the few runs still to do. Update it as runs
finish: inventory status, findings, and a decision-log entry.

**Time budget:** a few days of compute, excluding report writing. One local RTX 3060; the cluster (dgx01) is
draining since 2026-10-13 05:09 and takes no jobs until an admin resumes it.

## Final Variants

All three variants use ResNet-50 (ImageNet-pretrained, 224 px), fill-aware padding (`data.padding = "fill_aware"`),
seed 20260913, the same splits (Duke 5-fold CV; fixed Kermany, OCTDL and PAIMA splits) and fp32 evaluation.

| Variant | Question it answers | Geometry | Models |
|---|---|---|---|
| Per-dataset | How well does a model do on its own device? | none | Duke 5 folds; Kermany, OCTDL, PAIMA 1 each |
| Fused | How well does one model trained on all four devices do on each? | none (main); square crop (comparison) | 5 (Duke CV folds) |
| LOSO | How well does a model do on a device it never saw? | none and square crop, side by side | Duke held out: 1; others: 5 (Duke CV folds) |

Why square crop appears only as a comparison outside LOSO: it removes a cue (image size) that only misleads across
devices. Per-dataset and fused models are scored on devices they trained on, so there is nothing for it to fix.

## Inventory

Run paths are under `artifacts/runs/<source or fused>/resnet50/`. Status: `done`, `running`, `queued`, `todo`.

| Variant | Arm | Runs | Config | Status |
|---|---|---|---|---|
| Per-dataset | fill-aware | `local-final-fill-aware-duke-fold{1..5}`, `local-final-fill-aware-{octdl,paima,kermany}` | `resnet50-fill-aware.toml` | running (local, started 11:23) |
| Fused | fill-aware | `exp4-fill-aware-58577-fold{1..5}` | `resnet50-fused-fill-aware.toml` | done (cluster) |
| Fused | square crop | `fused-square-crop-58824-fold{1..5}` | `resnet50-fused-fill-aware-square-crop.toml` | queued (cluster job `58824`, node draining) |
| LOSO | fill-aware | `loso-<held out>-local-20261008-174938[-fold<N>]` | `resnet50-loso-fill-aware.toml` | done, all 4 held out |
| LOSO | square crop | `loso-<held out>-local-3a-r2-square-crop[-fold<N>]` | `resnet50-loso-fill-aware-square-crop.toml` | Kermany, PAIMA done; OCTDL, Duke running (local) |

Extra seeds 20260914 and 20260915 (exp. 2) exist for LOSO Kermany and PAIMA, both arms, under
`.scratchpad/049579f_exp2-seeds/runs/`. They set the noise floor; the final tables report seed 20260913 and give the
seed range next to it.

Superseded and not used in final tables: black-pad and median-pad runs of every variant (Sep 16 to Oct 1), mirror-pad
LOSO (exp. 3a, dropped). Per-dataset Kermany, OCTDL and PAIMA so far exist only as black-pad runs, hence the rerun.

Checkpoints: only `checkpoint-best.pt` is kept for finished runs (`checkpoint-last.pt` deleted 2026-10-13 to free
disk; exp. 4's stays on the cluster). Delete `checkpoint-last.pt` of each run above once it finishes.

## Findings So Far

Macro-F1 in percent. "Test split" means the source's own test split, which all three variants can be scored on;
"full source" means every image of a held-out source (LOSO only).

### Fused (seen devices)

| Source | Fill-aware (exp. 4) | Median pad (superseded) |
|---|---:|---:|
| Kermany | 97.4 ± 0.6 | 95.6 ± 3.4 |
| OCTDL | 95.9 ± 2.2 | 96.8 ± 1.2 |
| PAIMA | 90.8 ± 0.3 | 90.3 ± 0.5 |
| Duke | 95.2 ± 3.1 | 95.6 ± 4.3 |

Mean ± SD over 5 Duke folds; AUROC within 0.2 between the two. White fill does not inflate the seen score.

### LOSO (unseen devices), fill-aware

| Held out | Full source [95% CI] | AUROC | Test split | Seen (fused) | Gap on test split |
|---|---:|---:|---:|---:|---:|
| Kermany | 73.8 [72.5, 75.0] | 93.7 | 82.6 | 97.4 | ~15 |
| OCTDL | 92.0 [90.1, 93.8] | 98.7 | not computed | 95.9 | — |
| PAIMA | 86.4 [84.9, 87.7] | 92.8 | not computed | 90.8 | — |
| Duke | 89.7 [84.3, 93.7] | 97.8 | not computed | 95.2 | — |

CIs are patient-level bootstrap. Duke is one model on 45 eyes, so supporting evidence only.

### What causes the gap

- **Confirmed: image size gives away the label.** In Duke, every 768×496 image is DME and every 1024×496 is AMD;
  models trained with Duke learn this. Padding a normal Kermany image to 768 wide sends it to DME (12% → 60%); padding
  Duke's 512×496 images by 16 rows to Kermany's DME-only 512×512 sends 99% of normals to DME. All four held-out
  sources respond to image geometry (cross-device exps. 1, 1b, 1c).
- **Partial: DME seen only on other devices.** Removing Kermany's DME from training costs ~6 points.
- **Ruled out:** white rotation fill and padding, BatchNorm statistics, decision cutoff, training-set size,
  checkpoint selection.
- **Open:** device appearance (speckle, contrast, texture), and the PAIMA loss under square crop below.

### Square crop vs fill-aware, LOSO

Square crop trains on a random height × height window and tests on the center window, so the model cannot read the
image's width.

| Held out | Fill-aware (3 seeds) | Square crop (3 seeds) | Paired difference per seed |
|---|---|---|---|
| Kermany, full source | 73.8, 74.5, 76.4 | 86.6, 85.9, 84.8 | +12.9, +11.3, +8.4 |
| Kermany, AUROC | 93.7, 92.8, 94.0 | 97.6, 97.2, 97.0 | |
| Kermany, test split (seed 913) | 82.6 | 88.8 | +6.3 [+4.7, +7.8] |
| PAIMA | 86.4, 87.7, 87.0 | 85.2, 80.8, 75.6 | −1.2, −6.9, −11.5 |
| PAIMA, AUROC | 92.8, 93.3, 92.9 | 91.0, 83.6, 82.7 | |

- **Kermany: explained gain.** It holds at every seed and every image size, and is largest where the size cue was:
  768×496 normal → DME falls 36.6% → 4.5%. About half of the test-split gap closes (82.6 → 88.8 against 97.4 seen).
- **PAIMA: unexplained loss**, unstable across seeds, and in ranking (AUROC), not just the cutoff. Not explained by
  removing the size cue (mirror pad, which also weakens it, keeps PAIMA AUROC at 92.4; one seed only) or by
  test-time windowing (pooling several windows
  changes AUROC 91.0 → 91.1). Leading untested candidate: tissue scale. PAIMA images are all 768×496; the center
  496-wide window shows the tissue about 1.5× larger than a full-width resize does.

## Remaining Work

| # | Item | Where | Est. | Status |
|---|---|---|---|---|
| A | Per-dataset fill-aware | local | ~4-5 h shared GPU | running |
| B | LOSO square crop, OCTDL and Duke held out | local | ~4-5 h shared GPU | running |
| C | PAIMA tissue-scale check | local, after B | ~1 h | todo |
| D | Fused square-crop CV | cluster `58824`; local overnight as fallback | ~4-5 h | queued |
| E | Score all three variants | CPU | ~1 day | todo, as runs finish |

Dropped for this deadline: appearance augmentation (cross-device exp. 3b) and OCT-specific pretraining (exp. 5).
3b needs about half a day of code plus ~10 LOSO models, a full day of the only GPU; listed as future work.

### C. PAIMA tissue-scale check (no training)

**Question:** is square crop's PAIMA loss a tissue-scale mismatch?

**What:** rescore the three square-crop PAIMA-held-out seeds (15 models) with test windows that shrink the tissue
toward the scale the model trained on: resize the 768×496 image down before taking the center 496 window, at a few
factors (for example 1.0, 0.8, 0.65). Scored on Normal/AMD with the restricted softmax, as in the PAIMA evaluation.
Check first that factor 1.0 reproduces the stored predictions.

**Reading:** AUROC recovering toward fill-aware (~93) at some factor means the loss is a scale mismatch, a one-line
explanation for the report. No recovery leaves it as a stated open limitation.

### D. Fused square-crop CV

**Question:** what does square crop cost on devices the model trained on?

**What:** the fused fill-aware config plus `geometry = "square_crop"`. Notes and pull command:
`.scratchpad/83347e9_fused-square-crop/NOTES.md`.

**Scheduling:** when B finishes, check `58824`. Still pending → cancel it and run locally overnight. Running → leave
it. A local run is compared against exp. 4 across hardware; there is no fused seed noise floor, so treat differences
of a few points as unconfirmed.

**Reading:** if it scores within ~2 points of fused fill-aware on every source, square crop costs nothing in-domain
and the report can say so. A clear loss on a source stays in the report as the cost of removing the shortcut.

### E. Scoring

For every variant and arm:

- Macro-F1 and macro-AUROC with patient-level bootstrap CIs; paired deltas between arms on the same images.
- Per-class recall. PAIMA on Normal/AMD only, with the binary view next to the 3-class result.
- Seen vs held out on the same images: score held-out OCTDL, PAIMA and Duke on their test splits too, so the gap
  table is complete (only Kermany has it now).
- Kermany split by image size, as in cross-device exp. 1.

Measurement rules from [cross-device-plan.md](cross-device-plan.md#measurement-rules) apply, including the seed-based
minimum deltas.

## Reading Rule for the Side-by-Side LOSO Table

Fixed before the OCTDL and Duke square-crop results come in, so the choice does not depend on them.

- The final LOSO table shows both arms for all four held-out sources. Neither arm is chosen per source from test
  results.
- **If OCTDL and Duke are both within noise of fill-aware** (paired CI includes 0, or the delta is under the working
  delta for the arm): square crop is the recommended LOSO setup, with PAIMA's loss stated as an exception.
- **If either drops beyond noise:** fill-aware stays the main LOSO result and square crop is presented as an ablation
  that removes a confirmed shortcut, at a cost on some devices.
- Either way the Kermany result is reported with its mechanism (size cue, by-size breakdown), and the PAIMA loss with
  check C's outcome.
- Duke held out is one model on 45 eyes: it can confirm a direction but cannot by itself trigger the second case
  unless its CI excludes 0.

## Decision Log

- 2026-10-13: Final variants fixed as per-dataset, fused and LOSO, all fill-aware. Square crop goes in as a side-by-side
  arm for LOSO and a comparison for fused, not as a replacement: it explains Kermany's gain but costs PAIMA for an
  unknown reason. Per-dataset rerun with fill-aware because the existing single-source runs are black-pad. 3b and 5
  dropped for the deadline. Cluster job `58824` (fused square crop) queued behind a draining node; local fallback
  overnight.
