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
| Per-dataset | fill-aware | `local-final-fill-aware-duke-fold{1..5}`, `local-final-fill-aware-{octdl,paima,kermany}` | `resnet50-fill-aware.toml` | done |
| Fused | fill-aware | `exp4-fill-aware-58577-fold{1..5}` | `resnet50-fused-fill-aware.toml` | done (cluster) |
| Fused | square crop | `fused-square-crop-local-fold{1..5}` (local) or `fused-square-crop-58824-fold{1..5}` (cluster) | `resnet50-fused-fill-aware-square-crop.toml` | done (local, 2026-10-13 20:44 to 2026-10-14 02:48); `58824` cancelled |
| LOSO | fill-aware | `loso-<held out>-local-20261008-174938[-fold<N>]` | `resnet50-loso-fill-aware.toml` | done, all 4 held out |
| LOSO | square crop | `loso-<held out>-local-3a-r2-square-crop[-fold<N>]` | `resnet50-loso-fill-aware-square-crop.toml` | done, all 4 held out |

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

### Per-dataset (own device), fill-aware

| Source | Fill-aware | Old run (superseded) |
|---|---:|---:|
| Duke | 94.1 ± 5.7 (eye level 97.7 ± 5.1) | 91.3 ± 7.6 (median pad) |
| OCTDL | 97.6 | 97.1 (black pad) |
| PAIMA | 90.0 | 90.0 (black pad) |
| Kermany | 98.8 | — |

Test split. Duke is mean ± SD over 5 folds; its fold 4 is low (84.2) in every variant, so the spread reflects that
fold's test patients. AUROC: Duke 98.9, OCTDL 99.5, PAIMA 95.2, each within 0.2 of the old run; Kermany 99.97. Fill-aware changes
little on a model's own device, as expected.

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
| Kermany | 73.8 [72.4, 74.9] | 93.7 | 82.6 | 97.4 | −14.8 [−16.5, −13.3] |
| OCTDL | 92.0 [90.0, 93.8] | 98.7 | 91.6 | 95.9 | −4.3 [−9.8, −0.2] |
| PAIMA | 86.4 [84.8, 87.7] | 92.8 | 85.2 | 90.8 | −5.7 [−8.2, −3.6] |
| Duke | 89.7 [84.4, 93.7] | 97.8 | 89.7 | 95.2 | −5.5 [−10.1, −2.4] |

CIs are patient-level bootstrap. Duke is one model on 45 eyes, so supporting evidence only.

### What causes the gap

- **Confirmed: image size gives away the label.** In Duke, every 768×496 image is DME and every 1024×496 is AMD;
  models trained with Duke learn this. Padding a normal Kermany image to 768 wide sends it to DME (12% → 60%); padding
  Duke's 512×496 images by 16 rows to Kermany's DME-only 512×512 sends 99% of normals to DME. All four held-out
  sources respond to image geometry (cross-device exps. 1, 1b, 1c).
- **Partial: DME seen only on other devices.** Removing Kermany's DME from training costs ~6 points.
- **Ruled out:** white rotation fill and padding, BatchNorm statistics, decision cutoff, training-set size,
  checkpoint selection.
- **Open:** device appearance (speckle, contrast, texture), and the PAIMA loss under square crop below (not tissue scale, see check C).

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
  test-time windowing (pooling several windows changes AUROC 91.0 → 91.1). Tissue scale was tested (check C) and does not explain it: shrinking the tissue toward training scale lowers AUROC
  instead of recovering it (below).

OCTDL and Duke held out (seed 20260913 only; paired patient-level bootstrap, script and output in
`.scratchpad/14e161e_loso-sq-octdl-duke/`):

| Held out | Fill-aware | Square crop | Paired difference | AUROC (fill → square) | Δ AUROC |
|---|---:|---:|---:|---|---:|
| OCTDL, full source (5 models, 637 patients) | 92.0 [90.0, 93.7] | 87.2 [84.9, 89.2] | −4.8 [−7.2, −2.5] | 98.7 → 97.9 | −0.8 [−2.2, +0.2] |
| Duke, full source (1 model, 45 eyes) | 89.7 [84.4, 93.6] | 87.2 [81.2, 91.7] | −2.5 [−4.8, −0.5] | 97.8 → 98.6 | +0.8 [+0.0, +2.1] |

- **OCTDL: loss beyond noise, in the cutoff.** Normal recall 95.1 → 84.2 (−11.0 [−14.4, −7.7]); normal → DME 2 →
  181 images. AUROC holds, unlike PAIMA. Square-crop folds spread 82.9-91.7 vs 91.5-92.8. No OCTDL seed floor exists;
  the ~3-point square-crop Kermany working delta is used.
- **Duke: same direction, supporting only.** DME recall 75.8 → 67.6 (−8.3 [−13.5, −3.5]); the delta is under ~3.
- **Reading rule outcome:** OCTDL drops beyond noise, so fill-aware stays the main LOSO result and square crop is
  presented as an ablation that removes the size shortcut at a cost on some devices.

## Remaining Work

| # | Item | Where | Est. | Status |
|---|---|---|---|---|
| A | Per-dataset fill-aware | local | — | done |
| B | LOSO square crop, OCTDL and Duke held out | local | — | done |
| C | PAIMA tissue-scale check | local | ~1 h | done, negative |
| D | Fused square-crop CV | local, 2026-10-13 20:44 to 2026-10-14 02:48 (`58824` cancelled) | ~6 h total | done |
| E | Score all three variants | CPU | ~1 min | done 2026-10-14, seed 20260913 only |
| F | Fused zero-Kermany-DME CV, fill-aware | local | ~8 h | **stopped 2026-10-14 14:5x, invalid** (selection flaw, see F); fair rerun deferred |
| G | Logit-bias rescoring of fill-aware LOSO | CPU, `docs/paper` worktree | minutes | done 2026-10-14 (`docs/paper` `285ce16`) |
| H | AdaBN on fill-aware LOSO (port from `legacy/docs/cross-device-plan`) | port: temporary worktree; eval: local GPU, `docs/paper` worktree | ~1-1.5 h GPU | port done (`feat/adabn-port` `8a65095`, merged into `docs/paper`); done 2026-10-14 (`docs/paper` `ec4f501`) |
| I | Grad-CAM on final models | CPU, `docs/paper` notebook | minutes | done 2026-10-14 (`notebooks/gradcam_shortcut.py`, `e7e4982`; paper `4a075b0`) |

F-I rerun, on the final fill-aware models, the cross-device diagnostics the paper cites (originally run on median-pad
models, reports on the `legacy/docs/cross-device-plan` branch). The Kermany DME cut to 122 patients is not rerun; the
paper cites the legacy result in one sentence. G-I are analysis and are logged here when done.

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

**Result (done 2026-10-13):** negative. Factor 1.0 reproduces the stored metrics (max |Δ| F1 7e-5, AUROC 2.7e-3).
AUROC by factor, mean over 5 folds (scripts and output in `.scratchpad/14e161e_paima-tissue-scale/`):

| Factor | Seed 913 | Seed 914 | Seed 915 | Pooled Δ AUROC vs 1.0 [95% CI] |
|---|---:|---:|---:|---|
| 1.00 | 91.0 | 83.5 | 82.7 | |
| 0.90 | 89.5 | 81.8 | 80.9 | −1.7 [−1.9, −1.5] |
| 0.80 | 86.8 | 80.0 | 76.8 | −4.6 [−5.1, −4.1] |
| 0.70 | 84.4 | 76.1 | 72.8 | −8.0 [−8.7, −7.3] |
| 0.65 | 82.7 | 75.4 | 70.0 | −9.7 [−10.5, −9.0] |

Fill-aware reference: 92.8, 93.3, 92.9. Patient-level bootstrap, 437 groups, 15 models. The drop is monotone at every
seed and fold; macro-F1 falls further (−28 at 0.65) as the cutoff shifts. Shrinking the tissue does not recover PAIMA,
so tissue scale (in this direction) does not explain the loss. Enlarging the tissue was not tested. The loss is
fold-dependent (seed 915 fold 5 AUROC 63.7; seed 913 folds 84-94). Stated in the report as an open limitation.

### D. Fused square-crop CV

**Question:** what does square crop cost on devices the model trained on?

**What:** the fused fill-aware config plus `geometry = "square_crop"`. Notes and pull command:
`.scratchpad/83347e9_fused-square-crop/NOTES.md`.

**Scheduling:** `58824` was still pending after B and C, so it was cancelled and the five folds run locally in
sequence (launcher `.scratchpad/83347e9_fused-square-crop/run-local.sh`, log `run-local.log`). A local run is compared against exp. 4 across hardware; there is no fused seed noise floor, so treat differences
of a few points as unconfirmed.

**Reading:** if it scores within ~2 points of fused fill-aware on every source, square crop costs nothing in-domain
and the report can say so. A clear loss on a source stays in the report as the cost of removing the shortcut.

**Result (done 2026-10-14):** all five folds finished (6 h total, 0.9-1.7 h per fold). Macro-F1 (AUROC), mean ± SD over 5
Duke folds, test split, point estimates from `metrics-test.json` (no CIs yet, see E):

| Source | Square crop (local) | Fill-aware (exp. 4, cluster) | Δ F1 |
|---|---:|---:|---:|
| Kermany | 97.7 ± 0.9 (99.8) | 97.4 ± 0.6 (99.8) | +0.3 |
| OCTDL | 96.0 ± 0.7 (99.9) | 95.9 ± 2.2 (99.5) | +0.1 |
| PAIMA | 93.0 ± 0.8 (97.0) | 90.8 ± 0.3 (95.7) | +2.2 |
| Duke | 93.1 ± 6.1 (98.2) | 95.2 ± 3.1 (99.6) | −2.1 |

Kermany and OCTDL are within the ~2-point reading; PAIMA (+2.2) and Duke (−2.1) sit just past it in opposite
directions. Most of Duke's drop is fold 4 (83.7 vs 89.9; fold 4 is low in every variant). Hardware differs (local vs
cluster) and there is no fused seed floor, so these differences are unconfirmed. Provisional wording (superseded by E's paired result): square crop
costs nothing in-domain on Kermany and OCTDL, and moves PAIMA and Duke by about 2 points either way. Confirm with the
paired patient-level deltas in E before stating it. E confirmed it: see E's result below.

### E. Scoring

For every variant and arm:

- Macro-F1 and macro-AUROC with patient-level bootstrap CIs; paired deltas between arms on the same images.
- Per-class recall. PAIMA on Normal/AMD only, with the binary view next to the 3-class result.
- Seen vs held out on the same images: score held-out OCTDL, PAIMA and Duke on their test splits too, so the gap
  table is complete (only Kermany has it now).
- Kermany split by image size, as in cross-device exp. 1.

Measurement rules from [cross-device-plan.md](cross-device-plan.md#measurement-rules) apply, including the seed-based
minimum deltas.

**Result (done 2026-10-14):** script `.scratchpad/6fc182a_final-scoring/score.py` (+ `report.py`), outputs in
`data/final-scoring/` (`results.json`, `tables.md` with every table incl. per-class recall and the Normal/AMD view).
Patient-level cluster bootstrap (2000 draws; AUROC 500), one resample shared by all arms of a comparison so deltas are
paired. Duke 5-fold arms are pooled out-of-fold (all 3231 images, 45 eyes), the same images LOSO Duke sees. Stored
metrics reproduce exactly (F1 diff 0, AUROC diff <3e-4). Seed 20260913 only. Macro-F1 in percent.

**Seen devices, paired on the test split:**

| Source | Square crop − fused fill-aware | Fused fill-aware − per-dataset |
|---|---|---|
| Kermany | +0.3 [−0.5, +1.1] | −1.4 [−2.1, −0.7] |
| OCTDL | +0.2 [−3.4, +3.5] | −1.7 [−4.3, +0.6] |
| PAIMA (Normal/AMD) | +2.1 [+1.1, +3.3] | +0.8 [−0.5, +2.2] |
| Duke | −2.4 [−7.5, +1.7] | +1.1 [−2.2, +4.5] |

- **Square crop costs nothing in-domain:** no source loses beyond noise. PAIMA gains (AUROC +1.3 [+0.7, +2.1]). Duke's
  −2.4 is within noise (fold 4). Caveat: local vs cluster hardware, no fused seed floor.
- **Fusing costs a little on Kermany only** (−1.4, small but outside noise); the others are within noise of the
  per-dataset model.

**LOSO gap on the test split (held out minus fused, same images; macro-F1 [95% CI]):**

| Held out | Fill-aware | Square crop | Square − fill (held out) |
|---|---|---|---|
| Kermany | −14.8 [−16.5, −13.3] | −8.9 [−10.4, −7.4] | +6.3 [+4.7, +7.8] |
| OCTDL | −4.3 [−9.8, −0.2] | −10.7 [−15.9, −6.0] | −6.2 [−13.3, +0.5] |
| PAIMA | −5.7 [−8.2, −3.6] | −8.3 [−10.5, −6.4] | −0.5 [−2.5, +1.7] |
| Duke | −5.5 [−10.1, −2.4] | −5.7 [−11.4, −0.2] | −2.5 [−4.7, −0.4] |

- **Every held-out source loses beyond noise, in both arms.** Fill-aware gaps are 4-6 points for OCTDL, PAIMA and Duke,
  and ~15 for Kermany. Square crop closes about 40% of Kermany's gap and widens the others, so its Kermany gain does not
  generalise (consistent with the reading rule).
- **Most of the gap is DME recall.** Held-out DME recall falls 13-34 points against fused (3-class view), while the
  Normal/AMD-only gap is small for OCTDL (−1.7 [−4.5, +0.6]) and Duke (−1.5 [−4.2, +0.2]), and remains for Kermany
  (−11.2, normal recall) and PAIMA (−5.7, AMD recall −11.6). This matches the "DME seen only on other devices" cause.
- **Normal/AMD view, square crop vs fill-aware (full source):** OCTDL +0.3 [−0.7, +1.3] and Duke +0.5 [−0.1, +1.4]:
  the OCTDL and Duke losses above come from DME, not from Normal/AMD.
- **Kermany test split is not representative:** 942 of its 1000 images are 1024×496, so the held-out Kermany test
  score (82.6) sits above the full-source score (73.8); use the full source for the unseen-device claim.

**Kermany by image size (LOSO, full source, normal recall fill → square crop):** 512×496 80.5 → 86.4; 768×496
58.4 → 91.1 (normal → DME 36.6% → 4.5%); 1536×496 87.7 → 89.6; 1024×496 68.5 → 78.4. DME recall rises at four of five
sizes (512×512 69.1 → 81.7; 1536×496 55.9 → 78.1) but falls at 768×496 (89.0 → 79.3). The gain is largest where the size
cue was, as in the Kermany finding above. On the fused Kermany test split sizes other than 1024×496 have ≤ 29 images
per class, so no size claim is made there.

**PAIMA:** only Normal/AMD exists, so there is no 3-class view. Per-dataset 90.0, fused 90.8, fused square crop 93.0.

### F. Fused zero-Kermany-DME CV (stopped: checkpoint selection flaw)

**Question:** with the Kermany device seen in training but none of its DME, how much of Kermany's score is lost?
LOSO-Kermany differs from fused in two ways at once: the device is unseen, and DME is seen only from OCTDL and Duke
(122 patients). This run removes only the second, so it tests the "DME seen only on other devices" cause behind the
paper's "most of the remaining gap is DME".

**Flaw (found 2026-10-14):** the splits remove Kermany DME from train but keep it in **val**, and checkpoint
selection and early stopping use mean val macro-F1. Training teaches the model that Kermany-looking images are never
DME, so Kermany val falls as training goes on and the best checkpoint is the earliest one. Fill-aware fold 1 kept
**epoch 0** and stopped after 11 epochs (Kermany val 0.87 → ~0.70); the reference exp. 4 keeps epochs ~10-30. The
comparison with fused therefore mixes "no Kermany DME" with "barely trained".

**The legacy run has the same flaw:** `kdme0-local-20261008-011217` kept epochs 0-2 of 11-13, against the median-pad
reference's 10-23. Its conclusions (Kermany −5.6, "Kermany DME matters for Duke" −16.6 DME recall, the "unexplained"
PAIMA −4.7) may be partly or wholly undertraining and are **not cited**. The DME cut to 122 patients
(`kdme122-local-20261004-210330`) is not affected: it kept epochs 7-30, like the reference.

**Runs left behind:** `fused-zero-dme-local-fold1` (complete, invalid) and `fused-zero-dme-local-fold2` (partial,
stopped at 14:5x). Do not score them. Splits `artifacts/splits/duke-cv-kermany-dme0` (Kermany val still has DME).

**Fair rerun (deferred, not yet decided):** remove Kermany DME from val as well (keep it in test), so selection never
uses Kermany DME, as a LOSO model never validates on its held-out device. Split change only (filter `split in
{"train","val"}` instead of `"train"` in `make_splits.py`, new dir e.g. `duke-cv-kermany-dme0-noval`). Check fold 1's
best epoch is in the reference range before letting all 5 run (~8 h). Alternative: fixed epoch count (~28, the
reference median), which needs a config change. Until a fair run exists, the paper rests the DME claim on the LOSO
DME-recall breakdown only.

### H. AdaBN on fill-aware LOSO (handoff)

**Question:** does re-estimating BatchNorm statistics on the unseen device close the gap? Legacy median-pad result:
Kermany +1.1, OCTDL about −20 (`docs/reports/20261004-adabn.md` on `legacy/docs/cross-device-plan`). Smoke run on
fill-aware LOSO OCTDL fold 1: 91.7 → 82.3.

**What:** from the `docs/paper` worktree (`../repo-paper`, which has the port), the 16 fill-aware LOSO checkpoints
`loso-{kermany,octdl,paima}-local-20261008-174938-fold{1..5}` and `loso-duke-local-20261008-174938`, each on its
held-out source's full manifest, `--adapt-bn --adapt-bn-layers all` (patient cross-fit, default folds and seed),
output `<run>/evaluations/<source>-loso-adabn-all/`. Class labels as in the stored `<source>-loso/metrics.json`
(PAIMA: normal, amd). Score with the paired patient-level bootstrap against the stored `<source>-loso` predictions,
as `data/paper/logit-bias/` does; outputs in `data/paper/adabn/` (strict JSON).

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
- 2026-10-13 16:20 (progress): Per-dataset Duke, OCTDL and PAIMA done; within noise of the old runs on their own
  device. Square-crop LOSO folds run longer than fill-aware (35-36 epochs vs 23-31 for OCTDL held out), so B ends
  ~19:00-19:30 instead of ~17:30 and per-dataset Kermany ~20:00 (it shares the GPU and is slowed ~5×; LOSO is not
  slowed). C and the local fallback for D start after B. Cluster still draining.
- 2026-10-13 19:30 (A and B done): per-dataset Kermany 98.8 macro-F1. LOSO square crop vs fill-aware: OCTDL −4.8
  [−7.2, −2.5] (normal → DME; AUROC holds), Duke −2.5 [−4.8, −0.5] (DME recall). By the pre-set reading rule,
  fill-aware is the main LOSO result and square crop an ablation. C started 19:36 (local). Cluster still draining;
  D runs locally after C, start not yet set, and `58824` is cancelled then if still pending. Launcher:
  `.scratchpad/83347e9_fused-square-crop/run-local.sh`.
- 2026-10-13 20:44 (C done, D started): PAIMA tissue-scale check negative: shrinking the test tissue lowers AUROC
  monotonically (pooled −1.7 at 0.90 to −9.7 at 0.65), so scale does not explain square crop's PAIMA loss; it stays an
  open limitation. The first D launcher had died with the earlier session; relaunched detached at 20:44, `58824`
  cancelled, folds 1-5 train locally in sequence. `checkpoint-last.pt` of each fold is deleted by hand when it finishes.
- 2026-10-14 (D done): fused square-crop CV finished locally, five folds. Per-source macro-F1 within 0.3 of fused
  fill-aware on Kermany and OCTDL, +2.2 on PAIMA, -2.1 on Duke (mostly fold 4). Unconfirmed across hardware, no seed
  floor. All training runs are now done; only scoring (E) remains. `checkpoint-last.pt` of the five folds still to
  delete.
- 2026-10-14 (E done): all variants scored with patient-level bootstrap and paired deltas (`data/final-scoring/`).
  Square crop costs nothing in-domain (PAIMA +2.1, others within noise). Every held-out source loses beyond noise
  against its fused model on the same test split (fill-aware −4 to −6, Kermany −15), mostly through DME recall;
  square crop narrows Kermany's gap and widens OCTDL's, so the reading-rule outcome stands (fill-aware main, square
  crop ablation). Seed 20260913 only. Remaining: delete `checkpoint-last.pt` of the five fused square-crop folds
  (held back at user's request).
- 2026-10-14 (paper diagnostics): the paper cites the cross-device diagnostics, so they are rerun on the final
  fill-aware models for consistency: F (zero-Kermany-DME fused CV, local GPU), G (logit bias, CPU), H (AdaBN, port in
  a temporary worktree, GPU eval after F), I (Grad-CAM, after F). The DME cut to 122 patients is not rerun. Paper and
  analysis work moves to a `docs/paper` worktree (`../repo-paper`) branched from here; training stays here.
- 2026-10-14 (G done, H ported): logit bias on fill-aware LOSO (`data/paper/logit-bias/` on `docs/paper`). `none`
  reproduces final scoring exactly. Cross-fit bias: Kermany 73.8 → 77.5 (+3.8 [+3.5, +4.0]); OCTDL +0.5 and Duke +0.6,
  within noise; in-sample grid ceiling on Kermany 80.8 against 97.4 seen. Larger than the legacy +0.5 (fill-aware ranks
  DME better, AUROC 93.7 vs 91.5) but the conclusion holds: the cutoff is a minor part of the gap. AdaBN ported from
  `c3c1d93` (statistics pass now fp32); with adapt-bn off it reproduces stored LOSO OCTDL fold 1 (macro-F1 identical,
  max |Δp| 0.002 on CPU); smoke run with adapt-bn on: 91.7 → 82.3, as in legacy. Full eval waits for F.
- 2026-10-14 (F stopped): zero-DME fold 1 kept its epoch-0 checkpoint because Kermany val still holds DME while
  training removes it; the legacy zero-DME run had the same flaw (epochs 0-2). Run stopped during fold 2; legacy
  zero-DME numbers withdrawn from the paper. Fair rerun (no Kermany DME in val either) deferred. AdaBN (H) runs
  first on the free GPU, then Grad-CAM (I).
- 2026-10-14 (H and I done): AdaBN-all on the 16 fill-aware LOSO checkpoints (`data/paper/adabn/`): Kermany +2.7
  [+2.2, +3.2], Duke +0.3 (noise), OCTDL −7.5 [−9.9, −5.4] (DME over-called: DME recall 77.3 → 95.4, AMD 97.8 →
  89.0), PAIMA Normal/AMD −1.0. Same conclusion as legacy (Kermany +1.1, OCTDL −20), smaller OCTDL loss. Grad-CAM
  moved to a CPU notebook (`notebooks/gradcam_shortcut.py`, all 5 LOSO-Kermany folds, randomization check and a
  band-pixel intervention; summary `data/paper/gradcam/summary.json`). Changing only the letterbox band pixels
  removes most wrong DME calls on 768×496 normal and AMD images in every fold (normal: 0-28% remain with speckle
  bands), and correct calls barely move (0 of 800 become DME). Grad-CAM puts ~75% of the map on the bands in folds
  1-2 but 8-14% in folds 4-5, so the paper uses the intervention as evidence and Grad-CAM as illustration. The
  existing `analysis gradcam` CLI ignores `square_crop` geometry; the notebook uses `ManifestImageDataset` instead.
- 2026-10-14 (known minor issue, not fixed): the Normal/AMD view renormalises p(normal)+p(AMD); 143 of ~450k
  fill-aware LOSO-Kermany normal/AMD rows have both exactly 0 (fp32 underflow with p(DME)=1), giving NaN rows that
  argmax counts as normal. Effect on the Kermany Normal/AMD score is below 0.03 points; `score.py` and the
  diagnostics inherit it.
