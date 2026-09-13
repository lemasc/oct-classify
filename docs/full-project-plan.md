# Multi-Dataset OCT Classification — Project Plan

**Scope:** First pass: Normal / AMD (merged: AMD-unspecified + CNV + Drusen) / DME, across Duke, Kermany, OCTDL, and Paima/NEH. OCTID is deferred until a defensible split design is available.
**Timeline:** ~8 weeks
**Compute:** RTX 3060 (local) · A100 cluster (1g.10gb MIG slices, on-demand) · 7g.80gb slices (queued)

---

## 1. The Real Complexity: Missing DME Labels

After merging CNV + Drusen + AMD-unspecified into a single AMD class, your label availability looks like this:

| Class | Duke | Kermany | OCTDL | Paima/NEH |
|---|---|---|---|---|
| Normal | ✓ | ✓ | ✓ | ✓ |
| AMD (merged) | ✓ | ✓ (via CNV) | ✓ | ✓ (via CNV/Drusen) |
| DME | ✓ | ✓ | ✓ | **✗** |

Once merged, AMD and Normal are available everywhere — but **DME is missing from Paima/NEH**. This is a partial-label, not a missing-modality, problem: Paima/NEH is not "off-domain" for DME; it simply has no DME images at all (not even unlabeled ones you could pseudo-label). This shapes two decisions:

- **Training a unified 3-way classifier on fused labeled data:** naively training softmax cross-entropy with Paima batches will push DME logits toward zero for images that were never screened for DME, which is a false negative signal, not a true one. Use a **masked loss** (only backprop the classes a given dataset actually screened for) or train **one-vs-rest binary heads** instead of a single softmax, so absent classes don't contaminate gradients.
- **Cross-dataset evaluation:** evaluate Paima/NEH on Normal/AMD only. Report a **full 3-class grid** for the Duke/Kermany/OCTDL subset and a separate **2-class (Normal/AMD) 4×4 grid** for all active sources. Keep these visually and numerically separate in your results tables — conflating them will make your off-domain numbers look worse than they are.

---

## 2. Experimental Design

Mirroring your papers' structure, adapted for 4 sources and partial labels:

| Stage | What | Backbone(s) |
|---|---|---|
| **Baseline** | ImageNet-pretrained, no SSL, fine-tuned independently per dataset; establish within- and cross-source grids before fusion | ResNet-50 first; ViT-B follows through the same pipeline |
| **Fused supervised** | Train once on all labeled sources using a masked loss for Paima/NEH, only after the independent baseline and cross-source duplicate review are complete | ResNet-50 |
| **Proposed (SSL)** | MAE pretraining on fused *unlabeled* images from all 4 active sources → fine-tune per dataset (masked/one-vs-rest loss) → cross-dataset eval | SwinV2 (matches your papers), consider ViT-B as a second run if time allows |
| **Stretch: new architecture** | See §6 | TBD, gated on results from above |

**Splits:** patient-level, not image-level, wherever patient/volume IDs are available (Duke, Kermany especially — B-scans from the same volume are near-duplicates and will leak across train/test if split randomly). Where no patient ID exists, dedupe with perceptual hashing before splitting.

---

## 3. Compute Allocation

Your bottleneck isn't total compute, it's **queue time on the 7g.80gb slices**. Plan around that explicitly rather than treating all compute as interchangeable.

| Resource | Use it for | Why |
|---|---|---|
| **RTX 3060 (local)** | Data pipeline dev/debugging, preprocessing, small sanity-check training runs, writing eval/plotting scripts, a *tiny* pilot MAE run (low-res, 1-epoch, subsampled) to catch bugs | Fast iteration, no queue, but not enough VRAM for full MAE pretraining at target resolution/batch size |
| **1g.10gb MIG slices** | Future parallel baseline fine-tuning, linear-probe evaluation of SSL-pretrained encoders, hyperparameter sweeps, inference/eval jobs | Low memory but no queue wait — good for many small parallel jobs |
| **7g.80gb (queued)** | MAE self-supervised pretraining on the fused dataset (needs large batch size + memory), full fine-tuning of the largest runs | This is your scarce resource — debug elsewhere first, submit here only once the job is verified to run end-to-end |

**Practical rule:** never let a job's first-ever execution be on the queued 7g.80gb slice. Validate the exact training script on the 3060 or a MIG slice at reduced scale first.

---

## 4. Week-by-Week Timeline

| Week | Focus | Compute | Milestone |
|---|---|---|---|
| **1** | Dataset audit and split work is complete. Implement the supervised image loader, ResNet-50 model factory, checkpointing, and evaluation harness. Run a capped Duke train/validate/test flow locally on the RTX 3060, then resume it and evaluate the checkpoint against Paima using the Normal/AMD intersection. | 3060 | Reproducible local flow check; no SLURM submission |
| **2** | Run ImageNet-pretrained ResNet-50 independently for Duke, Kermany, OCTDL, and Paima. Report each source's held-out test result and preserve predictions. | 3060 | Four independent ResNet-50 baselines |
| **3** | Run the complete cross-dataset grid: 3-class among Duke/Kermany/OCTDL and a separate Normal/AMD grid including Paima. Add ViT-B through the same model interface only after ResNet results are stable. | 3060 | Baseline cross-evaluation table; optional ViT-B starts |
| **4** | Review retained cross-source duplicate candidates and decide whether fused training needs an additional manifest version. Then implement source-balanced, masked-loss fused supervised ResNet-50. | 3060 | Defensible fused-supervised baseline |
| **5** | Compare independent and fused supervised baselines; build bootstrap confidence intervals, error analysis, and embedding tooling. Decide whether measured cross-source gaps justify SSL. | 3060 + 1g.10gb if needed | Baseline decision point |
| **6** | Error analysis: confusion matrices per dataset, embedding t-SNE/UMAP, per-class AUC-ROC/PR, significance tests. **Decision point:** go/no-go on new architecture based on time remaining and where the SSL model is weak. | 1g.10gb | Results consolidated; go/no-go decision made |
| **7** | New architecture experiment (see §6) — scoped tightly to *one* idea, not several. | 7g.80gb (queued) for training, 1g.10gb for eval | New architecture results (even if preliminary) |
| **8** | Final evaluation pass, figures/tables, writeup. This week is also your buffer for any queue delays that pushed earlier weeks back. | 1g.10gb + 3060 | Final deliverable |

---

## 5. Risk Mitigation

- **Do not submit to SLURM until the local supervised flow is verified.** The first queued workload is now contingent on the independent and fused supervised baseline table showing a remaining generalization gap.
- **Checkpoint aggressively.** 7g.80gb jobs that get preempted or interrupted by queue churn should resume, not restart. Save encoder state every N steps.
- **Class imbalance.** AMD will likely dominate after merging (3 original classes flow into it). Use class-weighted loss or a balanced sampler on top of the masked-loss handling above — these solve different problems and you need both.
- **Leakage.** Double-check Kermany and Duke in particular for patient/volume-level duplication between train and test; both are known to have this issue if split naively.

---

## 6. Stretch Goal: New Architecture Ideas

If Weeks 1–6 land on schedule, pick **one** of these rather than spreading Week 7 thin:

1. **Partial-label-aware head.** Replace the shared softmax classifier with a one-vs-rest (sigmoid) head per class, explicitly designed for datasets with structurally absent classes — this is the most direct extension of what your papers didn't have to solve (they were binary).
2. **Domain-conditioned normalization.** Add per-source batch/instance normalization (or FiLM-style conditioning) inside the encoder to explicitly account for the 5 distinct acquisition domains during fine-tuning, rather than treating cross-dataset shift as purely a generalization problem.
3. **OCT-specific SSL augmentation.** Swap MAE's random masking for augmentations informed by OCT physics (speckle noise simulation, layer-aware intensity jitter) and compare against your MAE baseline — a smaller, cheaper experiment than a full new architecture, and a natural ablation on top of what's already built.

Option 1 is the most defensible as a contribution given your actual dataset structure — it directly addresses the problem the original papers never had to face.

---

## 7. End-of-Project Deliverables

- Harmonized, patient-split, 4-source dataset with documented label taxonomy
- Baseline table (ResNet-50, ViT-B) — per-dataset and fused, full cross-eval grid
- SSL (MAE + SwinV2) table — same grid, with significance tests vs. baseline
- Error analysis (confusion matrices, embeddings, per-class breakdowns)
- (If time allows) new architecture results, scoped as a focused ablation on top of the SSL pipeline
