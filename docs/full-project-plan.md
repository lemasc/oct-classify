# Multi-Dataset OCT Classification — Project Plan

**Scope:** Normal / AMD (merged: AMD-unspecified + CNV + Drusen) / DME, across Duke, Kermany, OCTID, OCTDL, Paima/NEH
**Timeline:** ~8 weeks
**Compute:** RTX 3060 (local) · A100 cluster (1g.10gb MIG slices, on-demand) · 7g.80gb slices (queued)

---

## 1. The Real Complexity: Missing DME Labels

After merging CNV + Drusen + AMD-unspecified into a single AMD class, your label availability looks like this:

| Class | Duke | Kermany | OCTID | OCTDL | Paima/NEH |
|---|---|---|---|---|---|
| Normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| AMD (merged) | ✓ | ✓ (via CNV) | ✓ | ✓ | ✓ (via CNV/Drusen) |
| DME | ✓ | ✓ | **✗** | ✓ | **✗** |

Once merged, AMD and Normal are available everywhere — but **DME is missing from OCTID and Paima/NEH**. This is a partial-label, not a missing-modality, problem: those datasets aren't "off-domain" for DME, they simply have no DME images at all (not even unlabeled ones you could pseudo-label). This shapes two decisions:

- **Training a unified 3-way classifier on fused labeled data:** naively training softmax cross-entropy with OCTID/Paima batches will push DME logits toward zero for images that were never screened for DME, which is a false negative signal, not a true one. Use a **masked loss** (only backprop the classes a given dataset actually screened for) or train **one-vs-rest binary heads** instead of a single softmax, so absent classes don't contaminate gradients.
- **Cross-dataset evaluation:** when evaluating a model trained on Duke against OCTID's test set, restrict the metric to Normal/AMD (OCTID literally cannot validate DME performance). Report a **full 3-class grid** only for the Duke/Kermany/OCTDL⇄Duke/Kermany/OCTDL subset, and a **2-class (Normal/AMD) grid** for the full 5×5 matrix. Keep these visually and numerically separate in your results tables — conflating them will make your off-domain numbers look worse than they are.

---

## 2. Experimental Design

Mirroring your papers' structure, adapted for 5 sources and partial labels:

| Stage | What | Backbone(s) |
|---|---|---|
| **Baseline** | ImageNet-pretrained, no SSL, fine-tuned per-dataset + once on fused labeled set | ResNet-50, ViT-B |
| **Proposed (SSL)** | MAE pretraining on fused *unlabeled* images from all 5 sources → fine-tune per dataset (masked/one-vs-rest loss) → cross-dataset eval | SwinV2 (matches your papers), consider ViT-B as a second run if time allows |
| **Stretch: new architecture** | See §6 | TBD, gated on results from above |

**Splits:** patient-level, not image-level, wherever patient/volume IDs are available (Duke, Kermany especially — B-scans from the same volume are near-duplicates and will leak across train/test if split randomly). Where no patient ID exists, dedupe with perceptual hashing before splitting.

---

## 3. Compute Allocation

Your bottleneck isn't total compute, it's **queue time on the 7g.80gb slices**. Plan around that explicitly rather than treating all compute as interchangeable.

| Resource | Use it for | Why |
|---|---|---|
| **RTX 3060 (local)** | Data pipeline dev/debugging, preprocessing, small sanity-check training runs, writing eval/plotting scripts, a *tiny* pilot MAE run (low-res, 1-epoch, subsampled) to catch bugs | Fast iteration, no queue, but not enough VRAM for full MAE pretraining at target resolution/batch size |
| **1g.10gb MIG slices** | Baseline ResNet-50/ViT-B fine-tuning (per dataset, in parallel across slices), linear-probe evaluation of SSL-pretrained encoders, hyperparameter sweeps, inference/eval jobs | Low memory but no queue wait — good for many small parallel jobs |
| **7g.80gb (queued)** | MAE self-supervised pretraining on the fused dataset (needs large batch size + memory), full fine-tuning of the largest runs | This is your scarce resource — debug elsewhere first, submit here only once the job is verified to run end-to-end |

**Practical rule:** never let a job's first-ever execution be on the queued 7g.80gb slice. Validate the exact training script on the 3060 or a MIG slice at reduced scale first.

---

## 4. Week-by-Week Timeline

| Week | Focus | Compute | Milestone |
|---|---|---|---|
| **1** | Data audit: inventory image counts/resolutions/formats per dataset; build unified label map; patient-level split design; preprocessing pipeline (resize, normalize, intensity harmonization). Build the fused unlabeled pool. **Submit a pilot (reduced) MAE job to the 7g.80gb queue as soon as the fused pool exists**, even before it's polished — you want a slot reserved and bugs surfaced early. | 3060 + 1g.10gb | Clean, harmonized, split datasets; MAE pilot job queued |
| **2** | Baseline supervised training: ResNet-50 & ViT-B (ImageNet-pretrained), fine-tuned per-dataset. Build the cross-dataset eval harness (on-domain/off-domain grid, masked-metric logic for DME-missing datasets). | 1g.10gb (parallel per-dataset jobs) | Baseline per-dataset models + first cross-eval numbers |
| **3** | Fused-supervised baseline (all 5 datasets combined, masked-loss/one-vs-rest head). Debug + relaunch full-scale MAE pretraining job on 7g.80gb once pilot confirms correctness. Monitor queue. | 1g.10gb + 7g.80gb (queued) | Fused baseline done; full MAE pretraining running |
| **4** | While MAE trains: build fine-tuning scripts (linear probe + full fine-tune variants), statistical testing utilities (bootstrap CIs), embedding visualization tooling. Buffer for MAE training/queue delays. | 3060 + 1g.10gb | Fine-tuning pipeline ready to go the moment pretraining checkpoints land |
| **5** | Fine-tune SSL-pretrained encoder per dataset (masked loss), run full cross-dataset grid (2-class 5×5 + 3-class subset). Compare against baseline. | 1g.10gb (parallel), occasional 7g.80gb for full fine-tune if linear probe underperforms | SSL vs baseline comparison table complete |
| **6** | Error analysis: confusion matrices per dataset, embedding t-SNE/UMAP, per-class AUC-ROC/PR, significance tests. **Decision point:** go/no-go on new architecture based on time remaining and where the SSL model is weak. | 1g.10gb | Results consolidated; go/no-go decision made |
| **7** | New architecture experiment (see §6) — scoped tightly to *one* idea, not several. | 7g.80gb (queued) for training, 1g.10gb for eval | New architecture results (even if preliminary) |
| **8** | Final evaluation pass, figures/tables, writeup. This week is also your buffer for any queue delays that pushed earlier weeks back. | 1g.10gb + 3060 | Final deliverable |

---

## 5. Risk Mitigation

- **Queue latency is your critical path.** The MAE pretraining run is the one thing everything downstream depends on — front-load it (Week 1 pilot, Week 3 full run) rather than scheduling it mid-project. If it slips, Weeks 5–7 slip with it.
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

- Harmonized, patient-split, 5-source dataset with documented label taxonomy
- Baseline table (ResNet-50, ViT-B) — per-dataset and fused, full cross-eval grid
- SSL (MAE + SwinV2) table — same grid, with significance tests vs. baseline
- Error analysis (confusion matrices, embeddings, per-class breakdowns)
- (If time allows) new architecture results, scoped as a focused ablation on top of the SSL pipeline