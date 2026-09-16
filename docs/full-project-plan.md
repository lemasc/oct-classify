# Multi-Dataset OCT Classification — Project Plan

**Scope:** First pass: Normal / AMD (merged: AMD-unspecified + CNV + Drusen) / DME, across Duke, Kermany, OCTDL, and PAIMA. OCTID is deferred until a defensible split design is available.
**Timeline:** ~6 weeks total: one completed exploratory week and approximately five remaining weeks.
**Compute:** RTX 3060 (local) · A100 cluster (1g.10gb MIG slices, on-demand) · 7g.80gb slices (queued)

---

## 1. The Real Complexity: Missing DME Labels

After merging CNV + Drusen + AMD-unspecified into a single AMD class, your label availability looks like this:

| Class | Duke | Kermany | OCTDL | PAIMA |
|---|---|---|---|---|
| Normal | ✓ | ✓ | ✓ | ✓ |
| AMD (merged) | ✓ | ✓ (via CNV) | ✓ | ✓ (via CNV/Drusen) |
| DME | ✓ | ✓ | ✓ | **✗** |

Once merged, AMD and Normal are available everywhere — but **DME is structurally absent from PAIMA**. PAIMA images were not screened or labeled for DME, so their labels must not be treated as negative DME targets. This shapes two decisions:

- **Training a unified 3-way classifier on fused labeled data:** naively training softmax cross-entropy with PAIMA batches will push DME logits toward zero for images that were never screened for DME, which is a false negative signal, not a true one. Use a **masked loss** (only backprop the classes a given dataset actually screened for) or train **one-vs-rest binary heads** instead of a single softmax, so absent classes don't contaminate gradients.
- **Cross-dataset evaluation:** evaluate PAIMA on Normal/AMD only. Report a **full 3-class grid** for the Duke/Kermany/OCTDL subset and a separate **2-class (Normal/AMD) 4×4 grid** for all active sources. Keep these visually and numerically separate in your results tables — conflating them will make your off-domain numbers look worse than they are.

---

## 2. Experimental Design

Mirroring your papers' structure, adapted for 4 sources and partial labels:

| Stage | What | Backbone(s) |
|---|---|---|
| **Comparable ResNet-50 campaign** | ImageNet-pretrained independent and source-balanced fused training from the same revised manifests and split definitions; retain predictions for all evaluations | ResNet-50 |
| **Generalization gate** | Leave-one-source-out (LOSO) training and evaluation, with confidence intervals and source-appropriate aggregation | ResNet-50 |
| **Architecture comparison** | Repeat the fixed ResNet-50 protocol with ViT-B only after the LOSO result is stable enough to interpret | ViT-B |
| **Proposed (SSL)** | MAE pretraining on fused *unlabeled* images from all 4 active sources, then fine-tune and evaluate under the same protocol | SwinV2 (matches the papers); ViT-B only if it passes the prior gate |
| **Stretch: new architecture** | See §6 | TBD, gated on results from above |

**Data and evaluation contract:** The completed data fix is the starting point for all comparable runs: Duke uses five patient/volume-level outer folds with eye/volume-level probability aggregation; PAIMA uses its per-B-scan `Label`, retains its cohort and CSV-derived eye identity as metadata, and remains group-split by cohort-local patient identifier. Kermany preserves its supplied test partition; OCTDL and PAIMA retain 70/15/15 group splits. PAIMA distance-two pHash candidates were reviewed and retained because they were predominantly false positives; only the contract's exact and distance-zero quarantine rules alter manifests.

**Fair-comparison rule:** Any model comparison uses the same regenerated manifests, split assignments, preprocessing, training budget, and evaluation protocol. The earlier ResNet baseline and fused runs establish runtime and pipeline feasibility, but Duke results from the former 70/15/15 split are exploratory and are not tabled against the CV campaign.

---

## 3. Compute Allocation

Your bottleneck isn't total compute, it's **queue time on the 7g.80gb slices**. Plan around that explicitly rather than treating all compute as interchangeable.

| Resource | Use it for | Why |
|---|---|---|
| **RTX 3060 (local)** | ResNet-50 independent, fused, and LOSO campaigns; preprocessing; evaluation and plotting; a *tiny* pilot MAE run (low-res, 1-epoch, subsampled) to catch bugs | The completed independent and fused ResNet-50 work took one week locally, so it is the default resource for the comparable rerun |
| **1g.10gb MIG slices** | Future parallel baseline fine-tuning, linear-probe evaluation of SSL-pretrained encoders, hyperparameter sweeps, inference/eval jobs | Low memory but no queue wait — good for many small parallel jobs |
| **7g.80gb (queued)** | MAE self-supervised pretraining on the fused dataset (needs large batch size + memory), full fine-tuning of the largest runs | This is your scarce resource — debug elsewhere first, submit here only once the job is verified to run end-to-end |

**Practical rule:** never let a job's first-ever execution be on the queued 7g.80gb slice. Validate the exact training script on the 3060 or a MIG slice at reduced scale first.

---

## 4. Week-by-Week Timeline

| Week | Focus | Compute | Milestone |
|---|---|---|---|
| **1 (completed)** | Implemented and exercised independent and fused ResNet-50 training locally. Completed the data fix: Duke five-fold CV, PAIMA labeling/eye metadata, contract and audit updates, and regenerated manifests/splits. | 3060 | Feasible pipeline and versioned data contract; prior Duke 70/15/15 results are exploratory |
| **2** | Rerun the comparable ResNet-50 independent and fused campaign from the revised manifests. Use Duke CV, preserve predictions, and report source-appropriate aggregate metrics. | 3060 | Fair independent-versus-fused supervised table |
| **3** | Run four ResNet-50 LOSO experiments. Report held-out-source metrics with confidence intervals; keep 3-class Duke/Kermany/OCTDL results separate from PAIMA's Normal/AMD evaluation. | 3060 + 1g.10gb if useful | Architecture-independent generalization result |
| **4** | Complete paired confidence intervals, calibration/decision-boundary analysis, confusion matrices, and compact validation subsets where needed. Decide whether the LOSO evidence supports an architecture comparison. | 3060 + 1g.10gb | Stable ResNet interpretation and ViT go/no-go |
| **5** | If the gate passes, run ViT-B under the identical data and evaluation protocol. Otherwise validate the MAE pipeline locally and document why architecture comparison is deferred. | 3060; 7g.80gb only for a verified MAE job | Interpretable ViT comparison or SSL-ready pipeline |
| **6** | Run the scoped SSL experiment if compute and the Week-4 gate support it; otherwise consolidate figures, tables, limitations, and writeup. This is the buffer for queue delays. | 7g.80gb + 1g.10gb for SSL; 3060 for reporting | Final deliverable |

---

## 5. Risk Mitigation

- **Do not submit to SLURM until the local supervised flow is verified.** That condition is met for supervised ResNet-50. A queued SSL job remains contingent on the LOSO result showing a remaining generalization gap and a local MAE smoke test.
- **Checkpoint aggressively.** 7g.80gb jobs that get preempted or interrupted by queue churn should resume, not restart. Save encoder state every N steps.
- **Class imbalance.** AMD will likely dominate after merging (3 original classes flow into it). Use class-weighted loss or a balanced sampler on top of the masked-loss handling above — these solve different problems and you need both.
- **Leakage and label unit.** Duke CV is volume-safe and evaluated at the eye/volume unit. PAIMA is group-safe by cohort-local patient ID but remains an image-labeled source; do not interpret its Normal/AMD score as an eye-diagnosis metric. Retain reviewed distance-two PAIMA pHash matches as observations, not exclusions.

---

## 6. Stretch Goal: New Architecture Ideas

If the core schedule lands early, pick **one** of these rather than spreading the remaining time thin:

1. **Partial-label-aware head.** Replace the shared softmax classifier with a one-vs-rest (sigmoid) head per class, explicitly designed for datasets with structurally absent classes — this is the most direct extension of what your papers didn't have to solve (they were binary).
2. **Domain-conditioned normalization.** Add per-source batch/instance normalization (or FiLM-style conditioning) inside the encoder to explicitly account for the 4 distinct acquisition domains during fine-tuning, rather than treating cross-dataset shift as purely a generalization problem.
3. **OCT-specific SSL augmentation.** Swap MAE's random masking for augmentations informed by OCT physics (speckle noise simulation, layer-aware intensity jitter) and compare against your MAE baseline — a smaller, cheaper experiment than a full new architecture, and a natural ablation on top of what's already built.

Option 1 is the most defensible as a contribution given your actual dataset structure — it directly addresses the problem the original papers never had to face.

---

## 7. End-of-Project Deliverables

- Harmonized 4-source dataset with documented taxonomy, labeling unit, and versioned splits; Duke uses five-fold volume-level CV
- Comparable ResNet-50 independent and fused table, plus LOSO evaluation and confidence intervals
- ViT-B table only if it passes the ResNet LOSO gate; otherwise an explicit deferred-architecture decision
- SSL (MAE + SwinV2) table if the compute and generalization gates pass, with significance tests versus baseline
- Error analysis (confusion matrices, embeddings, per-class breakdowns)
- (If time allows) new architecture results, scoped as a focused ablation on top of the SSL pipeline
