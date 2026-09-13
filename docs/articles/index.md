# Retinal OCT Research Index

This directory contains the primary research context for retinal OCT classification, datasets, and cross-dataset generalization. Start with the two SelfNet papers for the project method, then use the dataset papers to resolve dataset provenance and label meaning.

## Quick Navigation

| Need | Read |
| --- | --- |
| Current method, multiclass task, latest results | [Multi-OCT-SelfNet](#multi-oct-selfnet-2026) |
| Original method and binary Normal-vs-AMD protocol | [OCT-SelfNet](#oct-selfnet-2025) |
| Source and conventions for DS1 / Kermany | [Kermany](#kermany-2018) |
| OCTDL dataset composition and label ontology | [OCTDL](#octdl-2024) |
| Dataset aliases, splits, and task differences | [Dataset Map](#dataset-map) |

## Reading Files

Use the XML files as the canonical, structured full-text sources for the SelfNet papers. Their EPUB counterparts are alternate exports of the same articles. The HTML files are body-only exports, so their bibliographic metadata is incomplete.

| Article key | Canonical local source | Alternate export |
| --- | --- | --- |
| Multi-OCT-SelfNet | `Multi-OCT-SelfNet.xml` | `Multi-OCT-SelfNet.epub` |
| OCT-SelfNet | `OCT-SelfNet.xml` | `OCT-SelfNet.epub` |
| Kermany | `Kermany.html` | None |
| OCTDL | `OCTDL.html` | None |

## Papers

### Multi-OCT-SelfNet (2026)

**Jannat et al.** *Multi-OCT-SelfNet: integrating self-supervised learning with multi-source data fusion for enhanced multi-class retinal disease classification.* Frontiers in Systems Biology, 2026. DOI: `10.3389/fsysb.2026.1717398`.

- **Purpose:** Current and most complete framework for robust multiclass retinal-disease classification across domains.
- **Method:** Fuse unlabeled images from three sources for masked-autoencoder self-supervised pre-training with a SwinV2 encoder; replace the decoder with a classifier and fine-tune per downstream dataset.
- **Evaluation:** On-domain, cross-dataset, 50%-data, data-fusion, and SSL ablations. Primary metrics are AUC-ROC, AUC-PR, accuracy, and F1; AUC-ROC is emphasized because of imbalance.
- **Central claim:** Multi-source fusion and SSL improve transferability most strongly for the smaller DS2 and DS3 datasets. For example, DS2-to-DS3 AUC-ROC is 0.90 versus 0.59 for ResNet-50 and 0.61 for conventional SwinV2.
- **Scope warning:** Pre-training uses all available disease categories, while downstream fine-tuning evaluates Normal against disease classes present in each dataset. Do not assume the earlier binary-only task.
- **Where to look:** `Methodology`, `Datasets`, `Results`, and the data-fusion / SSL / limited-data ablations.

### OCT-SelfNet (2025)

**Jannat et al.** *OCT-SelfNet: a self-supervised framework with multi-source datasets for generalized retinal disease detection.* Frontiers in Big Data, 2025. DOI: `10.3389/fdata.2025.1609124`.

- **Purpose:** Predecessor framework and source for the original experimental protocol.
- **Method:** Multi-source unlabeled SSL pre-training with a SwinV2 masked autoencoder, followed by supervised fine-tuning; compares against ResNet-50 and ViT-based models.
- **Task:** Binary **Normal vs. AMD** classification only. DS1 Drusen images are treated as AMD; other source classes are removed from downstream training and evaluation.
- **Evidence:** Cross-dataset evaluation, no-SSL, no-data-fusion, unseen-dataset, and reduced-data experiments show the strongest benefit on DS2 and DS3.
- **Where to look:** Sections `3 Methodology`, `4 Datasets`, and `5.4` for the ablations; section `6 Summary` and `7 Conclusion` for interpretation.

### Kermany (2018)

**Kermany et al.** *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell, 2018. DOI: `10.1016/j.cell.2018.02.010`.

- **Purpose:** Foundational large OCT classification dataset and the source denoted **DS1** in both SelfNet papers.
- **Dataset:** 109,559 Spectralis OCT images in four labels: Normal, CNV, DME, and Drusen. The SelfNet data-cleaning procedure reports 101,565 images after duplicate removal.
- **Project convention:** Map DS1 `Drusen` to AMD only when reproducing the binary OCT-SelfNet task. Retain its original label for the multiclass Multi-OCT-SelfNet context unless the implementation specifies otherwise.
- **Where to look:** `Results` for the clinical deep-learning setting and `STAR Methods` / `Transfer Learning Methods` for original data and model details.

### OCTDL (2024)

**OCTDL dataset paper.** Scientific Data, 2024. DOI: `10.1038/s41597-024-03182-7`.

- **Purpose:** Dataset reference for image-level retinal pathology labels and dataset-combination baselines; it is not one of the DS1--DS3 sources in the SelfNet papers.
- **Dataset:** More than 2,000 macular raster-scan OCT images covering AMD, DME, ERM, RAO, RVO, and vitreomacular interface disease, with pathology labels including MNV, DRIL, drusen, macular edema, and macular hole.
- **Use carefully:** Its experimental combinations with OCTID and Kermany use their own splits and objectives. Do not merge its results with SelfNet results without reconciling labels, data sources, and split policy.
- **Where to look:** `Methods`, `Data Records`, and `Technical Validation`.

## Dataset Map

| Alias | Source | Acquisition / scale | Original classes relevant here | SelfNet handling |
| --- | --- | --- | --- | --- |
| DS1 | Kermany 2018 | Spectralis; 109,559 images before duplicate cleanup | Normal, CNV, DME, Drusen | Drusen relabeled AMD for the 2025 binary task; 80/10/10 image split |
| DS2 | Srinivasan et al. 2014 | Spectralis SD-OCT; 45 subjects | Normal, dry AMD, DME | Subject-level split: 10/2/3 subjects per class for train/validation/test |
| DS3 | OCTA-500 / Li et al. 2020 | RTVue-XR; 500 subjects; foveal B-scans used | Includes Normal, AMD, DR, CNV | Stratified 80/10/10 split; a small, domain-shift-sensitive setting |
| OCTDL | OCTDL 2024 | More than 2,000 macular raster scans | Multiple disease and pathology labels | Separate dataset paper; not DS1, DS2, or DS3 |

## Context Resolution Rules

- Treat **OCT-SelfNet** and **Multi-OCT-SelfNet** as related but non-interchangeable experiments: the former is binary Normal-vs-AMD; the latter is multiclass.
- Treat **DS1**, **DS2**, and **DS3** as project aliases only. Resolve them to Kermany, Srinivasan, and OCTA-500 respectively before comparing code, labels, or counts.
- Do not compare raw accuracy across datasets without considering class imbalance and domain shift. The SelfNet papers prioritize AUC-ROC, AUC-PR, and F1 for this reason.
- Distinguish a model's **pre-training dataset mixture** from its **fine-tuning dataset** and its **cross-dataset test set**. These are deliberately different in the generalization experiments.
- Use reported results as paper-specific evidence, not universal baselines: preprocessing, deduplication, class mapping, foveal-slice selection, and split policy materially affect them.
