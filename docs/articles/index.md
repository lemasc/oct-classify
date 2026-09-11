# Retinal OCT Article Index

This directory holds JATS/NLM XML exports of two related, published studies on self-supervised retinal OCT classification. Treat the XML as the authoritative source for experimental claims, tables, and citations.

## Quick Navigation

| Article | Scope | Primary model | Key outcome |
| --- | --- | --- | --- |
| [OCT-SelfNet (2025)](OCT-SelfNet.xml) | Binary Normal vs. AMD classification | MAE pre-training with ViT, Swin, or SwinV2 encoders; emphasis on SwinV2 | Establishes the two-phase, multi-source approach and evaluates cross-dataset generalization. |
| [Multi-OCT-SelfNet (2026)](Multi-OCT-SelfNet.xml) | Multi-class retinal disease classification | SwinV2 MAE, including a larger variant in pre-training experiments | Extends the earlier work to multi-class tasks and reports stronger cross-domain and limited-data performance. |

Each article is available in two complementary formats: XML is the source for structured text, metadata, tables, and references; the paired EPUB contains the rendered article and embedded figures.

## Relationship Between Articles

`Multi-OCT-SelfNet.xml` is the follow-on to `OCT-SelfNet.xml`, not an independent method. Both use the same high-level pipeline:

1. Fuse training and validation images from DS1, DS2, and DS3 for label-free masked-autoencoder (MAE) pre-training.
2. Transfer the encoder to a classifier and train/fine-tune per source dataset.
3. Evaluate both on the source test set and on the other datasets' test sets to measure domain generalization.

The material change is task scope: the 2025 article limits supervised classification to Normal versus AMD; the 2026 article retains multiple disease categories available in each dataset. Do not directly compare their metrics as if they are the same classification problem.

## Article Records

### OCT-SelfNet (2025)

- **File:** [OCT-SelfNet.xml](OCT-SelfNet.xml)
- **Figures:** [OCT-SelfNet.epub](OCT-SelfNet.epub) includes 15 figures and the Algorithm 1 image.
- **Citation:** Jannat et al., *Frontiers in Big Data*, 2025, DOI [10.3389/fdata.2025.1609124](https://doi.org/10.3389/fdata.2025.1609124)
- **Question:** Can MAE-based self-supervised learning and multi-source fusion improve generalization for binary OCT classification?
- **Target labels:** Normal and AMD. DS1 Drusen images are treated as AMD; other categories are excluded for supervised training.
- **Encoders/baselines:** OCT-SelfNet variants with ViT, Swin, and SwinV2; ResNet-50 and ViT-Base baselines.
- **Implementation reported:** 224 x 224 images, 70% random masking, 50 SSL epochs, 100 supervised epochs with early stopping; PyTorch 1.12.1, CUDA 11.2, Python 3.10.9.
- **Primary claims:** SwinV2-based OCT-SelfNet generally surpasses baselines, especially in cross-dataset and limited-label settings; both data fusion and SSL pre-training materially contribute in ablations.
- **Important limitations:** Data-fusion bias, binary-only scope, and need for human review of low-confidence clinical predictions.

Useful sections: [methodology](OCT-SelfNet.xml#s3), [datasets](OCT-SelfNet.xml#s4), [experiments/results](OCT-SelfNet.xml#s5), [summary](OCT-SelfNet.xml#s6), [conclusion](OCT-SelfNet.xml#s7), and [data availability](OCT-SelfNet.xml#s8).

### Multi-OCT-SelfNet (2026)

- **File:** [Multi-OCT-SelfNet.xml](Multi-OCT-SelfNet.xml)
- **Figures:** [Multi-OCT-SelfNet.epub](Multi-OCT-SelfNet.epub) includes all 9 figures.
- **Citation:** Jannat et al., *Frontiers in Systems Biology*, 2026, DOI [10.3389/fsysb.2026.1717398](https://doi.org/10.3389/fsysb.2026.1717398)
- **Question:** Does the multi-source SSL pipeline improve multi-class retinal disease classification and transfer to unseen datasets?
- **Target labels:** Dataset-specific combinations of Normal, AMD/Drusen, DME, CNV, and DR. DS3 excludes RVO, CSC, and heterogeneous `OTHERS` categories.
- **Encoders/baselines:** Multi-OCT-SelfNet with SwinV2; pre-training comparisons include Swin and SwinV2-large. Baselines are ResNet-50 and conventional SwinV2 without the proposed SSL multi-source pre-training.
- **Implementation reported:** 224 x 224 images, 70% random masking, 100 SSL epochs, 100 fine-tuning epochs with early stopping; unified ImageNet normalization; PyTorch 1.12.1, CUDA 11.2, Python 3.10.9.
- **Headline results:** On-domain AUC-ROC is 0.97 (DS1), 0.97 (DS2), and 0.89 (DS3). Cross-domain examples include DS2-to-DS3 AUC-ROC of 0.90 and DS3-to-DS2 of 0.94 for Multi-OCT-SelfNet-SwinV2.
- **Interpretation caveat:** Classes and class frequencies differ substantially across datasets, so cross-dataset scores should be read as robustness evidence rather than performance on a fully harmonized label space.
- **Future work:** Improve interpretability, introduce human-in-the-loop review, and address label quality and fusion-induced bias.

Useful sections: [methodology](Multi-OCT-SelfNet.xml#s3), [datasets](Multi-OCT-SelfNet.xml#s3-6), [implementation](Multi-OCT-SelfNet.xml#s4-1), [results](Multi-OCT-SelfNet.xml#s4-4), [future work](Multi-OCT-SelfNet.xml#s5), [conclusion](Multi-OCT-SelfNet.xml#s6), and [data availability](Multi-OCT-SelfNet.xml#s7).

## Dataset Map

| ID | Source and acquisition | Notes relevant to both papers |
| --- | --- | --- |
| DS1 | Kermany et al. 2018; Heidelberg Spectralis OCT | 109,559 original images; duplicates removed to yield 101,565. Split 80/10/10. Drusen is mapped to AMD. |
| DS2 | Srinivasan et al. 2014; Heidelberg Spectralis SD-OCT | 45 subjects: 15 each Normal, dry AMD, and DME. Subject-level split: 10 train, 2 validation, 3 test per class. |
| DS3 | Li et al. 2020 / OCTA-500; RTVue-XR spectral-domain OCT | 500 subjects; foveal slices are used. Multi-class work retains Normal, AMD, CNV, and DR, while removing sparse or heterogeneous classes. |

Public-source links are recorded in each article's data-availability section. Repository datasets should be referenced through the project's `datasets/` symlink paths, not their resolved locations.

## Agent Use Notes

- Use the 2025 article for binary Normal-vs-AMD design, ablations, and its ViT/Swin/SwinV2 comparison.
- Use the 2026 article for the current multi-class framing, reported SwinV2 settings, and headline quantitative results.
- Confirm numeric claims against the XML tables and captions before propagating them into code, documentation, or reports.
- Use the paired EPUB when a task requires visual inspection of figures; its `OPS/images/` archive entries contain the figure files.
