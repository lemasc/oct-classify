# Data Audit Log

## 2026-09-13: Consolidated Source Audit And Derived Manifest

**Scope:** Duke, PAIMA, Kermany, and OCTDL are enabled in `configs/datasets.toml`. OCTID is
configured but disabled for the first training pass because it has no group identifiers for the required
group-safe split. 

This is a snapshot of the repository's configured `datasets/...` symlinks,
not a substituted or third-party cleaned release.

Source images are immutable; exclusions and deduplication are applied only
to derived manifests.

### Source-Snapshot Audit

```bash
uv run oct-classify data audit --workers 4
```

The audit decoded all 131,072 source-manifest records successfully. It exits with status 1 by design:
PAIMA and Kermany contain components eligible for derived-manifest quarantine. All exact and perceptual
duplicate candidates remain available in the ignored JSON reports under `artifacts/audits/`.

| Source | Source records | Invalid | Unmanifested | Exact clusters | pHash clusters | Quarantine-eligible exact | Quarantine-eligible pHash-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Duke | 3,231 | 0 | 0 | 0 | 795 | 0 | 0 |
| PAIMA | 16,822 | 0 | 0 | 137 | 18,065 | 3 | 32 |
| Kermany | 109,309 | 0 | 0 | 7,092 | 38,253 | 82 | 153 |
| OCTDL | 1,710 | 0 | 354 ERM files | 1 | 134 | 0 | 0 |

All 354 unmanifested OCTDL files are ERM images, intentionally excluded from the Normal/AMD/DME task.
Cluster counts are candidate counts, not independent image counts; a record may participate in multiple
pHash candidates. Nonzero-distance pHash candidates remain review observations and do not independently
fail the audit. No pHash-zero component crossed a supplied split in this run.

### Cross-Source Result

- Exact-content duplicate clusters: 0.
- pHash-distance-zero clusters: 1.
- pHash-distance-two candidate clusters: 129.
- pHash-distance-four candidate clusters: 2,457.

The distance-zero cluster is `paima:CNV/126/025_CNV.tif` and
`kermany:train/CNV/CNV-135126-18.jpeg`, both unified AMD. Manual review found that they are not the
same image, so this is a reviewed false positive. Both records are retained. Recheck this cluster's
membership in `artifacts/audits/cross-source.json` after any source-snapshot change. No cross-source
records enter quarantine from this audit.

### Derived Manifest Result

```bash
uv run oct-classify data manifest --workers 4
```

The manifest command quarantines every member of an eligible component, retains only the lexical
`source:path` canonical member of each surviving same-label exact-duplicate component, and collapses
repeated metadata paths.

| Source | Derived records | Quarantined records | Same-label exact copies removed |
| --- | ---: | ---: | ---: |
| Duke | 3,231 | 0 | 0 |
| PAIMA | 16,601 | 87 | 134 |
| Kermany | 101,217 | 321 | 7,771 |
| OCTDL | 1,709 | 0 | 1 |

Auditing the derived manifests found no invalid images or quarantine-eligible components. All four
derived manifests pass supplied-split validation: no duplicate paths, missing group identifiers, or
groups crossing supplied splits.

### Evidence And Policy

PAIMA pHash-distance-two candidates were manually reviewed after this audit. They were predominantly
false positives, so the locked policy is retained: nonzero-distance pHash results remain review
observations rather than automatic exclusions. No PAIMA records are removed on that basis.

The locally extracted Kermany article states that each retained OCT image was independently graded by
ophthalmologists and its final label verified by senior retinal specialists
(`docs/articles/Kermany.html`, Image Labeling). It also states that the original test partition used
patients independent of training (`docs/articles/Kermany.html`, Transfer Learning Methods). Exact
cross-label copies are therefore incompatible with the intended image-level labels, while perceptual
matches require review rather than automatic relabeling.

The related project article records that the 109,559-image DS1 release contained known identical images
and that an external cleaning procedure yielded 101,565 images (`docs/articles/OCT-SelfNet.xml`). The
configured local Kermany source contains 109,309 records and is processed as its own versioned snapshot.

The operative rules are maintained in [the data contract](data-contract.md#locked-processing-decisions):
use configured source snapshots; select lexical canonical paths; preserve Kermany's supplied test
partition; exclude OCTDL ERM; and quarantine only exact label-conflict or pHash-distance-zero label/split
conflict components.

### PAIMA Distance-Two Review

PAIMA train/test pHash-distance-two candidates were manually reviewed. Most were false positives, so they
remain review observations rather than automatic exclusions. No PAIMA records were removed and the tracked
split assignments remain unchanged. This supports retaining the locked distance-zero-only linkage and
quarantine policy.

## 2026-10-08: Acquisition-Artifact EDA (no manifest changes)

Per-image features at native resolution: `uv run python scripts/extract_eda_features.py --workers 12`
(writes the git-ignored `artifacts/eda/features.csv`); explored in `notebooks/eda_acquisition_artifacts.py`.

- Border-connected white fill: Duke 98%, Kermany 61%, OCTDL 0.8%, PAIMA 0%.
- Percentile normalization compresses tissue where fill exists: median tissue ceiling 0.73 (Duke), 0.75 (Kermany 512-wide), 1.0 (PAIMA, OCTDL, Kermany 768/1536-wide).
- Canvas size leaks the label: Kermany 512x512 is 74% of DME and absent elsewhere; Duke Normal is 100% 512 wide. Image geometry alone gives 0.66 (Duke) and 0.58 (Kermany) patient-grouped balanced accuracy, chance 0.33.
- Kermany's supplied test split is dominated by 1024-wide images, which are rare in train.
- Duke black edge fill: 31% of DME, 5% of AMD, 0.1% of Normal.
- PAIMA "200 µm" scale bar is bottom-left in 98.8% of images (label-neutral).
