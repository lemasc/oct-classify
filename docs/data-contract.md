# Data Contract

Every source adapter emits `ImageRecord` entries and no downstream component may rely on a source-specific layout.

| Field | Meaning |
|---|---|
| `path` | Image path relative to the configured `datasets/...` root |
| `source` | Configured source name |
| `raw_label` | Source-native class label |
| `label` | Unified `normal`, `amd`, or `dme` label |
| `available_labels` | Classes the source screened for; used by partial-label losses and metrics |
| `group_id` | Patient or volume identity within a source, if known |
| `supplied_split` | Upstream split name, retained only as provenance |
| `split` | Derived experiment split, assigned only after manifest creation |

## Taxonomy

- `normal` maps from Duke `NORMAL`, Kermany `NORMAL`, and OCTDL `NO`.
- `amd` maps from Duke `AMD`, Kermany `CNV` and `DRUSEN`, and OCTDL `AMD`.
- `dme` maps directly from all current sources.
- OCTID and Paima/NEH must declare only `normal` and `amd` when added. Their absence of DME is structural, not a negative DME label.

## Split Rules

Splits must be created at the `source:group_id` level. A group may appear in one split only. Retained
pHash-distance-zero candidates spanning groups are linked into one split-assignment unit. Kermany
filenames follow `CLASS-GROUP_ID-BSCAN_INDEX.jpeg`; the adapter extracts the middle token as the grouping
key. The release copy has no sidecar metadata files, so demographic, eye, and acquisition metadata are
unavailable. `oct-classify validate-splits` verifies that no extracted group crosses its supplied train/test
split.

The tracked `configs/splits/v1.json` is the authoritative assignment. It records the seed, source split
ratios, and hashes of the exact derived manifests and audit reports used to create it. It preserves
Kermany's supplied test partition, splitting its supplied training partition 85/15 into train/validation.
Duke, OCTDL, and Paima are split 70/15/15 into train/validation/test. Generated per-image split manifests
remain ignored under `artifacts/splits/`.

## Commands

Run from the repository root:

```bash
uv run oct-classify audit
uv run oct-classify manifest
uv run oct-classify validate-splits
uv run oct-classify splits
```

The manifest command writes ignored JSONL files to `artifacts/manifests/` by default. It applies the
locked quarantine policy and retains only the lexical canonical path for same-label exact duplicates.

## Audit Contract

`oct-classify audit` writes JSON reports under `artifacts/audits/`. It profiles decoded image format,
mode, resolution, aspect ratio, and grayscale intensity statistics. It also reports image files that
are present below a dataset root but not emitted by the source adapter.

The audit detects byte-identical images and perceptually similar images using pHashes. It reports every
detected component, including review-only candidates, and annotates each with its quarantine eligibility
and reason codes. It fails only for invalid manifest images and components eligible under the locked
policy below. Similar images within a known patient/volume group remain a reported observation rather
than a failure.

`audit` also writes `cross-source.json`, which compares all configured sources. It reports only
duplicate candidates that span source roots; pHash findings remain candidates unless the locked policy
below defines an exclusion.

## Locked Processing Decisions

- Process the repository's configured dataset snapshots rather than substituting a third-party cleaned
  release. Source images remain immutable; all exclusions and deduplication happen in derived manifests.
- When a same-label exact-duplicate component is retained once, select its canonical member by
  lexicographically smallest `source:path`.
- Preserve Kermany's supplied test partition for evaluation. Its filename-derived group identifier is
  useful for leakage checks but is not treated as verified clinical patient provenance.
- Exclude OCTDL ERM files from the Normal/AMD/DME task. Their omission is intentional and is reported
  by the audit.
- Quarantine exact duplicate components with label conflicts and pHash-distance-zero components that
  conflict in label or supplied split. Treat other pHash results, including distances 2 and 4, as review
  candidates rather than automatic exclusions.
- Deterministic preprocessing converts to RGB, composites alpha over black, preserves aspect ratio while
  resizing into a centered square canvas, applies per-image 1st/99th percentile intensity scaling, and
  uses normalization statistics calculated from training records only. Stochastic augmentation is a
  training concern and is not part of this deterministic contract.
