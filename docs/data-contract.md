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

## Taxonomy

- `normal` maps from Duke `NORMAL`, Kermany `NORMAL`, and OCTDL `NO`.
- `amd` maps from Duke `AMD`, Kermany `CNV` and `DRUSEN`, and OCTDL `AMD`.
- `dme` maps directly from all current sources.
- OCTID and Paima/NEH must declare only `normal` and `amd` when added. Their absence of DME is structural, not a negative DME label.

## Split Rules

Splits must be created at the `source:group_id` level. A group may appear in one split only. Kermany filenames follow `CLASS-GROUP_ID-BSCAN_INDEX.jpeg`; the adapter extracts the middle token as the grouping key. The release copy has no sidecar metadata files, so demographic, eye, and acquisition metadata are unavailable. `oct-classify validate-splits` verifies that no extracted group crosses its supplied train/test split.

## Commands

Run from the repository root:

```bash
uv run oct-classify audit
uv run oct-classify manifest
uv run oct-classify validate-splits
```

The manifest command writes ignored JSONL files to `artifacts/manifests/` by default.
