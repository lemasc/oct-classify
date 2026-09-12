# Data Audit Log

## 2026-09-12: Configured Source Snapshot

**Scope:** Duke, Kermany, and OCTDL from `configs/datasets.toml`. This is a snapshot of the
repository's configured `datasets/...` symlinks, not a substituted or third-party cleaned release.

**Command:**

```bash
uv run oct-classify audit --hash-timing-log artifacts/audits/phash-timings-with-cross-source.tsv
```

The command audited 114,250 manifest records. All decoded successfully. It exited with status 1
because the existing per-source audit identifies label conflicts and Kermany supplied-split pHash
candidates; see the ignored reports under `artifacts/audits/` for the full record-level output.

### Per-Source Results

| Source | Manifest images | Invalid | Unmanifested | Exact clusters | pHash clusters | Integrity result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Duke | 3,231 | 0 | 0 | 0 | 795 | 4 cross-label pHash candidates |
| Kermany | 109,309 | 0 | 0 | 7,092 | 38,253 | Exact and pHash cross-label candidates; pHash split candidates |
| OCTDL | 1,710 | 0 | 354 ERM files | 1 | 134 | 2 cross-label pHash candidates |

All 354 unmanifested OCTDL files are ERM images and are intentionally excluded under the locked
Normal/AMD/DME task policy. The exact and pHash cluster counts are candidate counts, not independent
image counts: a file may participate in multiple pHash pairs.

### Cross-Source Leakage Result

- Exact-content duplicate clusters: 0.
- pHash-distance-zero clusters: 0.
- pHash-distance-two candidate clusters: 6.
- pHash-distance-four candidate clusters: 165.

The distance-two and distance-four candidates remain reported observations, not automatic exclusions.
No cross-source records enter quarantine from this audit run.

### Kermany Label And Duplicate Evidence

The locally extracted Kermany article states that each retained OCT image was independently graded by
ophthalmologists and its final label verified by senior retinal specialists
(`docs/articles/Kermany.html`, Image Labeling). It also states that the original test partition used
patients independent of those in training (`docs/articles/Kermany.html`, Transfer Learning Methods).
Accordingly, exact cross-label copies are treated as incompatible with the intended image-level labels,
while perceptual matches require review rather than automatic relabeling.

The related project article records that the 109,559-image DS1 release contained known identical images
and that an external cleaning procedure yielded 101,565 images (`docs/articles/OCT-SelfNet.xml`). The
configured local Kermany source contains 109,309 manifest records, so it is processed as its own
versioned source snapshot rather than assumed to be that cleaned derivative.

### Locked Decisions

The operative processing decisions are maintained in [the data contract](data-contract.md#locked-processing-decisions):
use the configured source snapshots, select lexical canonical paths, preserve Kermany's supplied test
partition, exclude OCTDL ERM from this task, and quarantine only the defined exact/pHash-zero conflict
components.
