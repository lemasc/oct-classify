# OCT Classification

Multi-dataset retinal OCT classification across Normal, AMD, and DME.

## Dataset Preparation

Dataset roots are configured in `configs/datasets.toml` and must remain `datasets/...` symlink paths. The
active configuration covers Duke, Kermany, OCTDL, and Paima; OCTID is disabled pending a defensible
group-safe split.

```bash
uv run oct-classify audit
uv run oct-classify manifest
uv run oct-classify validate-splits
uv run oct-classify splits
```

`audit` writes per-source reports and an aggregate summary to `artifacts/audits/`. It inventories
formats, dimensions, intensity distributions, manifest coverage, and exact/perceptual-hash duplicate
findings. Use `--no-perceptual-hashes` for a quicker inventory-only run, or adjust the near-duplicate
threshold with `--max-hash-distance`. `--hash-timing-log path.tsv` records each pHash duration.
`manifest` applies the locked quarantine and same-label exact-deduplication decisions before writing
derived JSONL manifests. `splits` creates ignored, per-image manifests in `artifacts/splits/` and creates
the tracked compact assignment in `configs/splits/v1.json`. Later runs verify the input hashes and reuse
that definition; use `--replace-definition` only when intentionally versioning a replacement split.

Open the interactive audit-result browser with:

```bash
uv run marimo edit notebooks/audit_results.py
```

It summarizes the generated audit reports and displays the source images in each selected exact or
pHash duplicate cluster.

Generated manifests and reports belong under ignored `artifacts/`. See `docs/data-contract.md` for the unified-label and patient/volume split contract.
