# OCT Classification

Multi-dataset retinal OCT classification across Normal, AMD, and DME.

## Dataset Preparation

Dataset roots are configured in `configs/datasets.toml` and must remain `datasets/...` symlink paths. The current configuration covers Duke, Kermany, and OCTDL.

```bash
uv run oct-classify audit
uv run oct-classify manifest
uv run oct-classify validate-splits
```

Generated manifests and reports belong under ignored `artifacts/`. See `docs/data-contract.md` for the unified-label and patient/volume split contract.
