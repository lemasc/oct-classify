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

## Local Baseline

The supervised baseline is ImageNet-pretrained ResNet-50 trained separately per source. Run a short,
end-to-end local flow check before a full experiment:

```bash
uv run oct-classify train --source duke --run-name smoke --epochs 1 --max-train-batches 2 --max-eval-batches 2
uv run oct-classify evaluate --checkpoint artifacts/runs/duke/resnet50/smoke/checkpoint-best.pt --source paima --max-eval-batches 2
```

`train` consumes the derived source split, calculates normalization from its training records only, and
writes configuration, normalization, checkpoints, history, metrics, and predictions under
`artifacts/runs/`. `evaluate` uses the class intersection between the checkpoint and target source, so
a three-class model is evaluated on Paima's Normal/AMD test examples only.
Metrics from capped smoke runs are flow-validation evidence only, not experimental results.

## Cluster Baseline

Submit the four per-dataset ResNet-50 runs as a SLURM job array:

```bash
sbatch scripts/train-resnet50.sbatch
```

Each array task requests one 10 GB MIG GPU slice, four CPUs, 32 GB memory, and up to 12 hours. It
trains Duke, Kermany, OCTDL, or Paima with `configs/training/resnet50.toml`; output runs are
uniquely named by their SLURM job and task IDs, and task logs are written to `artifacts/slurm/`.

Monitor the remote runs by starting TensorBoard from the repository root:

```bash
uv run tensorboard --logdir artifacts/runs
```

Training writes scalar loss, accuracy, balanced accuracy, macro-F1, and macro-AUROC to each run's
`events/` directory. Use SSH port forwarding rather than exposing TensorBoard directly to the
network.
