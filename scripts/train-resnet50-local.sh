#!/usr/bin/env bash
# Local sequential counterpart to train-resnet50.sbatch, for running on a single
# desktop GPU (e.g. RTX 3060) instead of the SLURM cluster.

set -euo pipefail

# Ordered smallest to largest dataset (by record count), so validity issues
# surface on the cheapest run first.
sources=(octdl duke paima kermany)
repo_root="$(git rev-parse --show-toplevel)"
run_id="local-$(date +%Y%m%d-%H%M%S)"

cd "$repo_root"
mkdir -p artifacts/local

for source_name in "${sources[@]}"; do
    echo "=== Training ${source_name} (${run_id}) ==="
    uv run oct-classify train baseline \
        --source "$source_name" \
        --run-name "${run_id}-${source_name}" \
        --device cuda \
        >"artifacts/local/resnet50-${source_name}-${run_id}.out" \
        2>&1
done
