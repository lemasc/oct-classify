#!/usr/bin/env bash
# Local sequential counterpart to train-fused.sbatch, for running on a single
# desktop GPU (e.g. RTX 3060) instead of the SLURM cluster.

set -euo pipefail

# One pass per Duke outer fold; the other sources' splits are fixed copies
# already baked into each fold-N directory by `oct-classify data duke-cv`.
folds=(1 2 3 4 5)
repo_root="$(git rev-parse --show-toplevel)"
run_id="local-$(date +%Y%m%d-%H%M%S)"

cd "$repo_root"
mkdir -p artifacts/local

for fold in "${folds[@]}"; do
    echo "=== Training fused fold ${fold} (${run_id}) ==="
    uv run oct-classify train fused \
        --split-dir "artifacts/splits/duke-cv/fold-${fold}" \
        --run-name "${run_id}-fold${fold}" \
        --device cuda \
        >"artifacts/local/fused-fold${fold}-${run_id}.out" \
        2>&1
done
