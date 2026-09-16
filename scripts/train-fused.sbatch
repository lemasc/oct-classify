#!/usr/bin/env bash
#SBATCH --job-name=oct-fused
#SBATCH --array=0-4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# One array task per Duke CV outer fold; the other sources' splits are fixed
# copies already baked into each fold-N directory by `oct-classify data duke-cv`.
fold=$((SLURM_ARRAY_TASK_ID + 1))
repo_root="$(git rev-parse --show-toplevel)"

cd "$repo_root"
mkdir -p artifacts/slurm

exec uv run oct-classify train fused \
    --split-dir "artifacts/splits/duke-cv/fold-${fold}" \
    --run-name "slurm-${SLURM_ARRAY_JOB_ID}-fold${fold}" \
    --device cuda \
    >"artifacts/slurm/fused-fold${fold}-${SLURM_ARRAY_JOB_ID}.out" \
    2>&1
