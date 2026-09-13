#!/usr/bin/env bash
#SBATCH --job-name=oct-resnet50
#SBATCH --array=0-3
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

sources=(duke kermany octdl paima)
source_name="${sources[$SLURM_ARRAY_TASK_ID]}"
repo_root="$(git rev-parse --show-toplevel)"

cd "$repo_root"
mkdir -p artifacts/slurm

exec uv run oct-classify train \
    --source "$source_name" \
    --run-name "slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}" \
    --device cuda \
    >"artifacts/slurm/resnet50-${source_name}-${SLURM_ARRAY_JOB_ID}.out" \
    2>&1
