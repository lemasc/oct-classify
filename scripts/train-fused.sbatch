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
#
# Optional environment (pass with `sbatch --export=ALL,NAME=value`):
#   TRAINING_CONFIG  training config (default: the CLI default, configs/training/resnet50.toml)
#   RUN_PREFIX       run-name prefix (default: slurm); runs are <prefix>-<array job id>-fold<N>
#   LOG_DIR          directory for task logs (default: artifacts/slurm)
fold=$((SLURM_ARRAY_TASK_ID + 1))
repo_root="$(git rev-parse --show-toplevel)"
log_dir="${LOG_DIR:-artifacts/slurm}"

cd "$repo_root"
mkdir -p "$log_dir"

config_args=()
[[ -z "${TRAINING_CONFIG:-}" ]] || config_args=(--training-config "$TRAINING_CONFIG")

{
    echo "commit=$(git rev-parse --short HEAD) node=$(hostname) config=${TRAINING_CONFIG:-default} fold=${fold}"
} >"${log_dir}/fused-fold${fold}-${SLURM_ARRAY_JOB_ID}.out"

exec .venv/bin/oct-classify train fused \
    --split-dir "artifacts/splits/duke-cv/fold-${fold}" \
    --run-name "${RUN_PREFIX:-slurm}-${SLURM_ARRAY_JOB_ID}-fold${fold}" \
    "${config_args[@]}" \
    --device cuda \
    >>"${log_dir}/fused-fold${fold}-${SLURM_ARRAY_JOB_ID}.out" \
    2>&1
