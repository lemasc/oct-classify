#!/usr/bin/env bash
#SBATCH --job-name=oct-loso
#SBATCH --array=0-3
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# One array task per held-out source. Trains fused on the other three (5
# Duke CV outer folds each, when Duke is among them; a single split
# otherwise), then evaluates the resulting checkpoint(s) against the
# held-out source's FULL manifest (train+val+test combined via
# `train evaluate --full-source`), since none of that source's data was
# used for training. The other sources' splits are fixed copies already
# baked into each Duke CV fold-N directory, so a single top-level
# artifacts/splits/<source>.jsonl is enough to evaluate the held-out source.
#
# Uses resnet50-loso.toml (batch_size=12) instead of the default
# resnet50.toml (batch_size=16): `train fused` requires data.batch_size to
# divide evenly across the number of --sources, and LOSO always trains on
# exactly 3 of the 4 sources.
sources=(duke kermany octdl paima)
held_out="${sources[$SLURM_ARRAY_TASK_ID]}"
train_sources=()
for source_name in "${sources[@]}"; do
    [[ "$source_name" == "$held_out" ]] || train_sources+=("$source_name")
done

repo_root="$(git rev-parse --show-toplevel)"
run_name="loso-${held_out}-${SLURM_ARRAY_JOB_ID}"
log_file="artifacts/slurm/loso-${held_out}-${SLURM_ARRAY_JOB_ID}.out"

cd "$repo_root"
mkdir -p artifacts/slurm

if [[ " ${train_sources[*]} " == *" duke "* ]]; then
    split_dir="artifacts/splits/duke-cv"
    folds=5
else
    split_dir="artifacts/splits"
    folds=1
fi

uv run oct-classify train fused \
    --sources "${train_sources[@]}" \
    --split-dir "$split_dir" \
    --folds "$folds" \
    --training-config configs/training/resnet50-loso.toml \
    --run-name "$run_name" \
    --device cuda \
    >"$log_file" \
    2>&1

if [[ "$folds" -eq 1 ]]; then
    checkpoints=("artifacts/runs/fused/resnet50/${run_name}/checkpoint-best.pt")
else
    checkpoints=()
    for fold in 1 2 3 4 5; do
        checkpoints+=("artifacts/runs/fused/resnet50/${run_name}-fold${fold}/checkpoint-best.pt")
    done
fi

for checkpoint in "${checkpoints[@]}"; do
    eval_tag="$(basename "$(dirname "$checkpoint")")"
    uv run oct-classify train evaluate \
        --checkpoint "$checkpoint" \
        --source "$held_out" \
        --split-dir artifacts/splits \
        --full-source \
        --output "artifacts/runs/fused/resnet50/${eval_tag}/evaluations/${held_out}-loso" \
        --device cuda \
        >>"$log_file" \
        2>&1
done
