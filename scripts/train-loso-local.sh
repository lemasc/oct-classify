#!/usr/bin/env bash
# Local sequential leave-one-source-out (LOSO) campaign, for running on a
# single desktop GPU (e.g. RTX 3060) instead of the SLURM cluster.
#
# For each held-out source: train fused on the other three, then evaluate
# the resulting checkpoint(s) against the held-out source's FULL manifest
# (train+val+test combined via `train evaluate --full-source`), since none
# of that source's data was used for training. When Duke is one of the
# three training sources, its 5 CV outer folds are each trained separately
# (`oct-classify data duke-cv`), giving 5 checkpoints for that held-out
# source instead of one; the other sources' splits are fixed copies already
# baked into each fold-N directory, so a single top-level
# artifacts/splits/<source>.jsonl is enough to evaluate the held-out source.
#
# Uses resnet50-loso.toml (batch_size=12) instead of the default
# resnet50.toml (batch_size=16): `train fused` requires data.batch_size to
# divide evenly across the number of --sources, and LOSO always trains on
# exactly 3 of the 4 sources.

set -euo pipefail

sources=(duke kermany octdl paima)
repo_root="$(git rev-parse --show-toplevel)"
run_id="local-$(date +%Y%m%d-%H%M%S)"

cd "$repo_root"
mkdir -p artifacts/local

for held_out in "${sources[@]}"; do
    train_sources=()
    for source_name in "${sources[@]}"; do
        [[ "$source_name" == "$held_out" ]] || train_sources+=("$source_name")
    done

    run_name="loso-${held_out}-${run_id}"
    log_file="artifacts/local/loso-${held_out}-${run_id}.out"

    if [[ " ${train_sources[*]} " == *" duke "* ]]; then
        split_dir="artifacts/splits/duke-cv"
        folds=5
    else
        split_dir="artifacts/splits"
        folds=1
    fi

    echo "=== LOSO holding out ${held_out}; training on ${train_sources[*]} (${run_id}) ==="
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
        echo "=== Evaluating ${eval_tag} on held-out ${held_out} (full source) ==="
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
done
