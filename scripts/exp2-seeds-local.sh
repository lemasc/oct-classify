#!/usr/bin/env bash
# Exp. 2 on the local RTX 3060: 8 campaigns (2 arms x 2 seeds x 2 targets),
# run one after another (~17 min per model, 5 models per campaign, ~11 h total).
# Usage: scripts/exp2-seeds-local.sh [--workers N]
# OUTPUT_ROOT sets where runs and logs go (default: artifacts).
# Safe to re-run: a campaign whose cv-summary already exists is skipped.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
output_root="${OUTPUT_ROOT:-artifacts}"
export OUTPUT_ROOT="$output_root"

for arm in fill-aware fill-aware-square-crop; do
    for seed in 20260914 20260915; do
        for target in kermany paima; do
            run_id="exp2-${arm}-s${seed}"
            if [[ -f "$output_root/runs/fused/resnet50/loso-${target}-${run_id}-cv-summary.json" ]]; then
                echo "skip ${target} ${run_id} (done)"
                continue
            fi
            TRAINING_CONFIG="configs/training/resnet50-loso-${arm}-seed${seed}.toml" \
            HELD_OUT="$target" RUN_ID="$run_id" \
                scripts/train-loso-local.sh "$@"
        done
    done
done
