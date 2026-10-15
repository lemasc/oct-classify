"""Seed variation of held-out Kermany and PAIMA, letterbox (fill-aware) and square-crop LOSO models (exp. 2).

Reads each run's stored full-source evaluation (`evaluations/<source>-loso/metrics.json`, 5 Duke-fold models per
seed) and writes `data/paper/seeds.json`, which `docs/paper/lib.typ` loads. Seed 20260913 runs are under
`artifacts/`; the two extra seeds stayed in the training worktree's scratchpad, passed with `--extra-root`.
Means over the 5 models, as in docs/reports/20261012-exp2-seed-noise.md; no bootstrap.
Run from the repository root: `uv run python scripts/paper/export_seeds.py`.
"""

import argparse
import json
from pathlib import Path

MAIN = Path("artifacts/runs/fused/resnet50")
OUT = Path("data/paper/seeds.json")
SOURCES = ["kermany", "paima"]
# Arm -> seed -> (root key, run name pattern).
RUNS = {
    "loso-fill": {
        "20260913": ("main", "loso-{s}-local-20261008-174938"),
        "20260914": ("extra", "loso-{s}-exp2-fill-aware-s20260914"),
        "20260915": ("extra", "loso-{s}-exp2-fill-aware-s20260915"),
    },
    "loso-sq": {
        "20260913": ("main", "loso-{s}-local-3a-r2-square-crop"),
        "20260914": ("extra", "loso-{s}-exp2-fill-aware-square-crop-s20260914"),
        "20260915": ("extra", "loso-{s}-exp2-fill-aware-square-crop-s20260915"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-root", type=Path,
                        default=Path("../repo-retry/.scratchpad/049579f_exp2-seeds/runs/fused/resnet50"))
    args = parser.parse_args()
    roots = {"main": MAIN, "extra": args.extra_root}

    out: dict = {}
    for source in SOURCES:
        out[source] = {}
        for arm, seeds in RUNS.items():
            out[source][arm] = {}
            for seed, (root, pattern) in seeds.items():
                run = pattern.format(s=source)
                paths = sorted(roots[root].glob(f"{run}-fold*/evaluations/{source}-loso/metrics.json"))
                metrics = [json.loads(p.read_text()) for p in paths]
                assert len(metrics) == 5 and all(m["full_source"] for m in metrics), (source, arm, seed, len(metrics))
                out[source][arm][seed] = {
                    name: sum(m[name] for m in metrics) / len(metrics) for name in ("macro_f1", "macro_auroc")
                }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
