"""Post-training analysis commands. Only `gradcam` imports torch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from oct_classify.analysis.loading import discover_prediction_sets, load_duplicate_paths

DEFAULT_SEED = 20260927


def _analysis_dir(run_dir: Path, output: Path | None) -> Path:
    return output / run_dir.name if output is not None else run_dir / "analysis"


def _format_interval(interval: dict[str, object]) -> str:
    point, low, high = interval["point"], interval["ci_low"], interval["ci_high"]
    if point is None or low is None or high is None:
        return "n/a"
    return f"{100 * float(point):.2f} [{100 * float(low):.2f}, {100 * float(high):.2f}]"


def _run(args: argparse.Namespace) -> None:
    from oct_classify.analysis.report import analyze_prediction_set

    duplicates: dict[str, set[str]] = {}
    for run_dir in args.run_dirs:
        for prediction_set in discover_prediction_sets(run_dir, args.manifests):
            if args.sets and prediction_set.name not in args.sets:
                continue
            if prediction_set.source not in duplicates:
                duplicates[prediction_set.source] = load_duplicate_paths(
                    args.audits, prediction_set.source
                )
            output = _analysis_dir(run_dir, args.output) / prediction_set.name
            summary = analyze_prediction_set(
                prediction_set,
                output,
                duplicates[prediction_set.source],
                n_boot=args.n_boot,
                n_boot_auroc=args.n_boot_auroc,
                seed=args.seed,
                gallery_size=args.gallery_size,
                max_per_group=args.max_per_group,
            )
            check = summary["reference_check"]
            status = "" if check is None or check["matches"] else "  WARNING: differs from saved metrics"
            metrics = summary["bootstrap"]["metrics"]  # type: ignore[index]
            print(
                f"{run_dir.name}/{prediction_set.name}: n={summary['images']} "
                f"groups={summary['groups']} macro-F1 {_format_interval(metrics['macro_f1'])} "
                f"ECE {100 * summary['calibration']['ece']:.2f}{status}"  # type: ignore[index]
            )


def _campaign(args: argparse.Namespace) -> None:
    from oct_classify.analysis.campaign import (
        aggregate_set,
        campaign_table,
        default_output,
        group_sets_by_name,
    )
    from oct_classify.analysis.report import json_safe, write_rows
    from oct_classify.training.outputs import write_json

    grouped = group_sets_by_name(
        [discover_prediction_sets(run_dir, args.manifests) for run_dir in args.run_dirs]
    )
    results = {
        name: aggregate_set(sets, n_boot=args.n_boot, n_boot_auroc=args.n_boot_auroc, seed=args.seed)
        for name, sets in sorted(grouped.items())
        if not args.sets or name in args.sets
    }
    output = args.output or default_output(args.name)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "summary.json",
        json_safe(
            {"campaign": args.name, "run_dirs": [str(path) for path in args.run_dirs], "sets": results}
        ),
    )
    table = campaign_table(args.name, results)
    write_rows(output / "table.csv", [json_safe(row) for row in table])  # type: ignore[misc]
    for row in table:
        if row["metric"] == "macro_f1":
            print(
                f"{row['set']} [{row['level']}, {row['mode']}, runs={row['runs']}, "
                f"units={row['units']}]: macro-F1 {_format_interval(row)}"
            )
    for name, result in results.items():
        if result["mode"] in ("overlapping", "skipped"):
            print(f"{name}: not aggregated ({result['reason']})")
    print(f"Wrote campaign analysis to {output}")


def _gradcam(args: argparse.Namespace) -> None:
    import torch

    from oct_classify.analysis.gradcam import run_gradcam
    from oct_classify.data.config import load_dataset_specs
    from oct_classify.training.outputs import write_json

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    roots = {spec.name: spec.root for spec in load_dataset_specs(args.config)}
    analysis_dir = _analysis_dir(args.run_dir, args.output)
    set_dirs = sorted(path.parent for path in analysis_dir.glob("*/errors.csv"))
    if args.sets:
        set_dirs = [path for path in set_dirs if path.name in args.sets]
    if not set_dirs:
        raise FileNotFoundError(f"No analyzed sets under {analysis_dir}; run `analyze run` first.")
    checkpoint = args.checkpoint or args.run_dir / "checkpoint-best.pt"
    for set_dir in set_dirs:
        summary = run_gradcam(
            set_dir,
            checkpoint,
            roots,
            device,
            layers=args.layers,
            batch_size=args.batch_size,
            randomization_sample=args.randomization_sample,
            seed=args.seed,
        )
        write_json(set_dir / "gradcam" / "summary.json", summary)
        print(f"{set_dir.name}: Grad-CAM for {summary['rows']} images")
        for row in summary["attention"]:  # type: ignore[union-attr]
            if ":" not in str(row["group"]) and row["target_role"] == "predicted":
                mean = row["outside_retina_mass_mean"]
                print(
                    f"  {row['group']:<8} {row['layer']}: outside-retina mass "
                    f"{'n/a' if mean is None else f'{100 * mean:.1f}%'} (n={row['n']})"
                )
        spearman = summary["randomization_check"].get("spearman_by_layer", {})  # type: ignore[union-attr]
        for layer, stats in spearman.items():
            if stats["median"] is not None:
                print(f"  randomization Spearman {layer}: median {stats['median']:.2f}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifests", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--sets", nargs="+", help="Only these set names, e.g. test-duke eval-paima-loso.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-boot-auroc", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Post-training analysis of saved runs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Bootstrap CIs, calibration, curves, slices, and galleries for each run."
    )
    run.add_argument("run_dirs", type=Path, nargs="+")
    _add_common(run)
    run.add_argument("--audits", type=Path, default=Path("artifacts/audits"))
    run.add_argument("--gallery-size", type=int, default=24, help="Images per confusion cell.")
    run.add_argument("--max-per-group", type=int, default=3, help="Gallery images per patient.")
    run.add_argument(
        "--output", type=Path, help="Write to <output>/<run name>/ instead of <run>/analysis/."
    )
    run.set_defaults(handler=_run)

    campaign = subparsers.add_parser(
        "campaign", help="Aggregate matching sets across the checkpoints of a campaign."
    )
    campaign.add_argument("run_dirs", type=Path, nargs="+")
    campaign.add_argument("--name", required=True)
    _add_common(campaign)
    campaign.add_argument("--output", type=Path, help="Defaults to data/analysis/<name>/.")
    campaign.set_defaults(handler=_campaign)

    gradcam = subparsers.add_parser(
        "gradcam", help="Grad-CAM and off-retina attention for gallery images (needs `run`)."
    )
    gradcam.add_argument("run_dir", type=Path)
    gradcam.add_argument("--config", type=Path, default=Path("configs/datasets.toml"))
    gradcam.add_argument("--checkpoint", type=Path, help="Defaults to <run>/checkpoint-best.pt.")
    gradcam.add_argument("--sets", nargs="+")
    gradcam.add_argument("--layers", nargs="+", default=["layer3", "layer4"])
    gradcam.add_argument("--device", default="cuda")
    gradcam.add_argument("--batch-size", type=int, default=16)
    gradcam.add_argument("--randomization-sample", type=int, default=32)
    gradcam.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gradcam.add_argument(
        "--output", type=Path, help="Analysis root used by `run --output`, if any."
    )
    gradcam.set_defaults(handler=_gradcam)

    args = parser.parse_args(argv)
    args.handler(args)
