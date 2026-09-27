import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image

    from oct_classify.data.config import load_dataset_specs
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image

    repository_root = Path(__file__).resolve().parents[1]
    dataset_roots = {
        spec.name: repository_root / spec.root
        for spec in load_dataset_specs(repository_root / "configs" / "datasets.toml")
    }
    set_summaries = {
        str(path.parent.relative_to(repository_root / "artifacts" / "runs")): path
        for path in sorted((repository_root / "artifacts" / "runs").glob("*/*/*/analysis/*/summary.json"))
    }
    campaign_tables = {
        path.parent.name: path
        for path in sorted((repository_root / "data" / "analysis").glob("*/table.csv"))
    }
    return (
        Image,
        PreprocessingSpec,
        alt,
        campaign_tables,
        dataset_roots,
        io,
        json,
        mo,
        np,
        pd,
        preprocess_image,
        set_summaries,
    )


@app.cell
def _(mo):
    mo.md("""
    # OCT Post-Training Analysis

    Browses the output of `oct-classify analyze run|campaign|gradcam`. Intervals are 95% cluster
    bootstrap intervals that resample patients or volumes (`source:group_id`), and Duke also reports
    eye-level metrics from mean B-scan probabilities. Rerun the analysis commands when runs change.
    """)
    return


@app.cell
def _(campaign_tables, mo):
    campaign = mo.ui.dropdown(
        options=list(campaign_tables),
        value=next(iter(campaign_tables), None),
        label="Campaign",
    )
    campaign_metric = mo.ui.dropdown(
        options=["macro_f1", "balanced_accuracy", "accuracy", "macro_auroc"],
        value="macro_f1",
        label="Metric",
    )
    mo.hstack([campaign, campaign_metric], justify="start", gap=2)
    return campaign, campaign_metric


@app.cell
def _(campaign, campaign_metric, campaign_tables, mo, pd):
    if campaign.value is None:
        campaign_view = mo.callout(
            "No campaign tables under data/analysis/; run `oct-classify analyze campaign`.",
            kind="neutral",
        )
    else:
        table = pd.read_csv(campaign_tables[campaign.value])
        table = table[table["metric"] == campaign_metric.value]
        for column in ("point", "ci_low", "ci_high", "run_sd"):
            table[column] = (100 * table[column]).round(2)
        campaign_view = mo.vstack(
            [
                mo.md(f"### Campaign `{campaign.value}` — {campaign_metric.value} (%)"),
                mo.ui.table(
                    table[
                        ["set", "level", "mode", "runs", "rows", "units", "point", "ci_low", "ci_high", "run_sd"]
                    ],
                    selection=None,
                    pagination=False,
                ),
                mo.md(
                    "`shared`: checkpoints scored on the same rows; the interval is for mean "
                    "checkpoint performance. `pooled`: disjoint out-of-fold rows concatenated."
                ),
            ]
        )
    campaign_view
    return


@app.cell
def _(mo, set_summaries):
    selected_set = mo.ui.dropdown(
        options=list(set_summaries),
        value=next(iter(set_summaries), None),
        label="Run / prediction set",
        searchable=True,
    )
    selected_set  # noqa: B018
    return (selected_set,)


@app.cell
def _(json, mo, selected_set, set_summaries):
    mo.stop(selected_set.value is None, mo.callout("No analyzed runs found.", kind="neutral"))
    set_dir = set_summaries[selected_set.value].parent
    summary = json.loads((set_dir / "summary.json").read_text(encoding="utf-8"))
    calibration_detail = json.loads((set_dir / "calibration.json").read_text(encoding="utf-8"))
    curves = json.loads((set_dir / "curves.json").read_text(encoding="utf-8"))
    return calibration_detail, curves, set_dir, summary


@app.cell
def _(mo, pd, summary):
    def interval_rows(bootstrap, level):
        rows = []
        for metric, interval in bootstrap["metrics"].items():
            if interval["point"] is None:
                continue
            rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "point": round(100 * interval["point"], 2),
                    "95% CI": (
                        f"{100 * interval['ci_low']:.2f} – {100 * interval['ci_high']:.2f}"
                        if interval["ci_low"] is not None
                        else "n/a"
                    ),
                    "units": bootstrap["unit_count"],
                }
            )
        return rows

    metric_rows = interval_rows(summary["bootstrap"], "image (patient bootstrap)")
    if "eye_level" in summary:
        metric_rows += interval_rows(summary["eye_level"]["bootstrap"], "eye")
    calibration = summary["calibration"]
    operating = summary["disease_operating_points"] or {}
    operating_rows = [
        {
            "operating point": name,
            **{
                key: (round(100 * value, 2) if isinstance(value, float) and key != "threshold" else value)
                for key, value in point.items()
            },
        }
        for name, point in operating.items()
        if isinstance(point, dict)
    ]
    check = summary["reference_check"]
    mo.vstack(
        [
            mo.md(
                f"### `{summary['set']}` — {summary['source']}, {summary['images']} images, "
                f"{summary['groups']} groups"
                + ("" if check is None or check["matches"] else " — **differs from saved metrics**")
            ),
            mo.hstack(
                [
                    mo.ui.table(pd.DataFrame(metric_rows), selection=None, pagination=False),
                    mo.ui.table(
                        {
                            "ECE": round(100 * calibration["ece"], 2),
                            "MCE": round(100 * calibration["mce"], 2),
                            "Brier": round(calibration["brier"], 4),
                            "NLL": round(calibration["nll"], 4),
                            "mean confidence (correct)": calibration["mean_confidence_correct"],
                            "mean confidence (wrong)": calibration["mean_confidence_wrong"],
                        },
                        selection=None,
                    ),
                ],
                justify="start",
                gap=2,
                align="start",
            ),
            mo.md(
                "#### Disease vs normal operating points (`1 - p(normal)`)\n"
                "`best_f1_test_tuned` is tuned on these labels: it measures threshold shift and is "
                "not a deployable operating point."
            ),
            mo.ui.table(pd.DataFrame(operating_rows), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(alt, calibration_detail, curves, mo, pd):
    reliability = pd.DataFrame(calibration_detail["reliability"]).dropna()
    reliability["center"] = (reliability["bin_low"] + reliability["bin_high"]) / 2
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x", y="y")
    reliability_chart = (
        diagonal
        + alt.Chart(reliability)
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_confidence", title="Mean confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["count", "mean_confidence", "accuracy"],
        )
    ).properties(title="Reliability", width=280, height=280)
    histogram = calibration_detail["confidence_histogram"]
    edges = histogram["edges"]
    histogram_frame = pd.DataFrame(
        [
            {"confidence": (edges[i] + edges[i + 1]) / 2, "outcome": outcome, "count": count}
            for outcome in ("correct", "wrong")
            for i, count in enumerate(histogram[outcome])
        ]
    )
    histogram_chart = (
        alt.Chart(histogram_frame)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("confidence:Q", bin=alt.Bin(step=edges[1] - edges[0]), title="Top-label confidence"),
            y=alt.Y("count:Q", stack=None, scale=alt.Scale(type="symlog")),
            color="outcome:N",
        )
        .properties(title="Confidence (symlog counts)", width=280, height=280)
    )
    roc_frame = pd.DataFrame(
        [
            {"class": f"{name} (AUC {curve['auroc']:.3f})", "fpr": fpr, "tpr": tpr}
            for name, curve in curves.items()
            if curve["roc"] is not None
            for fpr, tpr in zip(curve["roc"]["fpr"], curve["roc"]["tpr"], strict=True)
        ]
    )
    pr_frame = pd.DataFrame(
        [
            {"class": f"{name} (AP {curve['average_precision']:.3f})", "recall": r, "precision": p}
            for name, curve in curves.items()
            if curve["pr"] is not None
            for r, p in zip(curve["pr"]["recall"], curve["pr"]["precision"], strict=True)
        ]
    )
    roc_chart = (
        alt.Chart(roc_frame)
        .mark_line()
        .encode(x="fpr", y="tpr", color="class")
        .properties(title="ROC (one-vs-rest)", width=280, height=280)
    )
    pr_chart = (
        alt.Chart(pr_frame)
        .mark_line()
        .encode(x="recall", y=alt.Y("precision", scale=alt.Scale(zero=False)), color="class")
        .properties(title="Precision-recall", width=280, height=280)
    )
    mo.hstack(
        [
            mo.ui.altair_chart(reliability_chart),
            mo.ui.altair_chart(histogram_chart),
            mo.ui.altair_chart(roc_chart),
            mo.ui.altair_chart(pr_chart),
        ],
        justify="start",
        wrap=True,
    )
    return


@app.cell
def _(mo, pd, set_dir, summary):
    slices = pd.read_csv(set_dir / "slices.csv")
    groups = pd.read_csv(set_dir / "groups.csv")
    concentration = summary["error_concentration"]
    mo.vstack(
        [
            mo.md(
                "### Slices\n`predicted_*` is the fraction of the slice sent to each class, so raw "
                "labels show where merged classes (e.g. DRUSEN vs CNV) go."
            ),
            mo.ui.table(slices.round(4), selection=None, pagination=False),
            mo.md(
                f"### Groups by error count\n{concentration['groups_with_errors']} of "
                f"{concentration['groups']} groups have errors; the worst 10% of groups hold "
                f"{100 * (concentration['error_share_top_10pct_groups'] or 0):.1f}% of errors."
            ),
            mo.ui.table(groups.head(30).round(4), selection=None),
        ]
    )
    return


@app.cell
def _(json, mo, pd, set_dir):
    gradcam_summary_path = set_dir / "gradcam" / "summary.json"
    has_gradcam = gradcam_summary_path.is_file()
    if has_gradcam:
        gradcam_summary = json.loads(gradcam_summary_path.read_text(encoding="utf-8"))
        attention = pd.DataFrame(gradcam_summary["attention"])
        attention_view = mo.vstack(
            [
                mo.md(
                    "### Grad-CAM attention\nShare of CAM mass outside the estimated retina band "
                    "(ILM to just below the RPE; implausible masks excluded) and on padding or fill "
                    "wedges. The 7×7 `layer4` map is coarse, so compare errors with controls rather "
                    "than reading absolute values."
                ),
                mo.ui.table(attention.round(3), selection=None, pagination=False),
                mo.md(
                    "Randomization check (Spearman, trained vs `layer4`+`fc` re-initialized; low "
                    "means the maps depend on learned weights): "
                    + ", ".join(
                        f"{layer} median {stats['median']:.2f}"
                        for layer, stats in gradcam_summary["randomization_check"]
                        .get("spearman_by_layer", {})
                        .items()
                        if stats["median"] is not None
                    )
                ),
            ]
        )
    else:
        attention_view = mo.callout(
            "No Grad-CAM for this set; run `oct-classify analyze gradcam <run>`.", kind="neutral"
        )
    attention_view
    return (has_gradcam,)


@app.cell
def _(mo, pd, set_dir):
    gallery_rows = pd.concat(
        [pd.read_csv(set_dir / "errors.csv"), pd.read_csv(set_dir / "controls.csv")],
        ignore_index=True,
    )
    gallery_rows["cell"] = gallery_rows["target"] + " → " + gallery_rows["prediction"]
    cells = sorted(gallery_rows["cell"].unique())
    cell = mo.ui.dropdown(options=cells, value=cells[0], label="True → predicted")
    layer = mo.ui.radio(options=["layer4", "layer3"], value="layer4", label="CAM layer", inline=True)
    role = mo.ui.radio(options=["predicted", "true"], value="predicted", label="CAM target", inline=True)
    count = mo.ui.slider(start=4, stop=24, step=4, value=12, label="Images", show_value=True)
    mo.vstack([mo.md("### Gallery"), mo.hstack([cell, layer, role, count], justify="start", gap=2)])
    return cell, count, gallery_rows, layer, role


@app.cell
def _(
    Image,
    PreprocessingSpec,
    cell,
    count,
    dataset_roots,
    gallery_rows,
    has_gradcam,
    io,
    layer,
    mo,
    np,
    preprocess_image,
    role,
    set_dir,
):
    cams = np.load(set_dir / "gradcam" / "cams.npz") if has_gradcam else None
    masks = np.load(set_dir / "gradcam" / "masks.npz") if has_gradcam else None

    def unpack(key, size):
        return np.unpackbits(masks[key])[: size * size].reshape(size, size).astype(bool)

    def card(row):
        with Image.open(dataset_roots[row["source"]] / row["path"]) as image:
            gray = preprocess_image(image, PreprocessingSpec()).mean(axis=2)
        size = gray.shape[0]
        panels = [np.stack([gray] * 3, axis=-1)]
        if cams is not None:
            cam = cams[f"{row['row_id']}__{layer.value}__{role.value}"]
            cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize((size, size), Image.BILINEAR))
            cam = cam / cam.max() if cam.max() > 0 else cam
            retina = unpack(f"{row['row_id']}__retina", size)
            edge = retina & ~np.pad(retina, 1)[2:, 1:-1] | retina & ~np.pad(retina, 1)[:-2, 1:-1]
            overlay = np.stack([gray] * 3, axis=-1) * 0.6
            overlay[..., 0] = np.clip(overlay[..., 0] + cam, 0, 1)
            overlay[edge] = [0.2, 1.0, 0.2]
            panels.append(overlay)
        output = io.BytesIO()
        Image.fromarray((np.concatenate(panels, axis=1) * 255).astype(np.uint8)).save(output, "PNG")
        probabilities = ", ".join(
            f"{column.removeprefix('probability_')} {row[column]:.2f}"
            for column in row.index
            if column.startswith("probability_") and row[column] == row[column]
        )
        return mo.vstack(
            [
                mo.image(src=output.getvalue(), width=2 * 240 if cams is not None else 240),
                mo.md(f"`{row['path']}`  \n{row['kind']} · {row['group_key']} · {probabilities}"),
            ]
        )

    selected_rows = gallery_rows[gallery_rows["cell"] == cell.value].head(count.value)
    mo.hstack([card(row) for _, row in selected_rows.iterrows()], justify="start", gap=1, wrap=True)
    return


if __name__ == "__main__":
    app.run()
