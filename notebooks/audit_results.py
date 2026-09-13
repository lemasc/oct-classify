import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from PIL import Image

    repository_root = Path(__file__).resolve().parents[1]
    audit_directory = repository_root / "artifacts" / "audits"
    dataset_roots = {
        "duke": repository_root / "datasets" / "duke1",
        "kermany": repository_root / "datasets" / "kermany",
        "octdl": repository_root / "datasets" / "octdl",
        "octid": repository_root / "datasets" / "octid",
        "paima": repository_root / "datasets" / "paima",
    }
    report_paths = sorted(audit_directory.glob("*.json"))
    reports = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in report_paths
        if path.stem in dataset_roots
    }
    return Image, dataset_roots, io, mo, pd, reports


@app.cell
def _(mo):
    mo.md("""
    # OCT Audit Results

    Interactive review of the full exact-content and pHash duplicate audit. The images below are
    read directly from the configured `datasets/` symlinks; rerun `oct-classify audit` before
    opening this notebook when the audit artifacts are stale.
    """)
    return


@app.cell
def _(pd, reports):
    summary = pd.DataFrame(
        [
            {
                "dataset": name,
                "images": report["image_count"],
                "exact clusters": len(report["exact_duplicates"]),
                "pHash clusters": len(report["near_duplicates"]),
                "label-conflicting clusters": sum(
                    cluster["conflicting_labels"]
                    for kind in ("exact_duplicates", "near_duplicates")
                    for cluster in report[kind]
                ),
                "cross-split clusters": sum(
                    cluster["crosses_supplied_splits"]
                    for kind in ("exact_duplicates", "near_duplicates")
                    for cluster in report[kind]
                ),
                "unmanifested images": len(report["unmanifested_images"]),
                "invalid images": len(report["invalid_images"]),
            }
            for name, report in reports.items()
        ]
    ).sort_values("dataset")
    return (summary,)


@app.cell
def _(mo, summary):
    mo.ui.table(summary, selection=None, pagination=False)
    return


@app.cell
def _(mo, reports):
    dataset_names = sorted(reports)
    phash_distances = sorted(
        {
            cluster["distance"]
            for report in reports.values()
            for cluster in report["near_duplicates"]
        }
    )
    dataset = mo.ui.dropdown(options=dataset_names, value=dataset_names[0], label="Dataset")
    duplicate_kind = mo.ui.radio(
        options=["Exact-content duplicates", "Perceptual (pHash) duplicates"],
        value="Exact-content duplicates",
        label="Duplicate method",
    )
    phash_distance = mo.ui.dropdown(
        options=["All distances", *phash_distances],
        value="All distances",
        label="pHash distance",
    )
    scope = mo.ui.radio(
        options=[
            "All clusters",
            "Integrity findings only",
            "Non-integrity findings only",
        ],
        value="Integrity findings only",
        label="Cluster filter",
    )
    mo.hstack([dataset, duplicate_kind, phash_distance, scope], justify="start", gap=2)
    return dataset, duplicate_kind, phash_distance, scope


@app.cell
def _(dataset, duplicate_kind, phash_distance, reports, scope):
    selected_report = reports[dataset.value]
    duplicate_key = (
        "exact_duplicates"
        if duplicate_kind.value == "Exact-content duplicates"
        else "near_duplicates"
    )
    clusters = selected_report[duplicate_key]
    if (
        duplicate_kind.value == "Perceptual (pHash) duplicates"
        and phash_distance.value != "All distances"
    ):
        clusters = [cluster for cluster in clusters if cluster["distance"] == phash_distance.value]
    is_integrity_finding = lambda cluster: (
        cluster["conflicting_labels"] or cluster["crosses_supplied_splits"]
    )
    if scope.value == "Integrity findings only":
        clusters = [cluster for cluster in clusters if is_integrity_finding(cluster)]
    elif scope.value == "Non-integrity findings only":
        clusters = [cluster for cluster in clusters if not is_integrity_finding(cluster)]
    return clusters, selected_report


@app.cell
def _(clusters, duplicate_kind, mo):
    from collections import Counter as _Counter

    if duplicate_kind.value != "Perceptual (pHash) duplicates":
        phash_distribution = mo.callout(
            "Select Perceptual (pHash) duplicates to view its distance distribution.",
            kind="neutral",
        )
    else:
        _distance_counts = _Counter(cluster["distance"] for cluster in clusters)
        _largest_count = max(_distance_counts.values(), default=1)
        _total_clusters = sum(_distance_counts.values())
        _rows = "\n".join(
            f"| {distance} | {'#' * max(1, round(36 * count / _largest_count))} | {count} | "
            f"{count / _total_clusters:.1%} |"
            for distance, count in sorted(_distance_counts.items())
        )
        _table = (
            "| pHash distance | Relative frequency | Clusters | Share |\n"
            "| ---: | :--- | ---: | ---: |\n"
            f"{_rows}"
            if _distance_counts
            else "No pHash clusters match the selected filters."
        )
        phash_distribution = mo.vstack(
            [
                mo.md("## pHash Distance Distribution"),
                mo.md(_table),
            ]
        )
    phash_distribution
    return


@app.cell
def _(clusters, mo):
    cluster_number = mo.ui.slider(
        start=1,
        stop=max(len(clusters), 1),
        value=1,
        step=1,
        label=f"Cluster (of {len(clusters)})",
        show_value=True,
    )
    cluster_number  # noqa: B018
    return (cluster_number,)


@app.cell
def _(cluster_number, clusters, mo, selected_report):
    if not clusters:
        selected_details = mo.callout(
            "No duplicate clusters match the selected filters.", kind="neutral"
        )
        selected_cluster = None
    else:
        selected_cluster = clusters[cluster_number.value - 1]
        details = {
            "dataset": selected_report["dataset"],
            "paths": len(selected_cluster["paths"]),
            "labels": ", ".join(selected_cluster["labels"]),
            "supplied splits": ", ".join(selected_cluster["supplied_splits"]) or "none",
            "group keys": ", ".join(selected_cluster["group_keys"]) or "none",
            "pHash distance": selected_cluster.get("distance", "exact content"),
            "conflicting labels": selected_cluster["conflicting_labels"],
            "crosses supplied splits": selected_cluster["crosses_supplied_splits"],
        }
        selected_details = mo.vstack(
            [mo.md("### Selected Cluster"), mo.ui.table(details, selection=None)]
        )
    selected_details  # noqa: B018
    return (selected_cluster,)


@app.cell
def _(Image, dataset_roots, io, mo, selected_cluster):
    def image_card(source_path: str):
        source, _, relative_path = source_path.partition(":")
        image_path = dataset_roots[source] / relative_path
        try:
            with Image.open(image_path) as image:
                thumbnail = image.convert("L")
                thumbnail.thumbnail((360, 360))
                output = io.BytesIO()
                thumbnail.save(output, format="PNG")
        except OSError as error:
            return mo.vstack([mo.md(f"`{source_path}`"), mo.callout(str(error), kind="danger")])
        return mo.vstack([mo.image(src=output.getvalue(), width=360), mo.md(f"`{source_path}`")])

    if selected_cluster is None:
        gallery = None
    else:
        gallery = mo.vstack(
            [
                mo.md("### Duplicate Images"),
                mo.hstack(
                    [image_card(source_path) for source_path in selected_cluster["paths"]],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
            ]
        )
    gallery
    return


if __name__ == "__main__":
    app.run()
