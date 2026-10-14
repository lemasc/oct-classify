import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from scipy.stats import spearmanr

    from oct_classify.analysis.gradcam import GradCam, randomize_top, upsample
    from oct_classify.analysis.retina import content_mask, mass_fraction
    from oct_classify.data.config import load_dataset_specs
    from oct_classify.data.manifest import read_jsonl
    from oct_classify.data.preprocessing import PreprocessingSpec
    from oct_classify.data.taxonomy import UnifiedLabel
    from oct_classify.models import build_model
    from oct_classify.training.config import AugmentationConfig
    from oct_classify.training.dataset import ManifestImageDataset

    torch.set_num_threads(6)  # leave CPU for other jobs; everything here runs on CPU
    repository_root = Path(__file__).resolve().parents[1]
    runs_root = repository_root / "artifacts" / "runs" / "fused" / "resnet50"
    cache_dir = repository_root / "artifacts" / "analysis" / "gradcam-shortcut"
    summary_path = repository_root / "data" / "paper" / "gradcam" / "summary.json"
    kermany_root = {
        spec.name: repository_root / spec.root
        for spec in load_dataset_specs(repository_root / "configs" / "datasets.toml")
    }["kermany"]
    label_order = ["normal", "amd", "dme"]
    layer = "layer4"
    image_size = 224
    seed = 20261014
    return (
        AugmentationConfig,
        GradCam,
        Image,
        ManifestImageDataset,
        PreprocessingSpec,
        UnifiedLabel,
        alt,
        build_model,
        cache_dir,
        content_mask,
        image_size,
        json,
        kermany_root,
        label_order,
        layer,
        mass_fraction,
        mo,
        np,
        pd,
        randomize_top,
        read_jsonl,
        repository_root,
        runs_root,
        seed,
        spearmanr,
        summary_path,
        torch,
        upsample,
    )


@app.cell
def _(mo):
    mo.md("""
    # Where do held-out models look? Grad-CAM and the image-size shortcut

    Models trained with Duke learn that a 768×496 image is DME (every Duke image of that size is), and held-out
    Kermany models call many normal 768×496 Kermany images DME. Experiments 1b and 1c showed this by changing image
    size at test time ([`size_shortcut.py`](size_shortcut.py)). This notebook asks where in the image the evidence for
    those DME calls sits, and whether it is the letterbox bands, the empty strips added above and below a wide scan
    to make it square.

    **Models.** LOSO-Kermany fill-aware (`loso-kermany-local-20261008-174938-fold{1..5}`) and, for comparison,
    square crop (`loso-kermany-local-3a-r2-square-crop-fold1`). Images go through each model's own evaluation
    pipeline (`ManifestImageDataset`, `training=False`), so square-crop models see their centred square window.

    **Grad-CAM** is computed at `layer4` for the stored predicted class, on CPU. `layer4` is a 7×7 grid, so a map is
    coarse: on a 768×496 image the bands are about 40 px of the 224 px canvas at the top and at the bottom, roughly
    the first and last of the seven rows. *Band share* is the share of a map's mass that falls on the bands; the
    bands cover 35% of the canvas, so a map that ignored them would put about a third there only if it were flat.

    Sections: 1 setup and checks, 2 examples, 3 band share by group, 4 across folds, 5 randomization check,
    6 intervention on the band pixels, 7 reading.
    """)
    return


@app.cell
def _(
    AugmentationConfig,
    GradCam,
    ManifestImageDataset,
    PreprocessingSpec,
    UnifiedLabel,
    build_model,
    content_mask,
    image_size,
    kermany_root,
    layer,
    np,
    torch,
    upsample,
):
    def load_run(run_dir):
        checkpoint = torch.load(run_dir / "checkpoint-best.pt", map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        model = build_model(metadata["architecture"], len(metadata["class_labels"]), pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        return model.eval(), metadata

    def band_mask(width, height):
        return ~content_mask(width, height, image_size)

    def run_cams(model, metadata, records, sizes, band_fill="trained", rng=None, with_cam=True):
        """Probabilities and predicted-class layer4 maps for `records`, optionally changing the band pixels.

        `band_fill`: "trained" (as preprocessed), "black" (0 before normalization), or "speckle" (band pixels
        drawn from the darker half of the image's own content, i.e. background speckle).
        """
        config = metadata["config"]
        labels = tuple(UnifiedLabel(label) for label in metadata["class_labels"])
        mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
        stdev = np.maximum(np.asarray(metadata["normalization"]["stdev"], dtype=np.float32), 1e-6)
        dataset = ManifestImageDataset(
            {"kermany": kermany_root},
            records,
            labels,
            PreprocessingSpec(image_size=config["data"]["image_size"], padding=config["data"]["padding"]),
            mean,
            stdev,
            AugmentationConfig(**config["augmentation"]),
            training=False,
        )
        squared = config["augmentation"].get("geometry", "none") == "square_crop"
        probabilities, cams, displays = [], [], []
        cam = GradCam(model, [layer]) if with_cam else None
        for start in range(0, len(records), 16):
            batch = torch.stack([dataset[index][0] for index in range(start, min(start + 16, len(records)))])
            raw = batch * torch.tensor(stdev).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
            if band_fill != "trained" and not squared:
                for offset, (width, height) in enumerate(sizes[start : start + len(batch)]):
                    bands = torch.from_numpy(band_mask(width, height))
                    if band_fill == "black":
                        raw[offset][:, bands] = 0.0
                    elif band_fill == "speckle":
                        content = raw[offset][0][~bands]
                        dark = content[content <= content.median()]
                        draw = dark[torch.from_numpy(rng.integers(0, len(dark), int(bands.sum())))]
                        raw[offset][:, bands] = draw
                batch = (raw - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(stdev).view(1, 3, 1, 1)
            displays.append(raw.mean(dim=1).numpy())
            if with_cam:
                _, logits = cam(batch, torch.zeros(len(batch), dtype=torch.long))
                predicted = logits.argmax(1)
                maps, _ = cam(batch, predicted)
                cams.append(upsample(maps[layer], image_size).numpy().astype(np.float16))
            else:
                with torch.no_grad():
                    logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).numpy())
        if cam is not None:
            cam.close()
        return {
            "labels": [label.value for label in labels],
            "probabilities": np.concatenate(probabilities),
            "cams": np.concatenate(cams) if cams else None,
            "display": np.concatenate(displays).astype(np.float16),
        }

    return band_mask, load_run, run_cams


@app.cell
def _(mo):
    mo.md("""
    ## 1. Setup and checks

    Kermany image sizes come from `artifacts/eda/features.csv`; predictions from each model's stored full-source
    evaluation. Every sample below takes at most one image per patient.
    """)
    return


@app.cell
def _(pd, read_jsonl, repository_root, runs_root):
    features = pd.read_csv(
        repository_root / "artifacts" / "eda" / "features.csv", usecols=["source", "path", "width", "height"]
    )
    features = features[features.source == "kermany"].set_index("path")
    features["size"] = features.width.astype(str) + "x" + features.height.astype(str)
    kermany_records = {record.path: record for record in read_jsonl(repository_root / "artifacts" / "splits" / "kermany.jsonl")}

    fill_runs = {fold: runs_root / f"loso-kermany-local-20261008-174938-fold{fold}" for fold in range(1, 6)}
    square_run = runs_root / "loso-kermany-local-3a-r2-square-crop-fold1"

    def stored_predictions(run_dir):
        table = pd.read_csv(run_dir / "evaluations" / "kermany-loso" / "predictions.csv").set_index("path")
        table = table.join(features[["size", "width", "height"]])
        table["patient"] = [path.split("/")[-1].split("-")[1] for path in table.index]
        return table

    return fill_runs, kermany_records, square_run, stored_predictions


@app.cell
def _(np):
    GROUPS = {
        # name: (size, true class, predicted class)
        "768 normal → DME": ("768x496", "normal", "dme"),
        "768 normal → normal": ("768x496", "normal", "normal"),
        "768 AMD → DME": ("768x496", "amd", "dme"),
        "768 AMD → AMD": ("768x496", "amd", "amd"),
        "768 DME → DME": ("768x496", "dme", "dme"),
        "512 normal → DME": ("512x496", "normal", "dme"),
        "512 normal → normal": ("512x496", "normal", "normal"),
        "512 DME → DME": ("512x496", "dme", "dme"),
    }
    N_PER_GROUP = 40

    def sample_group(table, size, true, predicted, n, rng_seed):
        pool = table[(table["size"] == size) & (table.target == true) & (table.prediction == predicted)]
        pool = pool.groupby("patient").head(1)
        rng = np.random.default_rng(rng_seed)
        take = min(n, len(pool))
        return list(pool.index[np.sort(rng.choice(len(pool), size=take, replace=False))]) if take else []

    return GROUPS, N_PER_GROUP, sample_group


@app.cell
def _(
    GROUPS,
    N_PER_GROUP,
    band_mask,
    cache_dir,
    kermany_records,
    load_run,
    mass_fraction,
    mo,
    np,
    pd,
    run_cams,
    sample_group,
    seed,
    stored_predictions,
):
    def group_cams(run_dir, group, fold_seed):
        """Cached Grad-CAM for one run and group; returns a per-image table and the maps."""
        cache = cache_dir / f"{run_dir.name}__{group.replace(' ', '_').replace('→', 'to')}.npz"
        table = stored_predictions(run_dir)
        size, true, predicted = GROUPS[group]
        paths = sample_group(table, size, true, predicted, N_PER_GROUP, fold_seed)
        if not paths:
            return pd.DataFrame(), None
        if cache.exists():
            stored = np.load(cache, allow_pickle=False)
            assert list(stored["paths"]) == paths, f"stale cache {cache}"
            probabilities, cams, display = stored["probabilities"], stored["cams"], stored["display"]
            labels = list(stored["labels"])
        else:
            model, metadata = load_run(run_dir)
            sizes = [tuple(table.loc[path, ["width", "height"]].astype(int)) for path in paths]
            out = run_cams(model, metadata, [kermany_records[path] for path in paths], sizes)
            probabilities, cams, display, labels = out["probabilities"], out["cams"], out["display"], out["labels"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, paths=np.array(paths), probabilities=probabilities, cams=cams,
                                display=display, labels=np.array(labels))
        rows = []
        for index, path in enumerate(paths):
            width, height = (int(v) for v in table.loc[path, ["width", "height"]])
            stored_row = table.loc[path]
            rows.append({
                "run": run_dir.name, "group": group, "path": path,
                "band_share": mass_fraction(cams[index].astype(np.float32), band_mask(width, height)),
                "band_area": float(band_mask(width, height).mean()),
                "reproduced": labels[int(probabilities[index].argmax())] == stored_row.prediction,
                "max_abs_prob_diff": float(np.abs(
                    probabilities[index] - stored_row[[f"probability_{label}" for label in labels]].to_numpy(float)
                ).max()),
            })
        return pd.DataFrame(rows), {"cams": cams, "display": display, "paths": paths, "probabilities": probabilities}

    def all_groups(run_dir, fold_seed):
        tables, maps = [], {}
        with mo.status.progress_bar(total=len(GROUPS), title=f"Grad-CAM {run_dir.name}") as bar:
            for group in GROUPS:
                table, data = group_cams(run_dir, group, fold_seed)
                tables.append(table)
                maps[group] = data
                bar.update()
        return pd.concat(tables, ignore_index=True), maps

    _ = seed
    return all_groups, group_cams


@app.cell
def _(all_groups, fill_runs, seed):
    fold1_table, fold1_maps = all_groups(fill_runs[1], seed)
    return fold1_maps, fold1_table


@app.cell
def _(fold1_table, mo):
    _check = fold1_table.groupby("group").agg(
        n=("path", "size"), reproduced=("reproduced", "mean"), max_abs_prob_diff=("max_abs_prob_diff", "max")
    ).reset_index()
    mo.vstack([
        mo.md("**Check (fold 1):** recomputed predictions against the stored evaluation (CPU vs the original GPU run)."),
        mo.ui.table(_check.round(4), selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Examples

    Each image as the model sees it (224×224 canvas), with its Grad-CAM for the predicted class. Pick a group and
    page through. The square-crop model's map for the same image is shown below it.
    """)
    return


@app.cell
def _(GROUPS, mo):
    example_group = mo.ui.dropdown(list(GROUPS), value="768 normal → DME", label="Group (fill-aware fold 1)")
    example_page = mo.ui.number(start=0, stop=9, step=1, value=0, label="Page (6 per page)")
    mo.hstack([example_group, example_page], justify="start")
    return example_group, example_page


@app.cell
def _(Image, np):
    _STOPS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=float)

    def overlay(gray, heat, alpha=0.45):
        """Gray canvas with an inferno-like heat overlay, as a PIL image (no matplotlib)."""
        gray = np.clip(np.asarray(gray, dtype=float), 0, 1)
        heat = np.asarray(heat, dtype=float)
        heat = heat / heat.max() if heat.max() > 0 else heat
        position = heat * (len(_STOPS) - 1)
        low = np.floor(position).astype(int).clip(0, len(_STOPS) - 2)
        fraction = (position - low)[..., None]
        colour = _STOPS[low] * (1 - fraction) + _STOPS[low + 1] * fraction
        base = np.repeat(gray[..., None] * 255, 3, axis=2)
        mixed = base * (1 - alpha) + colour * alpha
        return Image.fromarray(mixed.clip(0, 255).astype(np.uint8)).resize((180, 180))

    return (overlay,)


@app.cell
def _(
    example_group,
    example_page,
    fold1_maps,
    fold1_table,
    kermany_records,
    load_run,
    mo,
    overlay,
    run_cams,
    square_run,
    stored_predictions,
):
    _data = fold1_maps[example_group.value]
    mo.stop(_data is None, mo.md("_No images in this group._"))
    _start = example_page.value * 6
    _paths = _data["paths"][_start : _start + 6]
    mo.stop(not _paths, mo.md("_No more images._"))
    _table = fold1_table.set_index("path")
    _square_model, _square_meta = load_run(square_run)
    _square_table = stored_predictions(square_run)
    _square = run_cams(_square_model, _square_meta, [kermany_records[p] for p in _paths],
                       [tuple(_square_table.loc[p, ["width", "height"]].astype(int)) for p in _paths])
    _cells = []
    for _i, _path in enumerate(_paths):
        _index = _start + _i
        _fill_label = ["normal", "amd", "dme"][int(_data["probabilities"][_index].argmax())]
        _square_label = _square["labels"][int(_square["probabilities"][_i].argmax())]
        _cells.append(mo.vstack([
            mo.md(f"`{_path.split('/')[-1]}`<br>fill-aware: **{_fill_label}**, band share "
                  f"{_table.loc[_path, 'band_share']:.0%}"),
            mo.image(overlay(_data["display"][_index], _data["cams"][_index])),
            mo.md(f"square crop: **{_square_label}**"),
            mo.image(overlay(_square["display"][_i], _square["cams"][_i])),
        ]))
    mo.hstack(_cells, justify="start", wrap=True)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Band share by group (fill-aware, fold 1)

    If the bands carry the shortcut, maps for wrong DME calls on 768×496 images should sit on the bands, and maps
    for correct calls should not. The 512×496 groups are a contrast: their bands are 3-4 px, so there is almost no
    band to read. True DME images called DME show whether real DME also leans on the bands.
    """)
    return


@app.cell
def _(alt, fold1_table, mo):
    _summary = fold1_table.groupby("group", sort=False).agg(
        n=("path", "size"),
        band_area=("band_area", "mean"),
        mean_band_share=("band_share", "mean"),
        median_band_share=("band_share", "median"),
        over_half=("band_share", lambda s: (s > 0.5).mean()),
    ).reset_index()
    _chart = (
        alt.Chart(fold1_table)
        .mark_tick(thickness=2, size=14, opacity=0.6, color="#6250d6")
        .encode(
            x=alt.X("band_share:Q", title="Share of Grad-CAM mass on the letterbox bands", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None, sort=list(_summary.group)),
            tooltip=["path", alt.Tooltip("band_share:Q", format=".0%")],
        )
        .properties(width=520, height=260)
    )
    mo.vstack([mo.ui.altair_chart(_chart), mo.ui.table(_summary.round(3), selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Across folds

    The same groups on all five fill-aware LOSO-Kermany models (each model's own predictions, its own sample).
    The first run computes and caches about 1,600 maps on CPU (several minutes); later runs load the cache.
    """)
    return


@app.cell
def _(mo):
    run_folds = mo.ui.checkbox(value=True, label="Include folds 2-5 (computed once, then cached)")
    run_folds
    return (run_folds,)


@app.cell
def _(all_groups, fill_runs, fold1_table, mo, pd, run_folds, seed):
    mo.stop(not run_folds.value, mo.md("_Folds 2-5 not included._"))
    folds_table = pd.concat(
        [fold1_table.assign(fold=1)]
        + [all_groups(fill_runs[fold], seed + fold)[0].assign(fold=fold) for fold in range(2, 6)],
        ignore_index=True,
    )
    return (folds_table,)


@app.cell
def _(alt, folds_table, mo):
    _by_fold = folds_table.groupby(["group", "fold"], sort=False).agg(
        n=("path", "size"),
        mean_band_share=("band_share", "mean"),
        over_half=("band_share", lambda s: (s > 0.5).mean()),
        reproduced=("reproduced", "mean"),
    ).reset_index()
    _chart = (
        alt.Chart(_by_fold)
        .mark_circle(size=60, color="#6250d6")
        .encode(
            x=alt.X("mean_band_share:Q", title="Mean band share", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None, sort=list(dict.fromkeys(_by_fold.group))),
            tooltip=["fold", "n", alt.Tooltip("mean_band_share:Q", format=".0%"), alt.Tooltip("over_half:Q", format=".0%")],
        )
        .properties(width=520, height=260)
    )
    mo.vstack([
        mo.md("One dot per fold."),
        mo.ui.altair_chart(_chart),
        mo.ui.table(_by_fold.round(3), selection=None, page_size=40),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Randomization check

    If a map barely changes after the learned top of the network (`layer4` and `fc`) is re-initialized, it reflects
    image structure rather than what the classifier learned (Adebayo et al., 2018). For the fold-1 "768 normal → DME"
    images we compare trained and randomized maps (Spearman over pixels) and the band share of each.
    """)
    return


@app.cell
def _(
    band_mask,
    fill_runs,
    fold1_maps,
    fold1_table,
    kermany_records,
    load_run,
    mass_fraction,
    mo,
    np,
    pd,
    randomize_top,
    run_cams,
    spearmanr,
    stored_predictions,
    torch,
):
    _data = fold1_maps["768 normal → DME"]
    _model, _meta = load_run(fill_runs[1])
    torch.manual_seed(0)
    _random = randomize_top(_model).eval()
    _table = stored_predictions(fill_runs[1])
    _sizes = [tuple(_table.loc[p, ["width", "height"]].astype(int)) for p in _data["paths"]]
    _out = run_cams(_random, _meta, [kermany_records[p] for p in _data["paths"]], _sizes)
    _rows = []
    for _i, (_path, (_w, _h)) in enumerate(zip(_data["paths"], _sizes)):
        _trained = _data["cams"][_i].astype(np.float32).ravel()
        _rand = _out["cams"][_i].astype(np.float32).ravel()
        _rows.append({
            "spearman": float(spearmanr(_trained, _rand).statistic) if _trained.std() and _rand.std() else np.nan,
            "band_share_trained": fold1_table.set_index("path").loc[_path, "band_share"],
            "band_share_randomized": mass_fraction(_out["cams"][_i].astype(np.float32), band_mask(_w, _h)),
        })
    randomization = pd.DataFrame(_rows)
    mo.ui.table(randomization.describe().round(3), selection=None)
    return (randomization,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Intervention: change only the band pixels

    Grad-CAM shows where the gradient points; it does not show that the bands *cause* the call. Here the band pixels
    are replaced and the prediction re-read, with the tissue untouched:

    - **black**: bands set to 0 before normalization, instead of the scan's median;
    - **speckle**: bands filled with pixels drawn from the darker half of the image's own content (background
      speckle), so they no longer look like a flat strip.

    Applied on every fold to that fold's 768×496 normal and AMD samples. If the flat median band drives the DME
    calls, they should drop when the band stops looking flat; the correct calls should not change much.
    """)
    return


@app.cell
def _(
    fill_runs,
    folds_table,
    kermany_records,
    load_run,
    np,
    pd,
    run_cams,
    seed,
    stored_predictions,
):
    # Every fold, on that fold's own sample of each group (the Grad-CAM sample from sections 3-4).
    _rows = []
    for _fold, _run in fill_runs.items():
        _model, _meta = load_run(_run)
        _table = stored_predictions(_run)
        for _group in ["768 normal → DME", "768 normal → normal", "768 AMD → DME", "768 AMD → AMD"]:
            _paths = list(folds_table[(folds_table.fold == _fold) & (folds_table.group == _group)].path)
            if not _paths:
                continue
            _sizes = [tuple(_table.loc[p, ["width", "height"]].astype(int)) for p in _paths]
            for _fill in ["trained", "black", "speckle"]:
                _out = run_cams(_model, _meta, [kermany_records[p] for p in _paths], _sizes, band_fill=_fill,
                                rng=np.random.default_rng(seed), with_cam=False)
                _labels = _out["labels"]
                _pred = np.array(_labels)[_out["probabilities"].argmax(1)]
                _true = _group.split()[1].lower()
                _rows.append({
                    "fold": _fold, "group": _group, "band fill": _fill, "n": len(_paths),
                    "called DME": float((_pred == "dme").mean()),
                    "called true class": float((_pred == _true).mean()),
                    "mean p(DME)": float(_out["probabilities"][:, _labels.index("dme")].mean()),
                })
    intervention = pd.DataFrame(_rows)
    return (intervention,)


@app.cell
def _(intervention, mo):
    _pivot = intervention.pivot_table(index=["group", "fold"], columns="band fill", values="called DME").reset_index()
    mo.vstack([
        mo.md("Share of images called DME, by band fill (columns), per fold."),
        mo.ui.table(_pivot.round(3), selection=None, page_size=40),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Reading

    Filled in from the results above once they are reviewed. The summary below is written to
    `data/paper/gradcam/summary.json` for the paper.
    """)
    return


@app.cell
def _(folds_table, intervention, json, mo, randomization, summary_path):
    _by_group = folds_table.groupby("group", sort=False).agg(
        n=("path", "size"),
        mean_band_share=("band_share", "mean"),
        median_band_share=("band_share", "median"),
        over_half=("band_share", lambda s: float((s > 0.5).mean())),
        band_area=("band_area", "mean"),
    )
    _by_fold = folds_table.groupby(["group", "fold"], sort=False)["band_share"].mean().unstack()
    summary = {
        "models": "LOSO-Kermany fill-aware folds 1-5 (loso-kermany-local-20261008-174938); layer4 Grad-CAM, predicted class",
        "per_group_all_folds": _by_group.reset_index().to_dict(orient="records"),
        "mean_band_share_by_fold": {group: row.dropna().to_dict() for group, row in _by_fold.iterrows()},
        "reproduced_rate": float(folds_table.reproduced.mean()),
        "randomization_fold1_768_normal_to_dme": {
            "n": int(len(randomization)),
            "spearman_median": float(randomization.spearman.median()),
            "band_share_trained_mean": float(randomization.band_share_trained.mean()),
            "band_share_randomized_mean": float(randomization.band_share_randomized.mean()),
        },
        "intervention_by_fold": intervention.to_dict(orient="records"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    mo.md(f"Wrote `{summary_path.relative_to(summary_path.parents[3])}`.")
    return


if __name__ == "__main__":
    app.run()
