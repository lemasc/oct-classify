import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import io
    import json
    import re
    from functools import cache, lru_cache
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from oct_classify.analysis.gradcam import GradCam, upsample
    from oct_classify.analysis.retina import content_mask, fill_mask, mass_fraction
    from oct_classify.data.config import load_dataset_specs
    from oct_classify.data.manifest import read_jsonl
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image
    from oct_classify.models import build_model
    from oct_classify.training.config import AugmentationConfig

    repository_root = Path(__file__).resolve().parents[1]
    runs_root = repository_root / "artifacts" / "runs"
    dataset_specs = {
        spec.name: spec
        for spec in load_dataset_specs(repository_root / "configs" / "datasets.toml")
        if spec.enabled
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        AugmentationConfig,
        GradCam,
        Image,
        PreprocessingSpec,
        alt,
        build_model,
        cache,
        content_mask,
        dataclasses,
        dataset_specs,
        device,
        fill_mask,
        io,
        json,
        lru_cache,
        mass_fraction,
        mo,
        np,
        pd,
        preprocess_image,
        re,
        read_jsonl,
        repository_root,
        runs_root,
        torch,
        upsample,
    )


@app.cell
def _(mo):
    mo.md("""
    # OCT Inference Demo

    Runs trained ResNet-50 checkpoints from `artifacts/runs/` on a single OCT B-scan. The model
    is **not final**: this notebook is for exploring the current runs, not for clinical use.

    Pick a campaign (one checkpoint per Duke-CV fold), pick an image, and the notebook shows the
    model input, the per-fold and ensemble (mean softmax) predictions, Grad-CAM, and an optional
    whole eye or volume score. Each image is tagged with its split *for each selected model*, so
    predictions on images a model trained on are flagged rather than passed off as results.
    """)
    return


@app.cell
def _(AugmentationConfig, dataclasses, json, re, runs_root):
    # Only finished runs whose config this branch can read. Runs that use options from other
    # branches (e.g. `augmentation.geometry`) are listed as skipped instead of loaded.
    known_augmentation = {field.name for field in dataclasses.fields(AugmentationConfig)}
    campaigns: dict[str, list] = {}
    skipped_campaigns: dict[str, str] = {}
    for _checkpoint in sorted(runs_root.glob("*/*/*/checkpoint-best.pt")):
        _run_dir = _checkpoint.parent
        _campaign = re.sub(r"-fold\d+$", "", _run_dir.name)
        _key = f"{_run_dir.parents[1].name}/{_campaign}"
        _config = json.loads((_run_dir / "config.json").read_text(encoding="utf-8"))
        _unknown = sorted(set(_config["augmentation"]) - known_augmentation)
        if _unknown:
            skipped_campaigns[_key] = f"unsupported config keys: {', '.join(_unknown)}"
        elif not (_run_dir / "metrics-test.json").exists():
            skipped_campaigns.setdefault(_key, "unfinished (no metrics-test.json)")
        else:
            campaigns.setdefault(_key, []).append(_run_dir)
    return campaigns, skipped_campaigns


@app.cell
def _(cache, json, read_jsonl, repository_root):
    @cache
    def run_info(run_dir):
        """Training config and source/split provenance of one run (no checkpoint load)."""
        dataset = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
        if "source" in dataset:
            # Single-source baselines record one source and manifest instead of the fused layout.
            dataset["sources"] = [dataset["source"]]
            dataset["split_manifests"] = {dataset["source"]: dataset["split_manifest"]}
        return {
            "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
            "dataset": dataset,
        }

    @cache
    def split_index(manifest):
        return {record.path: record.split for record in read_jsonl(repository_root / manifest)}

    def image_status(run_dir, source, path):
        """Where `path` sat for this run: held-out source, or its train/val/test split."""
        dataset = run_info(run_dir)["dataset"]
        if source not in dataset["sources"]:
            return "held-out"
        return split_index(dataset["split_manifests"][source]).get(path, "unlisted")

    return image_status, run_info


@app.cell
def _(campaigns, mo, run_info, skipped_campaigns):
    def _describe(key):
        _runs = campaigns[key]
        _padding = run_info(_runs[0])["config"]["data"].get("padding", "black")
        _sources = "+".join(run_info(_runs[0])["dataset"]["sources"])
        return f"{key} · {_padding} · {_sources} · {len(_runs)} model(s)"

    campaign_labels = {_describe(key): key for key in campaigns}
    _default = next(
        (
            label
            for label, key in campaign_labels.items()
            if key == "fused/loso-kermany-local-20261008-174938"
        ),
        next(iter(campaign_labels), None),
    )
    campaign = mo.ui.dropdown(
        options=campaign_labels, value=_default, label="Campaign", searchable=True
    )
    _skipped = (
        mo.accordion(
            {
                f"{len(skipped_campaigns)} campaign(s) not loadable on this branch": mo.md(
                    "\n".join(f"- `{key}`: {why}" for key, why in sorted(skipped_campaigns.items()))
                )
            }
        )
        if skipped_campaigns
        else mo.md("")
    )
    mo.vstack([campaign, _skipped])
    return (campaign,)


@app.cell
def _(campaign, campaigns, mo):
    mo.stop(
        campaign.value is None, mo.callout("No finished runs under artifacts/runs/.", kind="warn")
    )
    _names = [run_dir.name for run_dir in campaigns[campaign.value]]
    folds = mo.ui.multiselect(options=_names, value=_names, label="Models (ensemble)")
    folds  # noqa: B018
    return (folds,)


@app.cell
def _(campaign, campaigns, folds, mo, run_info):
    mo.stop(not folds.value, mo.callout("Select at least one model.", kind="warn"))
    selected_runs = [
        run_dir for run_dir in campaigns[campaign.value] if run_dir.name in folds.value
    ]
    _data = run_info(selected_runs[0])["config"]["data"]
    # Checkpoints from before `data.padding` existed were trained with black padding.
    image_size, padding = _data["image_size"], _data.get("padding", "black")
    trained_sources = sorted(
        {source for run_dir in selected_runs for source in run_info(run_dir)["dataset"]["sources"]}
    )
    return image_size, padding, selected_runs, trained_sources


@app.cell
def _(build_model, device, lru_cache, torch):
    @lru_cache(maxsize=10)
    def load_run(run_dir):
        """Model in eval mode on `device` plus checkpoint metadata (labels, normalization)."""
        checkpoint = torch.load(
            run_dir / "checkpoint-best.pt", map_location="cpu", weights_only=False
        )
        metadata = checkpoint["metadata"]
        model = build_model(
            metadata["architecture"], len(metadata["class_labels"]), pretrained=False
        )
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval(), metadata

    return (load_run,)


@app.cell
def _(device, load_run, mo, selected_runs):
    with mo.status.spinner(f"Loading {len(selected_runs)} checkpoint(s) on {device}..."):
        loaded = {run_dir: load_run(run_dir) for run_dir in selected_runs}
    return (loaded,)


@app.cell
def _(device, np, torch):
    def normalize(array, metadata):
        """HWC [0, 1] canvas to a normalized CHW tensor, as in ManifestImageDataset."""
        normalization = metadata["normalization"]
        mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        stdev = torch.tensor(np.maximum(normalization["stdev"], 1e-6), dtype=torch.float32).view(
            3, 1, 1
        )
        return (torch.from_numpy(array).permute(2, 0, 1) - mean) / stdev

    def predict(loaded, arrays, labels):
        """Per-model softmax over `labels` only, for a list of preprocessed canvases.

        Returns `{run_dir: (images, len(labels))}`. Restricting to the target source's labels
        matches `train evaluate` (e.g. a three-class model scored as Normal/AMD on PAIMA).
        """
        outputs = {}
        for run_dir, (model, metadata) in loaded.items():
            indices = [metadata["class_labels"].index(label) for label in labels]
            batch = torch.stack([normalize(array, metadata) for array in arrays])
            probabilities = []
            with torch.no_grad():
                for start in range(0, len(batch), 32):
                    logits = model(batch[start : start + 32].to(device))[:, indices]
                    probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            outputs[run_dir] = np.concatenate(probabilities)
        return outputs

    return normalize, predict


@app.cell
def _(mo):
    mo.md("""
    ## Image
    """)
    return


@app.cell
def _(dataset_specs, mo, trained_sources):
    # Default to a source no selected model trained on (the held-out source of a LOSO campaign).
    _held_out = [name for name in dataset_specs if name not in trained_sources]
    source = mo.ui.dropdown(
        options=list(dataset_specs),
        value=(_held_out or list(dataset_specs))[0],
        label="Source",
    )
    upload = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        label="…or upload a B-scan",
    )
    return source, upload


@app.cell
def _(cache, read_jsonl, repository_root):
    @cache
    def source_records(name):
        return tuple(read_jsonl(repository_root / "artifacts" / "splits" / f"{name}.jsonl"))

    return (source_records,)


@app.cell
def _(dataset_specs, mo, source):
    _labels = sorted(label.value for label in dataset_specs[source.value].available_labels)
    label_filter = mo.ui.dropdown(options=["any", *_labels], value="any", label="Class")
    unseen_only = mo.ui.checkbox(
        value=True, label="Only images no selected model trained or validated on"
    )
    return label_filter, unseen_only


@app.cell
def _(
    image_status,
    label_filter,
    np,
    selected_runs,
    source,
    source_records,
    unseen_only,
):
    candidates = []
    for _record in source_records(source.value):
        if label_filter.value != "any" and _record.label.value != label_filter.value:
            continue
        _statuses = [image_status(run_dir, source.value, _record.path) for run_dir in selected_runs]
        if unseen_only.value and any(status in ("train", "val") for status in _statuses):
            continue
        candidates.append(_record)
    # A fixed shuffle so stepping through the index visits mixed patients, not one volume.
    candidate_order = np.random.default_rng(0).permutation(len(candidates))
    return candidate_order, candidates


@app.cell
def _(candidates, label_filter, mo, source, unseen_only, upload):
    mo.stop(
        not candidates and not upload.value,
        mo.vstack(
            [
                mo.hstack([source, label_filter, unseen_only], justify="start", gap=2),
                mo.callout("No images match these filters.", kind="warn"),
                upload,
            ]
        ),
    )
    image_index = mo.ui.number(
        start=0, stop=max(len(candidates) - 1, 0), value=0, label=f"Image (of {len(candidates)})"
    )
    mo.vstack(
        [
            mo.hstack([source, label_filter, unseen_only], justify="start", gap=2),
            mo.hstack([image_index, upload], justify="start", gap=2),
        ]
    )
    return (image_index,)


@app.cell
def _(
    Image,
    PreprocessingSpec,
    candidate_order,
    candidates,
    dataset_specs,
    image_index,
    image_size,
    io,
    padding,
    preprocess_image,
    repository_root,
    upload,
):
    spec = PreprocessingSpec(image_size=image_size, padding=padding)
    if upload.value:
        # An uploaded image has no source, label, or split: it is scored with all model labels.
        selected_record = None
        _opened = Image.open(io.BytesIO(upload.value[0].contents))
        image_title = upload.value[0].name
    else:
        selected_record = candidates[int(candidate_order[image_index.value])]
        _opened = Image.open(
            repository_root / dataset_specs[selected_record.source].root / selected_record.path
        )
        image_title = f"{selected_record.source}: {selected_record.path}"
    with _opened:
        _opened.load()
        raw_image = _opened.copy()
    model_input = preprocess_image(raw_image, spec)
    return image_title, model_input, raw_image, selected_record, spec


@app.cell
def _(Image, io, np):
    def png(array_or_image):
        image = array_or_image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.clip(np.rint(image * 255), 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    return (png,)


@app.cell
def _(
    image_status,
    image_title,
    mo,
    model_input,
    padding,
    pd,
    png,
    raw_image,
    selected_record,
    selected_runs,
    spec,
):
    if selected_record is None:
        status_table = pd.DataFrame(
            {"model": [run_dir.name for run_dir in selected_runs], "split": "uploaded"}
        )
        _facts = "Uploaded image: no ground truth."
    else:
        status_table = pd.DataFrame(
            {
                "model": [run_dir.name for run_dir in selected_runs],
                "split": [
                    image_status(run_dir, selected_record.source, selected_record.path)
                    for run_dir in selected_runs
                ],
            }
        )
        _facts = (
            f"Label **{selected_record.label.value}** ({selected_record.raw_label}), "
            f"labelled per {selected_record.label_unit}; group `{selected_record.group_id}`."
        )
    mo.vstack(
        [
            mo.md(f"### `{image_title}`\n\n{_facts}"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"Original, {raw_image.width}×{raw_image.height}"),
                            mo.image(png(raw_image), width=360),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(f"Model input, {spec.image_size}² `{padding}` padding"),
                            mo.image(png(model_input), width=224),
                        ]
                    ),
                    mo.ui.table(status_table, selection=None, pagination=False),
                ],
                justify="start",
                gap=2,
                align="start",
            ),
        ]
    )
    return (status_table,)


@app.cell
def _(mo):
    restrict_labels = mo.ui.checkbox(
        value=True, label="Restrict to the source's labels (as in the reported metrics)"
    )
    return (restrict_labels,)


@app.cell
def _(dataset_specs, loaded, restrict_labels, selected_record):
    _model_labels = next(iter(loaded.values()))[1]["class_labels"]
    if selected_record is None or not restrict_labels.value:
        class_labels = list(_model_labels)
    else:
        _available = {
            label.value for label in dataset_specs[selected_record.source].available_labels
        }
        class_labels = [label for label in _model_labels if label in _available]
    return (class_labels,)


@app.cell
def _(class_labels, loaded, model_input, np, predict, status_table):
    _outputs = predict(loaded, [model_input], class_labels)
    fold_probabilities = {run_dir: probs[0] for run_dir, probs in _outputs.items()}
    ensemble_probabilities = np.mean(list(fold_probabilities.values()), axis=0)
    ensemble_prediction = class_labels[int(ensemble_probabilities.argmax())]
    _split = dict(zip(status_table["model"], status_table["split"], strict=True))
    prediction_rows = [
        {
            "model": run_dir.name,
            "split": _split[run_dir.name],
            **{label: round(float(p), 4) for label, p in zip(class_labels, probs, strict=True)},
            "prediction": class_labels[int(probs.argmax())],
        }
        for run_dir, probs in fold_probabilities.items()
    ]
    prediction_rows.append(
        {
            "model": "ensemble (mean)",
            "split": "",
            **{
                label: round(float(p), 4)
                for label, p in zip(class_labels, ensemble_probabilities, strict=True)
            },
            "prediction": ensemble_prediction,
        }
    )
    return (
        ensemble_prediction,
        ensemble_probabilities,
        fold_probabilities,
        prediction_rows,
    )


@app.cell
def _(
    alt,
    class_labels,
    ensemble_prediction,
    ensemble_probabilities,
    fold_probabilities,
    mo,
    pd,
    prediction_rows,
    restrict_labels,
    selected_record,
    status_table,
):
    _bars = pd.DataFrame({"class": class_labels, "probability": ensemble_probabilities})
    _points = pd.DataFrame(
        [
            {"class": label, "probability": float(p), "model": run_dir.name}
            for run_dir, probs in fold_probabilities.items()
            for label, p in zip(class_labels, probs, strict=True)
        ]
    )
    _y = alt.Y("class:N", sort=class_labels, title=None)
    _chart = (
        alt.Chart(_bars)
        .mark_bar(color="#2a78d6", cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Ensemble probability"),
            y=_y,
            tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
        )
        + alt.Chart(_points)
        .mark_tick(color="#52514e", thickness=2, size=22)
        .encode(
            x="probability:Q",
            y=_y,
            tooltip=["model", "class", alt.Tooltip("probability:Q", format=".3f")],
        )
    ).properties(width=420, height=36 * len(class_labels))

    _seen = (status_table["split"].isin(["train", "val"])).sum()
    _notes = []
    if _seen:
        _notes.append(
            mo.callout(
                f"{_seen} of {len(status_table)} models trained or validated on this image; "
                "their predictions are not out-of-sample.",
                kind="warn",
            )
        )
    _truth = "" if selected_record is None else f" · truth **{selected_record.label.value}**"
    mo.vstack(
        [
            mo.md(f"## Prediction\n\nEnsemble predicts **{ensemble_prediction}**{_truth}."),
            *_notes,
            restrict_labels,
            mo.md("Bars: ensemble mean. Ticks: individual models."),
            _chart,
            mo.ui.table(pd.DataFrame(prediction_rows), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Grad-CAM

    `layer4` Grad-CAM for the ensemble's predicted class, per model and averaged (each map scaled
    to a peak of 1 first). Duke-only models are known to put much of their attention on the
    letterbox padding of other sources, so the table reports the share of attention mass on the
    padding and on detected acquisition fill (e.g. Kermany's white rotation wedges).
    """)
    return


@app.cell
def _(
    GradCam,
    content_mask,
    device,
    ensemble_prediction,
    fill_mask,
    loaded,
    mass_fraction,
    model_input,
    normalize,
    np,
    raw_image,
    spec,
    torch,
    upsample,
):
    _content = content_mask(raw_image.width, raw_image.height, spec.image_size)
    _fill = fill_mask(model_input.mean(axis=2), _content)
    fold_cams = {}
    cam_rows = []
    for _run_dir, (_model, _metadata) in loaded.items():
        _gradcam = GradCam(_model, ["layer4"])
        try:
            _target = torch.tensor(
                [_metadata["class_labels"].index(ensemble_prediction)], device=device
            )
            _cams, _ = _gradcam(normalize(model_input, _metadata)[None].to(device), _target)
        finally:
            _gradcam.close()
        _cam = upsample(_cams["layer4"], spec.image_size)[0].cpu().numpy()
        _cam = _cam / _cam.max() if _cam.max() > 0 else _cam
        fold_cams[_run_dir.name] = _cam
        _padding_mass = mass_fraction(_cam, ~_content)
        _fill_mass = mass_fraction(_cam, _fill)
        cam_rows.append(
            {
                "model": _run_dir.name,
                "padding mass": None if _padding_mass is None else round(_padding_mass, 3),
                "fill mass": None if _fill_mass is None else round(_fill_mass, 3),
            }
        )
    mean_cam = np.mean(list(fold_cams.values()), axis=0)
    padding_area = float((~_content).mean())
    return cam_rows, fold_cams, mean_cam, padding_area


@app.cell
def _(model_input, np):
    def overlay(cam):
        """Grayscale model input with attention in a single hue (orange), alpha by strength."""
        gray = np.repeat(model_input.mean(axis=2, keepdims=True), 3, axis=2)
        color = np.array([0xEB, 0x68, 0x34], dtype=np.float32) / 255
        alpha = 0.65 * np.clip(cam, 0, 1)[..., None]
        return gray * (1 - alpha) + color * alpha

    return (overlay,)


@app.cell
def _(
    cam_rows,
    fold_cams,
    mean_cam,
    mo,
    overlay,
    padding_area,
    pd,
    png,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("Mean over models"), mo.image(png(overlay(mean_cam)), width=300)]
                    ),
                    mo.vstack(
                        [
                            mo.ui.table(pd.DataFrame(cam_rows), selection=None, pagination=False),
                            mo.md(
                                f"Padding is {100 * padding_area:.1f}% of the canvas; a padding "
                                "mass well above that means the model is looking at the frame."
                            ),
                        ]
                    ),
                ],
                justify="start",
                gap=2,
                align="start",
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"`{name.rsplit('-', 1)[-1]}`"),
                            mo.image(png(overlay(cam)), width=150),
                        ]
                    )
                    for name, cam in fold_cams.items()
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(mo, selected_record):
    mo.stop(selected_record is None)
    score_group = mo.ui.run_button(label=f"Score every B-scan of group {selected_record.group_id}")
    mo.vstack(
        [
            mo.md(
                f"""
    ## Whole {selected_record.label_unit if selected_record.label_unit == "eye" else "group"}

    Labels are per eye for Duke and per volume or patient elsewhere, so a single B-scan can be
    wrong while the eye-level call is right. This averages the ensemble probabilities over every
    manifest image sharing group `{selected_record.group_id}` (eye-level metrics in `train
    evaluate` use the same mean).
    """
            ),
            score_group,
        ]
    )
    return (score_group,)


@app.cell
def _(
    Image,
    class_labels,
    dataset_specs,
    image_status,
    loaded,
    mo,
    np,
    pd,
    predict,
    preprocess_image,
    repository_root,
    score_group,
    selected_record,
    source_records,
    spec,
):
    mo.stop(not score_group.value)
    _group = [
        record
        for record in source_records(selected_record.source)
        if record.group_id == selected_record.group_id
    ][:256]
    _root = repository_root / dataset_specs[selected_record.source].root
    with mo.status.spinner(f"Preprocessing {len(_group)} images..."):
        _arrays = []
        for _record in _group:
            with Image.open(_root / _record.path) as _image:
                _arrays.append(preprocess_image(_image, spec))
    _outputs = predict(loaded, _arrays, class_labels)
    _per_image = np.mean(list(_outputs.values()), axis=0)
    _group_probabilities = _per_image.mean(axis=0)
    _seen = sorted(
        {
            run_dir.name
            for run_dir in loaded
            for record in _group
            if image_status(run_dir, record.source, record.path) in ("train", "val")
        }
    )
    _labels = sorted({record.label.value for record in _group})
    _table = pd.DataFrame(
        {
            "path": [record.path for record in _group],
            "label": [record.label.value for record in _group],
            "prediction": [class_labels[i] for i in _per_image.argmax(axis=1)],
            **{label: _per_image[:, i].round(4) for i, label in enumerate(class_labels)},
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**{len(_group)} B-scans**, label(s) {', '.join(_labels)}. Mean probability: "
                + ", ".join(
                    f"{label} {p:.3f}"
                    for label, p in zip(class_labels, _group_probabilities, strict=True)
                )
                + f" → **{class_labels[int(_group_probabilities.argmax())]}**. "
                + f"Per-image predictions: {_table['prediction'].value_counts().to_dict()}."
            ),
            mo.callout(
                f"Models that trained or validated on part of this group: {', '.join(_seen)}.",
                kind="warn",
            )
            if _seen
            else mo.md(""),
            mo.ui.table(_table, selection=None, page_size=10),
        ]
    )
    return


@app.cell
def _(json, loaded, mo, pd, selected_record):
    mo.stop(selected_record is None)

    def _reported(run_dir, source):
        """Held-out `train evaluate` metrics for `source` if present, else the run's own test split."""
        for path in sorted((run_dir / "evaluations").glob("*/metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("target_source") == source:
                return f"held-out ({path.parent.name})", metrics
        metrics = json.loads((run_dir / "metrics-test.json").read_text(encoding="utf-8"))
        return "test split", metrics["by_source"].get(source)

    _rows = []
    for _run_dir in loaded:
        _kind, _metrics = _reported(_run_dir, selected_record.source)
        if _metrics is None:
            continue
        _rows.append(
            {
                "model": _run_dir.name,
                "scored on": _kind,
                **{
                    name: None if _metrics.get(name) is None else round(100 * _metrics[name], 2)
                    for name in ("macro_f1", "balanced_accuracy", "macro_auroc")
                },
                "eye-level macro_f1": (
                    round(100 * _metrics["eye_level"]["macro_f1"], 2)
                    if "eye_level" in _metrics
                    else None
                ),
            }
        )
    mo.vstack(
        [
            mo.md(
                f"""
    ## Reported performance on {selected_record.source}

    Image-level point estimates from the runs' saved metrics, for context only. They have no
    confidence intervals here: the Duke test split is 6 patients per fold, and the group-bootstrap
    intervals are in `docs/reports/20261010-resnet50-loso-fill-aware.md`.
    """
            ),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False)
            if _rows
            else mo.callout("No saved metrics for this source.", kind="neutral"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
