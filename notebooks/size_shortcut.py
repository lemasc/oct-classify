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
    from oct_classify.data.preprocessing import PreprocessingSpec, padding_fill, preprocess_image

    repository_root = Path(__file__).resolve().parents[1]
    dataset_roots = {
        spec.name: repository_root / spec.root
        for spec in load_dataset_specs(repository_root / "configs" / "datasets.toml")
    }
    features_path = repository_root / "artifacts" / "eda" / "features.csv"
    size_test_path = repository_root / "data" / "analysis" / "kermany-size-test" / "size-test.json"
    loso_predictions = sorted(
        (repository_root / "artifacts" / "runs" / "fused" / "resnet50").glob(
            "loso-kermany-local-20261008-174938-fold*/evaluations/kermany-loso/predictions.csv"
        )
    )
    spec = PreprocessingSpec(image_size=224, padding="fill_aware")
    label_order = ["normal", "amd", "dme"]
    return (
        Image,
        alt,
        dataset_roots,
        features_path,
        io,
        json,
        label_order,
        loso_predictions,
        mo,
        np,
        padding_fill,
        pd,
        preprocess_image,
        size_test_path,
        spec,
    )


@app.cell
def _(features_path, mo, pd):
    mo.stop(
        not features_path.exists(),
        mo.md(
            "Per-image features are missing. Create them with "
            "`uv run python scripts/extract_eda_features.py --workers 12`."
        ),
    )
    features = pd.read_csv(features_path, low_memory=False)
    features["size"] = features["width"].astype(str) + "x" + features["height"].astype(str)
    mo.md(
        """
        # The image-size shortcut, and how to remove it

        Held-out models learn to read an image's **shape** as a label. Duke, a training source for held-out
        Kermany, has two sizes that each carry one label: every 768x496 image is DME and every 1024x496 image
        is AMD. Our preprocessing keeps the shape visible, and no augmentation changes it.

        This notebook shows the problem in four steps: the confound in the data, what the model sees,
        the evidence from experiments 1 and 1b, and a comparison of candidate fixes. Background:
        `docs/cross-device-plan.md`, `docs/reports/20261010-kermany-canvas-breakdown.md` and
        `docs/reports/20261010-kermany-size-test.md`.
        """
    )
    return (features,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. The confound: some sizes carry a single label
    """)
    return


@app.cell
def _(alt, features, label_order, mo):
    _main = features[features["source"].isin(["duke", "kermany", "paima"])]
    _sizes = _main.groupby(["source", "size", "label"]).size().reset_index(name="n")
    _totals = _sizes.groupby(["source", "size"])["n"].transform("sum")
    _sizes = _sizes[_totals >= 0.02 * _sizes.groupby("source")["n"].transform("sum")]
    _chart = (
        alt.Chart(_sizes)
        .mark_bar()
        .encode(
            y=alt.Y("size:N", title="width x height", sort=["512x512", "512x496", "768x496", "1024x496", "1536x496"]),
            x=alt.X("n:Q", stack="normalize", title="share of images at this size"),
            color=alt.Color("label:N", sort=label_order, title="label"),
            row=alt.Row("source:N", title=None),
            tooltip=["source", "size", "label", "n"],
        )
        .properties(width=420, height=alt.Step(22), title="Label mix at each image size")
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                - **Duke:** 768x496 is DME only, 1024x496 is AMD only.
                - **Kermany:** 512x512 is DME only.
                - **PAIMA:** all 768x496, normal and AMD. PAIMA has no DME labels, and the masked loss
                  never trains the DME output on PAIMA images, so PAIMA cannot teach the model that a
                  768x496 image is not DME.

                OCTDL (not shown) uses its own sizes, shared with no other source.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. What the model sees

    `preprocess_image` letterboxes every image into a 224x224 square: it scales the width to 224 px and fills
    the rest with one constant value. A wider image gets thicker bands and a smaller retina. The bands are
    perfectly flat, so their height is easy for a CNN to measure, even though fill-aware padding gives them
    the background's brightness.

    Below: one Kermany 512x496 image as it is, and the same image mirror-padded to Duke's two
    single-label widths (the manipulation used in experiment 1b), next to real Duke images of those sizes.
    """)
    return


@app.cell
def _(mo):
    view_label = mo.ui.dropdown(options=["normal", "amd", "dme"], value="normal", label="Kermany label")
    view_seed = mo.ui.slider(0, 100, value=0, label="Random draw", full_width=True)
    mo.hstack([view_label, view_seed], justify="start")
    return view_label, view_seed


@app.cell
def _(Image, io, np):
    def to_png(array):
        _buffer = io.BytesIO()
        Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(_buffer, format="PNG")
        return _buffer.getvalue()

    def mirror_pad(image, width):
        _array = np.asarray(image.convert("RGB"))
        _extra = width - _array.shape[1]
        return Image.fromarray(
            np.pad(_array, [(0, 0), (_extra // 2, _extra - _extra // 2), (0, 0)], mode="reflect")
        )

    return mirror_pad, to_png


@app.cell
def _(
    Image,
    dataset_roots,
    features,
    mirror_pad,
    mo,
    preprocess_image,
    spec,
    to_png,
    view_label,
    view_seed,
):
    _kermany = features[
        (features["source"] == "kermany") & (features["size"] == "512x496") & (features["label"] == view_label.value)
    ].sample(1, random_state=view_seed.value).iloc[0]
    _duke = features[features["source"] == "duke"]
    _duke768 = _duke[_duke["size"] == "768x496"].sample(1, random_state=view_seed.value).iloc[0]
    _duke1024 = _duke[_duke["size"] == "1024x496"].sample(1, random_state=view_seed.value).iloc[0]

    def _tile(title, image):
        return mo.vstack([mo.md(title), mo.image(src=to_png(preprocess_image(image, spec)), width=240)])

    with Image.open(dataset_roots["kermany"] / _kermany["path"]) as _image:
        _image.load()
        _row1 = [
            _tile(f"**Kermany {view_label.value}, 512x496 as is**", _image),
            _tile("**same image, padded to 768 wide**", mirror_pad(_image, 768)),
            _tile("**same image, padded to 1024 wide**", mirror_pad(_image, 1024)),
        ]
    with Image.open(dataset_roots["duke"] / _duke768["path"]) as _a, Image.open(
        dataset_roots["duke"] / _duke1024["path"]
    ) as _b:
        _row2 = [
            mo.md(" "),
            _tile("**real Duke 768x496 (always DME)**", _a),
            _tile("**real Duke 1024x496 (always AMD)**", _b),
        ]
    mo.vstack(
        [
            mo.hstack(_row1, justify="start"),
            mo.hstack(_row2, justify="start"),
            mo.md(f"`{_kermany['path']}`"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Evidence that the model reads the shape

    ### Experiment 1: held-out Kermany errors by image size
    Fill-aware LOSO-Kermany models (trained on Duke, OCTDL and PAIMA), predictions pooled over the 5 models.
    """)
    return


@app.cell
def _(alt, features, label_order, loso_predictions, mo, pd):
    mo.stop(
        not loso_predictions,
        mo.md("_LOSO-Kermany predictions are missing under `artifacts/runs/fused/resnet50/`._"),
    )
    _k = features[features["source"] == "kermany"][["path", "size"]]
    _pred = pd.concat([pd.read_csv(_p) for _p in loso_predictions]).merge(_k, on="path")
    _pred = _pred[_pred["size"].isin(["512x512", "512x496", "768x496", "1024x496", "1536x496"])]
    _mix = _pred.groupby(["target", "size", "prediction"]).size().reset_index(name="n")
    _mix["share"] = _mix["n"] / _mix.groupby(["target", "size"])["n"].transform("sum")
    _chart = (
        alt.Chart(_mix)
        .mark_bar()
        .encode(
            y=alt.Y("size:N", sort=["512x512", "512x496", "768x496", "1024x496", "1536x496"], title=None),
            x=alt.X("share:Q", stack="normalize", title="share predicted as", axis=alt.Axis(format="%")),
            color=alt.Color("prediction:N", sort=label_order, title="predicted"),
            row=alt.Row("target:N", sort=label_order, title="true class"),
            tooltip=["target", "size", "prediction", "n", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=420, height=alt.Step(22))
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                Normal images are called DME most at 768x496 (Duke's DME-only size) and pushed to AMD most at
                1024x496 (Duke's AMD-only size). DME is missed most at 1536x496, a size no training source has.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Experiment 1b: change only the size, keep the image
    The same models rescored after mirror-padding Kermany 512x496 images to 768 or 1024 wide, or cropping
    wider images to 512. Bars show the change in percentage points, with the paired patient-level 95% CI.
    """)
    return


@app.cell
def _(alt, json, mo, pd, size_test_path):
    mo.stop(not size_test_path.exists(), mo.md(f"_Missing `{size_test_path}`._"))
    _raw = json.loads(size_test_path.read_text())
    _arms = {
        ("pad", "512x496"): "512x496 padded to 768",
        ("pad1024", "512x496"): "512x496 padded to 1024",
        ("crop", "768x496"): "768x496 cropped to 512",
        ("crop", "1024x496"): "1024x496 cropped to 512",
    }
    _rows = []
    for (_arm, _size), _name in _arms.items():
        for _key, (_before, _after, _delta, _low, _high) in _raw[_arm][_size].items():
            _true, _pred = _key.split("->")
            if _true != _pred:
                _rows.append(
                    {"change": _name, "confusion": f"{_true} → {_pred}", "true": _true, "before": _before,
                     "after": _after, "delta": _delta, "low": _low, "high": _high}
                )
    _table = pd.DataFrame(_rows)
    _base = alt.Chart(_table).encode(
        y=alt.Y("confusion:N", title=None),
        tooltip=["change", "confusion", alt.Tooltip("before:Q", format=".1f"),
                 alt.Tooltip("after:Q", format=".1f"), alt.Tooltip("delta:Q", format="+.1f")],
    )
    _chart = (
        alt.layer(
            _base.mark_bar().encode(
                x=alt.X("delta:Q", title="change (percentage points)"),
                color=alt.condition("datum.delta > 0", alt.value("#c0504d"), alt.value("#4f81bd")),
            ),
            _base.mark_rule().encode(x="low:Q", x2="high:Q"),
        )
        .properties(width=300, height=alt.Step(16))
        .facet(column=alt.Column("change:N", sort=list(_arms.values()), title=None))
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                Padding to 768 pushes every class toward DME; padding the **same images** to 1024 pushes them
                toward AMD. The 1024 arm has the same kind of mirror seam and an even smaller retina, so neither
                explains the opposite shifts: the model reads the shape. Cropping 768x496 images removes most
                of their DME pull.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Candidate fixes

    Each candidate is a change to how an image becomes the 224x224 input. All of them keep fill-aware
    intensity scaling. "Train" is what a training image looks like; "test" is the deterministic version.

    | Fix | Train | Test | Padding bands | Tissue scale across sizes | Lateral view kept |
    |---|---|---|---|---|---|
    | **Letterbox** (current) | whole image, aspect kept | same | yes, height reveals width | differs (224 / width) | all |
    | **Stretch** | whole image squeezed to a square | same | none | vertical same, horizontal squeezed by width | all |
    | **Random-width crop** | random crop of width between the height and the full width, then letterbox | whole image, letterbox | yes, but random height | varies with crop width | random part |
    | **Square crop** | random height x height window | several windows, predictions averaged | none | same for all 496-high images | height / width |
    | **Random flat padding** | letterbox as if the image were a random width between its own and 3.2x its height | whole image, letterbox | yes, random height | varies with the random width | all |
    | **Random mirror padding** | mirror the tissue sideways to a random width between its own and 3.2x its height, then letterbox | whole image, letterbox | yes, random height | varies with the random width | all, plus mirrored copies |
    | **Mirror padding, capped** | as above, but add at most a set share of the image's own width (slider below) | whole image, letterbox | yes, random height | varies, within the cap | all, plus smaller mirrored copies |

    The two padding fixes keep every pixel of tissue, so no lesion is ever cut away. Flat padding only shrinks
    the image inside a larger flat frame. Mirror padding fills the extra width with mirrored tissue, so the
    tissue itself takes a random shape (the same operation experiment 1b used to move predictions). Mirroring
    can duplicate a lesion and leaves a seam where the copy starts. Padding can only make an image wider, so
    a wide image can never be shown narrow.

    Experiment 1c found that doubling an image with mirrored tissue pushes current held-out models toward AMD
    on its own (PAIMA padded from 768 to 1536: normal called AMD 5% → 33%). The capped variant adds at most a
    set share of the image's own width (slider below), so mirrored copies stay smaller. The cost: with a 50% cap
    a 512 image can reach 768 but not 1024, so 1024 cannot become a shape every label shares.

    Pick an image to see each fix. Rerun the random draw to see different random crops and paddings.
    """)
    return


@app.cell
def _(mo):
    mirror_cap = mo.ui.slider(
        0.25, 2.0, value=0.5, step=0.25, label="Mirror-padding cap (share of the image's own width)", full_width=True
    )
    mirror_cap
    return (mirror_cap,)


@app.cell
def _(Image, mirror_cap, np, padding_fill, preprocess_image, spec):
    def fix_letterbox(image, rng):
        return preprocess_image(image, spec)

    def fix_stretch(image, rng):
        return preprocess_image(image.resize((spec.image_size, spec.image_size), Image.Resampling.BILINEAR), spec)

    def fix_random_width(image, rng):
        _w, _h = image.size
        _crop = int(rng.integers(min(_w, _h), _w + 1))
        _left = int(rng.integers(0, _w - _crop + 1))
        return preprocess_image(image.crop((_left, 0, _left + _crop, _h)), spec)

    def fix_square(image, rng):
        _w, _h = image.size
        _side = min(_w, _h)
        _left = int(rng.integers(0, _w - _side + 1))
        _top = int(rng.integers(0, _h - _side + 1))
        return preprocess_image(image.crop((_left, _top, _left + _side, _top + _side)), spec)

    # Widest native aspect among Duke, Kermany and PAIMA is 1536x496 (3.1:1); padding goes up to about that.
    max_aspect = 3.2

    # Exp. 1c: doubling an image with mirrored tissue pushes held-out models toward AMD on its own, so the capped
    # variant adds at most `mirror_cap` times the image's own width.
    def _random_target_width(image, rng, cap=None):
        _w, _h = image.size
        _high = max(_w, max_aspect * _h)
        if cap is not None:
            _high = min(_high, (1 + cap) * _w)
        return rng.uniform(_w, _high)

    def fix_random_flat_pad(image, rng):
        # Letterbox as if the image were wider: the tissue shrinks and flat padding surrounds it.
        _w, _h = image.size
        _array = preprocess_image(image, spec)
        _size = spec.image_size
        _shrunk = max(1, round(_size * max(_w, _h) / max(_random_target_width(image, rng), _h)))
        _small = np.stack(
            [np.asarray(Image.fromarray(_array[..., _c]).resize((_shrunk, _shrunk), Image.Resampling.BILINEAR))
             for _c in range(3)],
            -1,
        )
        _canvas = np.empty_like(_array)
        _canvas[:] = padding_fill(_array)
        _offset = (_size - _shrunk) // 2
        _canvas[_offset : _offset + _shrunk, _offset : _offset + _shrunk] = _small
        return _canvas

    def fix_random_mirror_pad(image, rng, cap=None):
        # Extend the tissue sideways by mirroring it, so the tissue itself takes a random width.
        _array = np.asarray(image)
        _extra = int(round(_random_target_width(image, rng, cap))) - _array.shape[1]
        _left = int(rng.integers(0, _extra + 1))
        _padded = np.pad(_array, [(0, 0), (_left, _extra - _left)] + [(0, 0)] * (_array.ndim - 2), mode="reflect")
        return preprocess_image(Image.fromarray(_padded), spec)

    fixes = {
        "letterbox (current)": fix_letterbox,
        "stretch": fix_stretch,
        "random-width crop": fix_random_width,
        "square crop": fix_square,
        "random flat padding": fix_random_flat_pad,
        "random mirror padding": fix_random_mirror_pad,
        f"mirror padding, cap {mirror_cap.value:.0%}": lambda image, rng: fix_random_mirror_pad(image, rng, mirror_cap.value),
    }
    return (fixes,)


@app.cell
def _(mo):
    gallery_seed = mo.ui.slider(0, 100, value=0, label="Random draw (images and crops)", full_width=True)
    gallery_seed
    return (gallery_seed,)


@app.cell
def _(Image, dataset_roots, features, fixes, gallery_seed, mo, np, to_png):
    _examples = [
        ("duke", "512x496", "normal"),
        ("duke", "768x496", "dme"),
        ("duke", "1024x496", "amd"),
        ("kermany", "512x512", "dme"),
        ("kermany", "768x496", "normal"),
        ("kermany", "1536x496", "normal"),
        ("paima", "768x496", "amd"),
        ("octdl", None, "amd"),
    ]
    _rng = np.random.default_rng(gallery_seed.value)
    _header = [mo.md("**image**")] + [mo.md(f"**{_name}**") for _name in fixes]
    _rows = [mo.hstack(_header, widths=[1] * len(_header))]
    for _source, _size, _label in _examples:
        _pool = features[(features["source"] == _source) & (features["label"] == _label)]
        if _size:
            _pool = _pool[_pool["size"] == _size]
        _row = _pool.sample(1, random_state=gallery_seed.value).iloc[0]
        with Image.open(dataset_roots[_source] / _row["path"]) as _image:
            _image = _image.convert("RGB")
            _cells = [mo.md(f"{_source}<br>{_row['size']}<br>{_label}")]
            _cells += [mo.image(src=to_png(_fix(_image, _rng)), width=150) for _fix in fixes.values()]
        _rows.append(mo.hstack(_cells, widths=[1] * len(_cells)))
    mo.vstack(_rows)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Shape probe: how much size information does each fix leave?

    Take Kermany 1536x496 scans and center-crop each to four widths (512, 768, 1024, 1536). The four versions
    are the same scan at the same pixel scale, so they differ **only** in width. Pass each version through a
    fix (training mode) and train a small logistic regression on the 224x224 input to guess which width it
    came from. Cross-validation is grouped by scan.

    Accuracy near 25% (chance) means the fix hides width. Accuracy near 100% means a model could still learn
    "this shape means this label". This measures whether the cue is **available**; only training (exp. 3a)
    shows whether removing it closes the gap.

    Two limits of the design:

    - Wider versions contain more peripheral retina. A random window in the 1536 version can land on the
      periphery, which the 512 version never shows, so a crop can score above chance from **where** the
      window falls rather than from shape.
    - Padding fixes can only widen an image, so the 1536 version is always recognizable to them. That
      matters little for our data: no source has a single-label 1536 size. The harmful sizes are Duke's
      768x496 (DME only) and 1024x496 (AMD only), so look at those rows of the per-width table.

    The probe preprocesses a few thousand images and takes about a minute.
    """)
    return


@app.cell
def _(mo):
    probe_scans = mo.ui.slider(50, 400, value=150, step=50, label="Scans", full_width=True)
    probe_run = mo.ui.run_button(label="Run shape probe")
    mo.hstack([probe_scans, probe_run], justify="start")
    return probe_run, probe_scans


@app.cell
def _(
    Image,
    dataset_roots,
    features,
    fixes,
    mo,
    np,
    pd,
    probe_run,
    probe_scans,
):
    mo.stop(not probe_run.value, mo.md("_Press **Run shape probe** to compute._"))
    from concurrent.futures import ThreadPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _widths = [512, 768, 1024, 1536]
    _scans = features[(features["source"] == "kermany") & (features["size"] == "1536x496")].sample(
        probe_scans.value, random_state=0
    )

    def _probe_features(array):
        # Row and column profiles catch band height and tissue extent; an 8x8-pooled thumbnail catches layout.
        _gray = array.mean(-1)
        _thumb = _gray.reshape(28, 8, 28, 8).mean((1, 3)).ravel()
        return np.concatenate([_gray.mean(1), _gray.std(1), _gray.std(0), _thumb])

    def _versions(job):
        _index, _path = job
        _rng = np.random.default_rng(_index)
        _out = []
        with Image.open(dataset_roots["kermany"] / _path) as _image:
            _image = _image.convert("RGB")
            for _width in _widths:
                _left = (1536 - _width) // 2
                _version = _image.crop((_left, 0, _left + _width, 496))
                for _name, _fix in fixes.items():
                    _out.append((_name, _index, _width, _probe_features(_fix(_version, _rng))))
        return _out

    with mo.status.spinner(title="Preprocessing probe images"):
        with ThreadPoolExecutor(max_workers=12) as _pool:
            _results = [_r for _batch in _pool.map(_versions, enumerate(_scans["path"])) for _r in _batch]

    _rows, _per_width = [], []
    for _name in fixes:
        _subset = [_r for _r in _results if _r[0] == _name]
        _x = np.stack([_r[3] for _r in _subset])
        _y = np.array([_r[2] for _r in _subset])
        _groups = np.array([_r[1] for _r in _subset])
        _model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        _folds = list(GroupKFold(5).split(_x, _y, _groups))
        _pred = cross_val_predict(_model, _x, _y, cv=_folds)
        _scores = [(_pred[_test] == _y[_test]).mean() for _, _test in _folds]
        _rows.append({"fix": _name, "accuracy": np.mean(_scores), "low": min(_scores), "high": max(_scores)})
        _per_width.append(
            {"fix": _name, **{f"{_w} wide": (_pred[_y == _w] == _w).mean() for _w in _widths}}
        )
    probe_table = pd.DataFrame(_rows)
    probe_per_width = pd.DataFrame(_per_width)
    return probe_per_width, probe_table


@app.cell
def _(alt, fixes, mo, probe_per_width, probe_table):
    _base = alt.Chart(probe_table).encode(y=alt.Y("fix:N", sort=list(fixes), title=None))
    _chart = alt.layer(
        _base.mark_bar().encode(
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                    title="probe accuracy at guessing the width (chance 25%)"),
            tooltip=["fix", alt.Tooltip("accuracy:Q", format=".1%"), alt.Tooltip("low:Q", format=".1%"),
                     alt.Tooltip("high:Q", format=".1%")],
        ),
        _base.mark_rule().encode(x="low:Q", x2="high:Q"),
        alt.Chart(probe_table).mark_rule(strokeDash=[4, 4], color="gray").encode(x=alt.datum(0.25)),
    ).properties(width=420, height=alt.Step(28))
    mo.vstack(
        [
            _chart,
            mo.md("Whiskers span the five cross-validation folds."),
            mo.md(
                "**Per width:** share of each version the probe recognized correctly. Lower is better; "
                "768 and 1024 are the sizes Duke ties to one label."
            ),
            mo.ui.table(probe_per_width.round(2), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Reading the comparison

    A 150-scan probe run (2026-10-10). Overall accuracy, then the share recognized at each width
    (chance 25%). The random fixes vary by a few points between runs.

    | Fix | Overall | 512 | 768 | 1024 | 1536 |
    |---|---:|---:|---:|---:|---:|
    | letterbox (current) | 99% | 99% | 99% | 99% | 100% |
    | random flat padding | 97% | 97% | 96% | 97% | 99% |
    | random-width crop | 62% | 85% | 61% | 45% | 57% |
    | random mirror padding | 63% | 55% | 34% | 64% | 98% |
    | mirror padding, cap 50% | 90% | 94% | 81% | 86% | 99% |
    | mirror padding, cap 100% | 75% | 83% | 50% | 69% | 98% |
    | mirror padding, cap 200% | 65% | 53% | 43% | 67% | 96% |
    | stretch | 58% | 66% | 29% | 54% | 82% |
    | square crop | 38% | 51% | 23% | 33% | 45% |

    - **Letterbox** leaves the width in plain view (band height), which is the cue experiment 1b found.
    - **Random flat padding** barely helps. It hides band height, but the tissue keeps its own
      width-to-height ratio inside the flat frame, and the probe reads that instead.
    - **Random mirror padding** keeps every pixel of tissue and blurs 768 and 1024, the two sizes Duke ties
      to one label, about as well as the crops do. It cannot hide 1536, which no source ties to a label.
      Its costs are mirrored lesion copies and seams; both appear in every class, so they should not become
      label cues, but a model may still spend capacity on them.
    - **Capping the mirroring gives the hiding back.** A 50% cap leaves 768 and 1024 recognizable 81-86% of the
      time, close to the letterbox. Hiding needs widths that overlap across sizes, and only large mirroring gets
      there: a 100% cap (at most doubling) reaches 50% at 768 and 69% at 1024. The AMD shift from heavy mirroring
      in experiment 1c was measured on models that never saw mirrored images in training; training with mirroring
      in every class is what should remove it. So a cap protects little and costs most of the benefit. Check the
      trained model with the 1c pad test instead.
    - **Stretch** removes the bands, but squeezes tissue horizontally by an amount that depends on the
      width, so the shape is still readable from the tissue texture. It also distorts retinal thickness,
      the main DME sign, differently per source.
    - **Random-width crop** makes band height vary within each size, which breaks a fixed "this band means
      this label" rule during training. The test image is still letterboxed, and tissue scale still follows
      the crop width.
    - **Square crop** shows every 496-high image at the same scale with no bands. It gives up lateral
      context (two thirds of a 1536-wide scan per window), which multi-window averaging at test time has to
      recover. OCTDL images are 211-412 px high, so they would be shown at a different scale unless
      resized to a common height first.

    **Cropping can cut lesions away.** Labels describe the whole B-scan, so a crop that misses the fluid or
    drusen still carries the disease label and teaches the model a wrong example. The gallery shows a Duke
    AMD crop that misses the drusen. Padding fixes avoid this; crops trade it for hiding the shape better.

    The probe says which fix hides the cue; experiment 3a has to show which one closes the gap without
    costing in-domain accuracy. Random mirror padding is the candidate that hides the harmful sizes while
    keeping all tissue; square crop is the strongest hider and the natural comparison arm.
    """)
    return


if __name__ == "__main__":
    app.run()
