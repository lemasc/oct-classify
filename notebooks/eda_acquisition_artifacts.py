import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image

    from oct_classify.data.config import load_dataset_specs
    from oct_classify.data.eda import static_overlay_maps
    from oct_classify.data.fill import fill_masks
    from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image

    repository_root = Path(__file__).resolve().parents[1]
    dataset_roots = {
        spec.name: repository_root / spec.root
        for spec in load_dataset_specs(repository_root / "configs" / "datasets.toml")
    }
    features_path = repository_root / "artifacts" / "eda" / "features.csv"
    return (
        Image,
        PreprocessingSpec,
        alt,
        dataset_roots,
        features_path,
        fill_masks,
        io,
        mo,
        np,
        pd,
        preprocess_image,
        static_overlay_maps,
    )


@app.cell
def _(features_path, mo, pd):
    mo.stop(
        not features_path.exists(),
        mo.md(
            "Per-image features are missing. Create them (about 6 minutes on 14 workers) with "
            "`uv run python scripts/extract_eda_features.py --workers 12`."
        ),
    )
    features = pd.read_csv(features_path, low_memory=False)
    features["size"] = features["width"].astype(str) + "x" + features["height"].astype(str)
    features["has_white"] = features["white_frac"] > 0.005
    features["has_black"] = features["black_frac"] > 0.005
    # Fraction of the model's square canvas that holds image content after aspect-preserving resize.
    features["canvas_fill"] = 1 / features[["aspect"]].assign(inv=1 / features["aspect"]).max(
        axis=1
    )
    # Where the content's top percentile lands after whole-canvas percentile normalization.
    features["content_ceiling"] = (features["p99_content"] - features["p1_all"]) / (
        features["p99_all"] - features["p1_all"]
    ).clip(lower=1)
    features["content_ceiling"] = features["content_ceiling"].clip(upper=1)
    source_order = ["duke", "kermany", "octdl", "paima"]
    label_order = ["normal", "amd", "dme"]
    mo.md(
        f"""
        # Acquisition artifacts and shortcut risk: an EDA beyond deduplication

        Source: `{features_path.name}`, {len(features):,} images from the derived manifests
        (post-deduplication), measured at **native resolution, before** resizing.
        "White fill" and "black fill" mean saturated (>=250) or empty (<=3) regions that touch the image
        border and are at least 300 px, i.e. rotation/crop padding rather than speckle or anatomy.

        **Why it matters here.** `preprocess_image` scales intensities with the 1st/99th percentile of the
        whole canvas. White fill pins the 99th percentile near 255, so the anatomy is compressed in some
        images and not in others. Sources and even single classes differ in whether they have fill, in
        canvas shape, and in overlays. Each is a cue a model can use to identify the source or the label
        instead of the pathology.
        """
    )
    return features, label_order, source_order


@app.cell
def _(mo):
    mo.md("""
    ## 1. Composition: what is actually in the data
    """)
    return


@app.cell
def _(alt, features, label_order, mo, source_order):
    _counts = (
        features.groupby(["source", "label", "supplied_split"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    _counts["supplied_split"] = _counts["supplied_split"].fillna("none")
    _chart = (
        alt.Chart(_counts)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=label_order, title=None),
            y=alt.Y("n:Q", title="images"),
            color=alt.Color("supplied_split:N", title="supplied split"),
            column=alt.Column("source:N", sort=source_order, title=None),
            tooltip=["source", "label", "supplied_split", "n"],
        )
        .properties(width=140, height=180, title="Images per source, label and supplied split")
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                "Duke (3.2k) and OCTDL (1.7k) are tiny next to Kermany (101k); OCTDL has only 147 DME images."
            ),
        ]
    )
    return


@app.cell
def _(alt, features, label_order, mo, source_order):
    _sizes = features.groupby(["source", "label", "size"]).size().reset_index(name="n")
    _sizes["share"] = _sizes["n"] / _sizes.groupby(["source", "label"])["n"].transform("sum")
    # OCTDL has hundreds of near-unique sizes; bucket the tail so that the chart stays readable.
    _top = _sizes.groupby(["source", "size"])["n"].sum().reset_index()
    _top = _top[_top["n"] >= 100][["source", "size"]]
    _sizes = _sizes.merge(_top.assign(keep=True), how="left", on=["source", "size"])
    _sizes["size"] = _sizes["size"].where(_sizes["keep"].fillna(False), "other")
    _sizes = _sizes.groupby(["source", "label", "size"], as_index=False)[["n"]].sum()
    _sizes["share"] = _sizes["n"] / _sizes.groupby(["source", "label"])["n"].transform("sum")
    _chart = (
        alt.Chart(_sizes)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=label_order, title=None),
            y=alt.Y("share:Q", title="share of class", stack="normalize"),
            color=alt.Color("size:N", title="width x height"),
            column=alt.Column("source:N", sort=source_order, title=None),
            tooltip=["source", "label", "size", "n", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=140, height=200, title="Image size mix inside each class")
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                **The canvas shape alone separates classes.**

                - **Duke:** Normal is 100% 512x496. DME is 512 or 768 wide. AMD is 512 or 1024 wide.
                - **Kermany:** 74% of DME is 512x512, and that size exists in no other class.
                - **PAIMA:** one size (768x496), so no cue.
                - **OCTDL:** about 2.8:1 wide images of many sizes, nothing like the other sources.
                """
            ),
        ]
    )
    return


@app.cell
def _(features, mo):
    _k = features[features["source"] == "kermany"]
    _geometry = (
        _k.assign(is_test=_k["supplied_split"] == "test")
        .groupby("size")
        .agg(images=("path", "size"), test_share=("is_test", "mean"))
        .sort_values("images", ascending=False)
        .reset_index()
    )
    _test = (
        _k[_k["supplied_split"] == "test"]["size"]
        .value_counts(normalize=True)
        .rename("share of test")
    )
    _train = (
        _k[_k["supplied_split"] == "train"]["size"]
        .value_counts(normalize=True)
        .rename("share of train")
    )
    _mix = (
        _train.to_frame()
        .join(_test, how="outer")
        .fillna(0)
        .round(3)
        .sort_values("share of train", ascending=False)
        .reset_index()
        .rename(columns={"index": "size"})
    )
    mo.vstack(
        [
            mo.md(
                """
                ### Kermany's supplied test split has a different geometry mix than train

                Held-out Kermany is therefore not only a new set of patients: it is dominated by 1024-wide
                images, which are rare in train. Any gap between train and test performance is confounded with this.
                """
            ),
            mo.ui.table(_mix, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Look at it: gallery browser

    Pick a subset. The right-hand image of each pair is what `preprocess_image` feeds the network
    (current `black` padding at 224x224), with the **detected fill outlined**: white fill in red,
    black fill in cyan.
    """)
    return


@app.cell
def _(features, label_order, mo, source_order):
    gallery_source = mo.ui.dropdown(options=source_order, value="kermany", label="Source")
    gallery_label = mo.ui.dropdown(options=["any", *label_order], value="dme", label="Label")
    _sizes = sorted(features["size"].unique(), key=lambda s: -int((features["size"] == s).sum()))[
        :12
    ]
    gallery_size = mo.ui.dropdown(options=["any", *_sizes], value="512x512", label="Size")
    gallery_filter = mo.ui.dropdown(
        options=["any", "white fill >5%", "white fill touching top", "black fill >5%", "no fill"],
        value="white fill touching top",
        label="Artifact",
    )
    gallery_seed = mo.ui.slider(0, 50, value=0, label="Random draw")
    gallery_count = mo.ui.slider(2, 8, value=4, label="Pairs")
    mo.hstack(
        [gallery_source, gallery_label, gallery_size, gallery_filter, gallery_seed, gallery_count],
        justify="start",
        wrap=True,
    )
    return (
        gallery_count,
        gallery_filter,
        gallery_label,
        gallery_seed,
        gallery_size,
        gallery_source,
    )


@app.cell
def _(
    Image,
    PreprocessingSpec,
    dataset_roots,
    features,
    fill_masks,
    gallery_count,
    gallery_filter,
    gallery_label,
    gallery_seed,
    gallery_size,
    gallery_source,
    io,
    mo,
    np,
    preprocess_image,
):
    _subset = features[features["source"] == gallery_source.value]
    if gallery_label.value != "any":
        _subset = _subset[_subset["label"] == gallery_label.value]
    if gallery_size.value != "any":
        _subset = _subset[_subset["size"] == gallery_size.value]
    _masks = {
        "white fill >5%": _subset["white_frac"] > 0.05,
        "white fill touching top": _subset["white_touch_top"],
        "black fill >5%": _subset["black_frac"] > 0.05,
        "no fill": ~_subset["has_white"] & ~_subset["has_black"],
    }
    if gallery_filter.value != "any":
        _subset = _subset[_masks[gallery_filter.value]]
    mo.stop(_subset.empty, mo.md("_No images match this combination._"))
    _picked = _subset.sample(
        min(gallery_count.value, len(_subset)), random_state=gallery_seed.value
    )

    def _png(array):
        _buffer = io.BytesIO()
        Image.fromarray(array).save(_buffer, format="PNG")
        return _buffer.getvalue()

    def _outlined(array_rgb, white, black):
        _out = array_rgb.copy()
        for _mask, _color in ((white, (255, 0, 0)), (black, (0, 255, 255))):
            _edge = _mask & ~(
                np.roll(_mask, 1, 0)
                & np.roll(_mask, -1, 0)
                & np.roll(_mask, 1, 1)
                & np.roll(_mask, -1, 1)
            )
            _out[_edge] = _color
        return _out

    _cards = []
    for _, _row in _picked.iterrows():
        with Image.open(dataset_roots[_row["source"]] / _row["path"]) as _image:
            _native = np.asarray(_image.convert("RGB"))
            _model_view = (preprocess_image(_image, PreprocessingSpec()) * 255).astype(np.uint8)
        _white, _black = fill_masks(_native)
        _model_white, _model_black = (
            np.asarray(
                Image.fromarray(_m.astype(np.uint8) * 255).resize(
                    (224, 224), Image.Resampling.NEAREST
                )
            )
            > 0
            for _m in (_white, _black)
        )
        _cards.append(
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.image(src=_png(_outlined(_native, _white, _black)), width=300),
                            mo.image(src=_png(_model_view), width=224),
                        ],
                        align="start",
                    ),
                    mo.md(
                        f"`{_row['path']}`  \n{_row['label']} · {_row['size']} · "
                        f"white {_row['white_frac']:.1%} · black {_row['black_frac']:.1%} · "
                        f"p99 all/content {_row['p99_all']:.0f}/{_row['p99_content']:.0f}"
                    ),
                ]
            )
        )
    mo.vstack(
        [
            mo.md(f"{len(_subset):,} images match; showing {len(_picked)}."),
            mo.hstack(_cards, wrap=True, align="start", gap=1.5),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. White fill (rotation / crop padding)
    """)
    return


@app.cell
def _(alt, features, label_order, mo, source_order):
    _prevalence = (
        features.groupby(["source", "label"])
        .agg(
            any_white=("has_white", "mean"),
            touches_top=("white_touch_top", "mean"),
            n=("path", "size"),
        )
        .reset_index()
        .melt(["source", "label", "n"], var_name="measure", value_name="share")
    )
    _chart = (
        alt.Chart(_prevalence)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=label_order, title=None),
            xOffset="measure:N",
            y=alt.Y("share:Q", title="share of images", axis=alt.Axis(format="%")),
            color=alt.Color("measure:N", title=None),
            column=alt.Column("source:N", sort=source_order, title=None),
            tooltip=["source", "label", "measure", "n", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=140, height=180, title="Images with border-connected white fill")
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                White fill is essentially **a Duke and Kermany property** (about 98% of Duke, about 60% of Kermany) and
                absent from PAIMA and OCTDL. A source-held-out model trained on Duke or Kermany has never seen an
                image without it at train time, or the reverse.
                """
            ),
        ]
    )
    return


@app.cell
def _(alt, features, label_order, mo):
    _data = features[features["source"].isin(["duke", "kermany"]) & features["has_white"]]
    _data = _data.assign(group=_data["source"] + " " + _data["size"])
    _data = _data[_data.groupby("group")["path"].transform("size") >= 200]
    # Exact quartiles over all images; embedding raw rows would make the chart output huge.
    _quartiles = (
        _data.groupby(["group", "label"])["white_frac"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={"25%": "q1", "50%": "median", "75%": "q3", "count": "n"})[
            ["n", "min", "q1", "median", "q3", "max"]
        ]
        .reset_index()
    )
    _base = alt.Chart(_quartiles).encode(
        y=alt.Y("group:N", title=None),
        yOffset="label:N",
        color=alt.Color("label:N", sort=label_order),
        tooltip=["group", "label", "n", "min", "q1", "median", "q3", "max"],
    )
    _title = "fraction of image that is white fill (min, quartiles, max)"
    _axis = alt.Axis(format="%")
    _chart = (
        _base.mark_rule().encode(x=alt.X("min:Q", title=_title, axis=_axis), x2="max:Q")
        + _base.mark_bar(size=8).encode(x=alt.X("q1:Q", title=_title, axis=_axis), x2="q3:Q")
        + _base.mark_tick(color="black", thickness=2, size=8).encode(
            x=alt.X("median:Q", title=_title, axis=_axis)
        )
    ).properties(
        width=500, height=320, title="How much white fill, by size and class (images with fill)"
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                Fill is not a few stray pixels: in Kermany 512x512 DME it is a **median of a few percent and up to about half
                of the frame** (the large white top regions you noticed). 768- and 1536-wide Kermany images
                almost never have it, so the *same* disease is shown with or without it depending on the
                acquisition export.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Worst offenders
    Largest white fill per source, for fast spot checks.
    """)
    return


@app.cell
def _(features, mo, source_order):
    _worst = (
        features.sort_values("white_frac", ascending=False)
        .groupby("source")
        .head(8)
        .sort_values(["source", "white_frac"], ascending=[True, False])
    )[
        [
            "source",
            "label",
            "path",
            "size",
            "white_frac",
            "white_top",
            "white_bottom",
            "white_left",
            "white_right",
        ]
    ]
    _worst = _worst[_worst["source"].isin(source_order)]
    mo.ui.table(_worst, selection=None, page_size=40)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Why it matters: percentile normalization
    `preprocess_image` rescales so the 1st and 99th percentile of the canvas map to 0 and 1.
    With white fill the 99th percentile sits at ~250 regardless of the anatomy, so the **brightest
    tissue ends up well below 1.0**. `content ceiling` below is where the 99th percentile of the *tissue* lands.
    """)
    return


@app.cell
def _(alt, features, mo, np, pd, source_order):
    _bins = np.linspace(0.3, 1.0, 36)
    _rows = []
    for _source in source_order:
        _hist, _ = np.histogram(
            features.loc[features["source"] == _source, "content_ceiling"], bins=_bins
        )
        for _left, _count in zip(_bins[:-1], _hist, strict=True):
            _rows.append(
                {"source": _source, "ceiling": _left, "share": _count / max(_hist.sum(), 1)}
            )
    _chart = (
        alt.Chart(pd.DataFrame(_rows))
        .mark_area(opacity=0.55, interpolate="step")
        .encode(
            x=alt.X(
                "ceiling:Q",
                title="where tissue's 99th percentile lands after normalization (1.0 = full range)",
            ),
            y=alt.Y("share:Q", title="share of images", stack=None),
            color=alt.Color("source:N", sort=source_order),
        )
        .properties(width=620, height=240)
    )
    _summary = (
        features.groupby("source")["content_ceiling"]
        .describe(percentiles=[0.1, 0.5, 0.9])[["10%", "50%", "90%"]]
        .round(2)
        .reset_index()
    )
    mo.vstack(
        [
            _chart,
            mo.ui.table(_summary, selection=None),
            mo.md(
                """
                **Duke tissue is rescaled to a median of 0.73 of the range, while PAIMA and OCTDL tissue spans 1.0.**
                Kermany is split by export format: about 0.75 for 512/1024-wide (and 512x512), but 1.0 for 768/1536-wide images. This is a preprocessing-induced domain shift that applies to the very sources and
                classes that fail in LOSO, and it does not require the model to "see" the white fill at all.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Same image, three preprocessing choices
    Compare the current `black`, the existing `median` padding, and a fill-aware variant
    (percentiles over tissue only, fill grown a few px into the JPEG edge and re-drawn from the nearby background texture) on any image with white fill.
    """)
    return


@app.cell
def _(features, mo):
    _candidates = features[features["white_frac"] > 0.03]
    compare_seed = mo.ui.slider(0, 200, value=3, label="Random draw", full_width=True)
    compare_source = mo.ui.dropdown(
        options=sorted(_candidates["source"].unique()), value="kermany", label="Source"
    )
    mo.hstack([compare_source, compare_seed], justify="start")
    return compare_seed, compare_source


@app.cell
def _(
    Image,
    PreprocessingSpec,
    compare_seed,
    compare_source,
    dataset_roots,
    features,
    io,
    mo,
    np,
    preprocess_image,
):
    _pool = features[(features["source"] == compare_source.value) & (features["white_frac"] > 0.03)]
    mo.stop(_pool.empty, mo.md("_No image with white fill for this source._"))
    _row = _pool.sample(1, random_state=compare_seed.value).iloc[0]

    def _png(array):
        _buffer = io.BytesIO()
        Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(_buffer, format="PNG")
        return _buffer.getvalue()

    with Image.open(dataset_roots[_row["source"]] / _row["path"]) as _image:
        _black = preprocess_image(_image, PreprocessingSpec(padding="black"))
        _median = preprocess_image(_image, PreprocessingSpec(padding="median"))
        _aware = preprocess_image(_image, PreprocessingSpec(padding="fill_aware"))
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**current: black pad**"), mo.image(src=_png(_black), width=300)]
                    ),
                    mo.vstack([mo.md("**median pad**"), mo.image(src=_png(_median), width=300)]),
                    mo.vstack([mo.md("**fill-aware**"), mo.image(src=_png(_aware), width=300)]),
                ],
                justify="start",
            ),
            mo.md(
                f"`{_row['path']}` · {_row['label']} · {_row['size']} · white {_row['white_frac']:.1%} · "
                f"tissue ceiling {_row['content_ceiling']:.2f}"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Canvas geometry after resizing
    The network sees a 224x224 square. How much of it is tissue depends on the source's aspect ratio.
    """)
    return


@app.cell
def _(alt, features, mo, source_order):
    _geom = (
        features.groupby("source")
        .agg(aspect_median=("aspect", "median"), canvas_fill=("canvas_fill", "median"))
        .round(2)
        .reset_index()
    )
    _chart = (
        alt.Chart(features[["source", "canvas_fill"]].sample(8000, random_state=0))
        .transform_density("canvas_fill", groupby=["source"], as_=["canvas_fill", "density"])
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("canvas_fill:Q", title="share of 224x224 canvas that is image content"),
            y="density:Q",
            color=alt.Color("source:N", sort=source_order),
        )
        .properties(width=560, height=200)
    )
    mo.vstack(
        [
            _chart,
            mo.ui.table(_geom, selection=None),
            mo.md(
                "Median canvas fill is about 0.97 for 512-wide Duke/Kermany, 0.65 for PAIMA and 0.37 for OCTDL. Kermany "
                "also spans 0.32 (1536-wide) to 1.0 (512x512). The padding extent is itself a source cue, which the "
                "median padding option reduces in contrast but cannot remove."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Black fill and static overlays
    """)
    return


@app.cell
def _(alt, features, label_order, mo, source_order):
    _black = (
        features.groupby(["source", "label"])
        .agg(share=("has_black", "mean"), frac=("black_frac", "mean"), n=("path", "size"))
        .reset_index()
    )
    _chart = (
        alt.Chart(_black)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=label_order, title=None),
            y=alt.Y("share:Q", title="images with black fill", axis=alt.Axis(format="%")),
            column=alt.Column("source:N", sort=source_order, title=None),
            color=alt.Color("label:N", sort=label_order, legend=None),
            tooltip=[
                "source",
                "label",
                "n",
                alt.Tooltip("share:Q", format=".1%"),
                alt.Tooltip("frac:Q", format=".2%"),
            ],
        )
        .properties(width=140, height=180, title="Images with border-connected black fill")
    )
    _duke = features[features["source"] == "duke"]
    _by_size = (
        _duke.groupby(["label", "size"])
        .agg(
            n=("path", "size"),
            black_left_share=("black_left", lambda s: (s > 0).mean()),
            mean_black_frac=("black_frac", "mean"),
        )
        .round(3)
        .reset_index()
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                - **PAIMA:** the black is the denoised, exactly-zero background of the deep layers (about a third of the
                  image). It is genuine image content, not padding. Duke/Kermany backgrounds are speckled instead,
                  which is another source cue.
                - **Duke:** black fill is class-linked: 31% of DME, 5% of AMD, 0.1% of Normal. In 768-wide DME
                  images it is a thin black strip at the left and mostly the right edge (a few pixels wide); in
                  512-wide AMD it is black rows at the top/bottom. A small area, but strongly tied to class.
                """
            ),
            mo.ui.table(_by_size, selection=None),
        ]
    )
    return


@app.cell
def _(features, mo, source_order):
    _groups = (
        features.assign(group=features["source"] + " " + features["size"])
        .groupby("group")
        .filter(lambda g: len(g) >= 150)["group"]
        .unique()
    )
    overlay_group = mo.ui.dropdown(
        options=sorted(_groups, key=lambda g: (source_order.index(g.split()[0]), g)),
        value="paima 768x496",
        label="Source and size",
    )
    overlay_threshold = mo.ui.slider(
        0.5, 1.0, step=0.05, value=0.9, label="Static if bright in at least"
    )
    mo.vstack(
        [
            mo.md(
                "Overlays: for same-size images, the share of images in which each pixel is bright (>=200). A fixed "
                "overlay is bright in nearly all of them."
            ),
            mo.hstack([overlay_group, overlay_threshold], justify="start"),
        ]
    )
    return overlay_group, overlay_threshold


@app.cell
def _(
    Image,
    dataset_roots,
    features,
    io,
    mo,
    np,
    overlay_group,
    overlay_threshold,
    static_overlay_maps,
):
    _source, _size = overlay_group.value.split()
    _width, _height = (int(v) for v in _size.split("x"))
    _pool = features[
        (features["source"] == _source)
        & (features["width"] == _width)
        & (features["height"] == _height)
    ]
    _pool = _pool.sample(min(300, len(_pool)), random_state=0)

    def _gray_images():
        for _path in _pool["path"]:
            with Image.open(dataset_roots[_source] / _path) as _image:
                yield np.asarray(_image.convert("L"))

    _frequency, _std = static_overlay_maps(_gray_images())
    _static = _frequency >= overlay_threshold.value
    _ys, _xs = np.nonzero(_static)
    _buffer = io.BytesIO()
    Image.fromarray((_frequency * 255).astype(np.uint8)).save(_buffer, format="PNG")
    if _static.any():
        _box = f"x {_xs.min()}-{_xs.max()}, y {_ys.min()}-{_ys.max()} (image is {_width}x{_height})"
        _verdict = f"**{int(_static.sum())} static pixels** at {_box}."
    else:
        _verdict = "**No static overlay** at this threshold."
    mo.vstack(
        [
            mo.md(f"{len(_pool)} images sampled. {_verdict}"),
            mo.image(src=_buffer.getvalue(), width=min(_width, 700)),
        ]
    )
    return


@app.cell
def _(Image, dataset_roots, features, mo, np):
    # PAIMA scale bar: the box found above, measured on every PAIMA image.
    from concurrent.futures import ThreadPoolExecutor

    _paima = features[features["source"] == "paima"].copy()

    def _bar_pixels(path):
        with Image.open(dataset_roots["paima"] / path) as _image:
            _gray = np.asarray(_image.convert("L"))
        return int((_gray[436:487, 10:56] >= 200).sum()), int(
            (_gray[436:487, -56:-10] >= 200).sum()
        )

    with ThreadPoolExecutor(8) as _pool:
        _counts = np.array(list(_pool.map(_bar_pixels, _paima["path"])))
    _paima["bar_left"], _paima["bar_right"] = _counts[:, 0] >= 100, _counts[:, 1] >= 100
    _table = (
        _paima.groupby("label")
        .agg(
            images=("path", "size"),
            bar_bottom_left=("bar_left", "mean"),
            bar_bottom_right=("bar_right", "mean"),
        )
        .round(4)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ### PAIMA scale bar
                The "200 µm" scale bar is at the **bottom-left**, not bottom-right, in {_paima["bar_left"].mean():.1%} of PAIMA images.
                It is nearly label-neutral, so it identifies the *source* rather than the diagnosis. The few images
                without it come from specific patients (e.g. `CNV/159`) and skew toward AMD.
                """
            ),
            mo.ui.table(_table, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Shortcut probe: how well do artifacts alone predict the label?
    A gradient-boosted classifier sees **no pixel content**, only scalar descriptors, with 5-fold cross-validation
    grouped by patient. Balanced accuracy well above chance (1/#classes) means those artifacts leak the label
    within that source, so a pixel model could exploit them too.
    """)
    return


@app.cell
def _(mo):
    probe_run = mo.ui.run_button(label="Run probe (about 30 s)")
    mo.hstack([probe_run], justify="start")
    return (probe_run,)


@app.cell
def _(alt, features, mo, pd, probe_run, source_order):
    mo.stop(not probe_run.value, mo.md("_Press the button to run._"))
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    _geometry = ["width", "height", "aspect"]
    _fill = [c for c in features.columns if c.startswith(("white_", "black_"))]
    _intensity = ["gray_mean", "p1_all", "p99_all", "p1_content", "p50_content", "p99_content"]
    _sets = {
        "geometry only": _geometry,
        "fill only": _fill,
        "intensity only": _intensity,
        "all artifacts": _geometry + _fill + _intensity,
    }
    _rows = []
    for _source in source_order:
        _data = features[features["source"] == _source]
        if len(_data) > 30000:
            _data = _data.sample(30000, random_state=0)
        _groups = _data["group_id"].astype(str)
        for _name, _columns in _sets.items():
            _pred = cross_val_predict(
                HistGradientBoostingClassifier(max_iter=100, random_state=0),
                _data[_columns],
                _data["label"],
                groups=_groups,
                cv=GroupKFold(5),
            )
            _rows.append(
                {
                    "source": _source,
                    "features": _name,
                    "balanced_accuracy": balanced_accuracy_score(_data["label"], _pred),
                    "chance": 1 / _data["label"].nunique(),
                }
            )
    probe = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(probe)
        .mark_bar()
        .encode(
            x=alt.X("features:N", title=None, sort=list(_sets)),
            y=alt.Y(
                "balanced_accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="balanced accuracy"
            ),
            color=alt.Color("features:N", legend=None, sort=list(_sets)),
            column=alt.Column("source:N", sort=source_order, title=None),
            tooltip=[
                "source",
                "features",
                alt.Tooltip("balanced_accuracy:Q", format=".3f"),
                alt.Tooltip("chance:Q", format=".2f"),
            ],
        )
        .properties(
            width=110, height=200, title="Label prediction from artifacts alone (no pixels)"
        )
    )
    mo.vstack(
        [
            _chart,
            mo.md(
                """
                Chance is 0.33 for Duke/Kermany/OCTDL and 0.50 for PAIMA. **Canvas size alone predicts Duke's label at about 0.66 and
                Kermany's at about 0.58.** OCTDL has no fill, and only intensity descriptors lift it (to about 0.50). PAIMA
                reaches about 0.6 from its black-region and intensity descriptors (its exact-zero background varies with the patient and class).
                These are leakage indicators, not accuracy claims: patient-grouped CV on Duke has
                45 groups, so the Duke figures are noisy.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Summary of what the EDA found

    | # | Finding | Evidence | Likely consequence |
    | --- | --- | --- | --- |
    | 1 | White rotation/crop fill is common in Duke and Kermany only | ~98% of Duke, ~60% of Kermany, 0% of PAIMA/OCTDL | Source cue. Absent in held-out PAIMA/OCTDL |
    | 2 | Fill breaks percentile normalization | median tissue ceiling 0.73 Duke, 0.80 Kermany (0.75 for 512-wide, 1.0 for 768/1536-wide), 1.0 PAIMA/OCTDL | Unequal contrast per source, even per size within Kermany |
    | 3 | Canvas size correlates with class | Duke Normal 100% 512; Kermany DME 74% 512x512 | Label shortcut, probe 0.66 / 0.58 vs 0.33 chance |
    | 4 | Kermany test split is a different geometry mix | test dominated by 1024-wide | Train-test gap confounded with acquisition format |
    | 5 | Duke black edge strips are class-linked | black fill in 31% of DME, 5% of AMD, 0.1% of Normal | Small area, strong class cue |
    | 6 | PAIMA has a static bottom-left scale bar | ~99% of images, label-neutral | Source ID only |
    | 7 | OCTDL is ~2.8:1 and mostly padding after resize | canvas fill 0.37 | Strong source cue, small tissue area |

    Nothing here changes the manifests. These are options to consider next, not applied changes:
    mask or crop the fill and recompute percentiles on tissue only (as in section 4),
    crop out the PAIMA bar, stratify train/eval reporting by size, and rerun a LOSO ablation with fill-aware
    preprocessing to see whether the DME drop shrinks.
    """)
    return


if __name__ == "__main__":
    app.run()
