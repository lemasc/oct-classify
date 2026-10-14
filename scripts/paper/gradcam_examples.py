"""Grad-CAM examples for the paper: held-out Kermany 768x496 normal images, fill-aware vs square-crop LOSO models.

Kermany 768x496 is the width Duke labels DME only. The examples are images the fill-aware model calls DME and the
square-crop model calls normal (the size shortcut), plus controls both models get right. One model per arm (fold 1),
so the two arms are compared on the same images. Images go through the models' own evaluation pipeline
(`ManifestImageDataset`, training=False: centred square crop for square-crop models), and the stored predictions are
reproduced as a check. Grad-CAM is for the predicted class at layer4.

The quantitative analysis (all folds, band intervention, randomization check) is in notebooks/gradcam_shortcut.py,
which writes data/paper/gradcam/summary.json; this script only draws the example figure and writes examples.json.
CPU is enough (a dozen images). Run from the repository root:
`uv run --with matplotlib python scripts/paper/gradcam_examples.py`
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import style
from oct_classify.analysis.gradcam import GradCam, upsample
from oct_classify.analysis.retina import content_mask, mass_fraction
from oct_classify.data.config import load_dataset_specs
from oct_classify.data.manifest import read_jsonl
from oct_classify.data.preprocessing import PreprocessingSpec
from oct_classify.data.taxonomy import UnifiedLabel
from oct_classify.models import build_model
from oct_classify.training.config import AugmentationConfig
from oct_classify.training.dataset import ManifestImageDataset

RUNS = style.ROOT / "artifacts/runs/fused/resnet50"
ARMS = {
    "fill": RUNS / "loso-kermany-local-20261008-174938-fold1",
    "sq": RUNS / "loso-kermany-local-3a-r2-square-crop-fold1",
}
OUT = style.ROOT / "data/paper/gradcam"
SIZE = (768, 496)
N_SHORTCUT, N_CONTROL, SEED = 4, 2, 20261014
LAYER = "layer4"


def select() -> pd.DataFrame:
    """One image per patient: fill-aware says DME and square crop says normal, plus controls both get right."""
    features = pd.read_csv(style.ROOT / "artifacts/eda/features.csv", usecols=["source", "path", "width", "height"])
    sized = set(features.loc[(features.source == "kermany") & (features.width == SIZE[0])
                             & (features.height == SIZE[1]), "path"])
    preds = {
        arm: pd.read_csv(run / "evaluations/kermany-loso/predictions.csv").set_index("path")
        for arm, run in ARMS.items()
    }
    both = preds["fill"][["target", "prediction"]].join(preds["sq"][["prediction"]], rsuffix="_sq")
    both = both[both.index.isin(sized) & (both.target == "normal")]
    both["patient"] = [path.split("/")[-1].split("-")[1] for path in both.index]
    rng = np.random.default_rng(SEED)

    def pick(mask: pd.Series, n: int, kind: str) -> pd.DataFrame:
        pool = both[mask].groupby("patient").head(1)
        chosen = pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))].copy()
        chosen["kind"] = kind
        return chosen

    shortcut = pick((both.prediction == "dme") & (both.prediction_sq == "normal"), N_SHORTCUT, "shortcut")
    control = pick((both.prediction == "normal") & (both.prediction_sq == "normal")
                   & ~both.patient.isin(shortcut.patient), N_CONTROL, "control")
    counts = {
        "normal_768x496": int(len(both)),
        "fill_dme_sq_normal": int(((both.prediction == "dme") & (both.prediction_sq == "normal")).sum()),
        "both_normal": int(((both.prediction == "normal") & (both.prediction_sq == "normal")).sum()),
    }
    return pd.concat([shortcut, control]), counts


def run_arm(run, paths: list[str], records_by_path: dict, root) -> dict:
    checkpoint = torch.load(run / "checkpoint-best.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    config = metadata["config"]
    labels = tuple(UnifiedLabel(label) for label in metadata["class_labels"])
    preprocessing = PreprocessingSpec(image_size=config["data"]["image_size"], padding=config["data"]["padding"])
    dataset = ManifestImageDataset(
        root, [records_by_path[path] for path in paths], labels, preprocessing,
        np.asarray(metadata["normalization"]["mean"], dtype=np.float32),
        np.asarray(metadata["normalization"]["stdev"], dtype=np.float32),
        AugmentationConfig(**config["augmentation"]), training=False,
    )
    model = build_model(metadata["architecture"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cam = GradCam(model, [LAYER])
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    # One pass for the logits, a second for the predicted-class map (the hooks need gradients, so no `no_grad`).
    _, logits = cam(images, torch.zeros(len(images), dtype=torch.long))
    probabilities = torch.softmax(logits, dim=1).numpy()
    maps, _ = cam(images, torch.as_tensor(probabilities.argmax(1)))
    cam.close()
    full = upsample(maps[LAYER], preprocessing.image_size).numpy()
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)[:, None, None]
    stdev = np.asarray(metadata["normalization"]["stdev"], dtype=np.float32)[:, None, None]
    display = (images.numpy() * np.maximum(stdev, 1e-6) + mean).mean(axis=1)
    return {
        "labels": [label.value for label in labels], "probabilities": probabilities, "cams": full,
        "display": display, "geometry": config["augmentation"].get("geometry", "none"), "size": preprocessing.image_size,
    }


def main() -> None:
    style.apply()
    chosen, counts = select()
    paths = list(chosen.index)
    root = {spec.name: spec.root for spec in load_dataset_specs(style.ROOT / "configs/datasets.toml")}["kermany"]
    records_by_path = {r.path: r for r in read_jsonl(style.ROOT / "artifacts/splits/kermany.jsonl") if r.path in set(paths)}
    results = {arm: run_arm(run, paths, records_by_path, root) for arm, run in ARMS.items()}

    # Check: the recomputed probabilities reproduce the stored evaluation.
    check, summary_rows = {}, []
    for arm, run in ARMS.items():
        stored = pd.read_csv(run / "evaluations/kermany-loso/predictions.csv").set_index("path").loc[paths]
        columns = [f"probability_{label}" for label in results[arm]["labels"]]
        check[arm] = float(np.abs(stored[columns].to_numpy() - results[arm]["probabilities"]).max())
    in_image = content_mask(SIZE[0], SIZE[1], results["fill"]["size"])
    for index, path in enumerate(paths):
        row = {"path": path, "kind": chosen.loc[path, "kind"]}
        for arm in ARMS:
            r = results[arm]
            row[f"{arm}_prediction"] = r["labels"][int(r["probabilities"][index].argmax())]
            row[f"{arm}_p_dme"] = float(r["probabilities"][index][r["labels"].index("dme")])
        # Share of the fill-aware map on the letterbox bands (square-crop inputs have none).
        row["fill_cam_on_bands"] = mass_fraction(results["fill"]["cams"][index], ~in_image)
        summary_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"layer": LAYER, "seed": SEED, "image_size": list(SIZE), "counts_fold1": counts,
               "max_abs_probability_diff_vs_stored": check, "band_area_fraction": float((~in_image).mean()),
               "examples": summary_rows}
    (OUT / "examples.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
    for row in summary_rows:
        print(row)

    fig, axes = plt.subplots(2, len(paths), figsize=(style.TEXT_WIDTH_IN, 2.55),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    for column, path in enumerate(paths):
        for line, arm in enumerate(ARMS):
            ax, r = axes[line, column], results[arm]
            heat = r["cams"][column] / max(r["cams"][column].max(), 1e-12)
            ax.imshow(r["display"][column], cmap="gray", vmin=0, vmax=1)
            ax.imshow(heat, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            prediction = summary_rows[column][f"{arm}_prediction"]
            name = {"normal": "Normal", "amd": "AMD", "dme": "DME"}[prediction]
            ax.set_title(name, fontsize=7.5, pad=2, color=style.INK,
                         fontweight="normal" if prediction == "normal" else "bold")
            if column == 0:
                ax.set_ylabel(style.ARM_NAME[arm], fontsize=7.5, color=style.INK_2)
    for column, path in enumerate(paths):
        if chosen.loc[path, "kind"] == "control":
            axes[0, column].text(0.5, 1.22, "control", transform=axes[0, column].transAxes, ha="center",
                                 fontsize=7, color=style.INK_2)
    path = style.FIGURES / "gradcam.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {path.relative_to(style.ROOT)}")


if __name__ == "__main__":
    main()
