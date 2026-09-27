"""Grad-CAM for gallery images, with a retina-mass metric and a weight-randomization check."""

from __future__ import annotations

import copy
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional

from oct_classify.analysis.retina import (
    content_mask,
    fill_mask,
    mass_fraction,
    plausible_retina,
    retina_mask,
)
from oct_classify.data.preprocessing import PreprocessingSpec, preprocess_image
from oct_classify.models import build_model

DEFAULT_LAYERS = ("layer3", "layer4")
TARGET_ROLES = ("predicted", "true")


class GradCam:
    """Hook named modules and weight their activations by spatially pooled class gradients."""

    def __init__(self, model: nn.Module, layer_names: Sequence[str]) -> None:
        modules = dict(model.named_modules())
        unknown = [name for name in layer_names if name not in modules]
        if unknown:
            raise ValueError(f"Model has no modules named {unknown}.")
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = [
            modules[name].register_forward_hook(self._store(name)) for name in layer_names
        ]

    def _store(self, name: str):
        def hook(module: nn.Module, inputs: object, output: torch.Tensor) -> None:
            output.retain_grad()
            self.activations[name] = output

        return hook

    def __call__(
        self, images: torch.Tensor, class_indices: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return non-negative `(batch, h, w)` maps at each layer's native resolution, and logits."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(images)
            logits.gather(1, class_indices[:, None]).sum().backward()
        cams = {}
        for name, activation in self.activations.items():
            weights = activation.grad.mean(dim=(2, 3), keepdim=True)
            cams[name] = torch.relu((weights * activation).sum(dim=1)).detach()
        return cams, logits.detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def upsample(cam: torch.Tensor, size: int) -> torch.Tensor:
    return functional.interpolate(
        cam[:, None], size=(size, size), mode="bilinear", align_corners=False
    )[:, 0]


def randomize_top(model: nn.Module) -> nn.Module:
    """Copy the model with `layer4` and `fc` re-initialized (cascading randomization, top stage)."""
    randomized = copy.deepcopy(model)
    for module_name in ("layer4", "fc"):
        for module in getattr(randomized, module_name).modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            if hasattr(module, "reset_running_stats"):
                module.reset_running_stats()
    return randomized


def _load_image(
    path: Path, spec: PreprocessingSpec, mean: np.ndarray, stdev: np.ndarray
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        width, height = image.size
        array = preprocess_image(image, spec)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    # Same normalization as ManifestImageDataset.
    tensor = (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(
        np.maximum(stdev, 1e-6)
    ).view(3, 1, 1)
    gray = array.mean(axis=2)
    content = content_mask(width, height, spec.image_size)
    content &= ~fill_mask(gray, content)
    return tensor.float(), gray, content


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_gradcam(
    set_dir: Path,
    checkpoint_path: Path,
    roots: Mapping[str, Path],
    device: torch.device,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    batch_size: int = 16,
    randomization_sample: int = 32,
    seed: int = 20260927,
) -> dict[str, object]:
    rows = _read_rows(set_dir / "errors.csv") + _read_rows(set_dir / "controls.csv")
    if not rows:
        raise FileNotFoundError(f"No errors.csv/controls.csv rows in {set_dir}; run `analyze run`.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model_labels = list(metadata["class_labels"])
    spec = PreprocessingSpec(image_size=metadata["config"]["data"]["image_size"])
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
    stdev = np.asarray(metadata["normalization"]["stdev"], dtype=np.float32)
    model = build_model(metadata["architecture"], len(model_labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    cam = GradCam(model, layers)

    cams: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    upsampled_predicted: dict[str, dict[str, np.ndarray]] = {}
    images_by_row: dict[str, torch.Tensor] = {}
    rng = np.random.default_rng(seed)
    sanity_ids = set(
        rng.choice(
            [row["row_id"] for row in rows], size=min(randomization_sample, len(rows)), replace=False
        ).tolist()
    )
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        loaded = [
            _load_image(roots[row["source"]] / row["path"], spec, mean, stdev) for row in batch
        ]
        images = torch.stack([item[0] for item in loaded]).to(device)
        indices = {
            "predicted": torch.tensor([model_labels.index(row["prediction"]) for row in batch]),
            "true": torch.tensor([model_labels.index(row["target"]) for row in batch]),
        }
        maps_by_role = {}
        for role in TARGET_ROLES:
            maps, _ = cam(images, indices[role].to(device))
            maps_by_role[role] = maps
        for offset, row in enumerate(batch):
            row_id = row["row_id"]
            _, gray, content = loaded[offset]
            retina = retina_mask(gray, content)
            plausible = plausible_retina(retina, content)
            masks[f"{row_id}__content"] = np.packbits(content)
            masks[f"{row_id}__retina"] = np.packbits(retina)
            if row_id in sanity_ids:
                images_by_row[row_id] = images[offset].cpu()
            for role, maps in maps_by_role.items():
                for layer, layer_maps in maps.items():
                    native = layer_maps[offset : offset + 1]
                    full = upsample(native, spec.image_size)[0].cpu().numpy()
                    cams[f"{row_id}__{layer}__{role}"] = native[0].cpu().numpy().astype(np.float32)
                    if role == "predicted" and row_id in sanity_ids:
                        upsampled_predicted.setdefault(row_id, {})[layer] = full
                    records.append(
                        {
                            "row_id": row_id,
                            "kind": row["kind"],
                            "path": row["path"],
                            "target": row["target"],
                            "prediction": row["prediction"],
                            "layer": layer,
                            "target_role": role,
                            "target_class": row["prediction" if role == "predicted" else "target"],
                            "padding_mass": mass_fraction(full, ~content),
                            "outside_retina_mass": mass_fraction(full, ~retina),
                            "retina_area_fraction": float(retina.sum() / max(content.sum(), 1)),
                            "mask_plausible": plausible,
                        }
                    )
    cam.close()
    sanity = _randomization_check(
        model, layers, images_by_row, upsampled_predicted, rows, model_labels, spec, device
    )
    output = set_dir / "gradcam"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "cams.npz", **cams)
    np.savez_compressed(output / "masks.npz", **masks)
    with (output / "gradcam.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {
        "checkpoint": str(checkpoint_path),
        "layers": list(layers),
        "image_size": spec.image_size,
        "rows": len(rows),
        "attention": summarize_attention(records),
        "randomization_check": sanity,
    }


def _randomization_check(
    model: nn.Module,
    layers: Sequence[str],
    images_by_row: Mapping[str, torch.Tensor],
    trained_maps: Mapping[str, Mapping[str, np.ndarray]],
    rows: Sequence[Mapping[str, str]],
    model_labels: Sequence[str],
    spec: PreprocessingSpec,
    device: torch.device,
) -> dict[str, object]:
    """Spearman correlation between trained and top-randomized CAMs for the predicted class.

    If maps barely change after destroying the learned `layer4`/`fc` weights, they reflect image
    structure (edges, bright layers) rather than what the classifier learned.
    """
    if not images_by_row:
        return {"images": 0}
    randomized = randomize_top(model).eval().to(device)
    cam = GradCam(randomized, layers)
    prediction_by_row = {row["row_id"]: row["prediction"] for row in rows}
    row_ids = sorted(images_by_row)
    correlations: dict[str, list[float]] = {layer: [] for layer in layers}
    for start in range(0, len(row_ids), 16):
        batch_ids = row_ids[start : start + 16]
        images = torch.stack([images_by_row[row_id] for row_id in batch_ids]).to(device)
        indices = torch.tensor([model_labels.index(prediction_by_row[row_id]) for row_id in batch_ids])
        maps, _ = cam(images, indices.to(device))
        for layer, layer_maps in maps.items():
            full = upsample(layer_maps, spec.image_size).cpu().numpy()
            for offset, row_id in enumerate(batch_ids):
                trained = trained_maps[row_id][layer].ravel()
                random = full[offset].ravel()
                if trained.std() == 0 or random.std() == 0:
                    continue
                correlations[layer].append(float(spearmanr(trained, random).statistic))
    cam.close()
    return {
        "images": len(row_ids),
        "randomized_modules": ["layer4", "fc"],
        "spearman_by_layer": {
            layer: {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for layer, values in correlations.items()
        },
    }


def summarize_attention(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Mean/median off-retina and padding mass by kind (and kind:cell), layer, and target role."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for record in records:
        cell = f"{record['target']}->{record['prediction']}"
        for group in (str(record["kind"]), f"{record['kind']}:{cell}"):
            key = (group, str(record["layer"]), str(record["target_role"]))
            grouped.setdefault(key, []).append(record)
    rows = []
    for (group, layer, role), members in sorted(grouped.items()):
        row: dict[str, object] = {"group": group, "layer": layer, "target_role": role, "n": len(members)}
        for metric in ("outside_retina_mass", "padding_mass"):
            values = [
                float(member[metric])
                for member in members
                if member[metric] is not None
                and (metric == "padding_mass" or member["mask_plausible"])
            ]
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_median"] = float(np.median(values)) if values else None
        rows.append(row)
    return rows
