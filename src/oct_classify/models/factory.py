from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(architecture: str, num_classes: int, *, pretrained: bool) -> nn.Module:
    """Build a classifier while keeping head replacement specific to each backbone."""
    if num_classes < 2:
        raise ValueError("A classifier must have at least two classes.")
    if architecture == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model architecture: {architecture}")
