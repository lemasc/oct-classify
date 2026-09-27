import torch

from oct_classify.analysis.gradcam import GradCam, randomize_top, upsample
from oct_classify.models import build_model


def test_gradcam_returns_non_negative_maps_per_layer() -> None:
    model = build_model("resnet50", 3, pretrained=False).eval()
    cam = GradCam(model, ["layer3", "layer4"])

    maps, logits = cam(torch.randn(2, 3, 64, 64), torch.tensor([0, 2]))
    cam.close()

    assert logits.shape == (2, 3)
    assert maps["layer3"].shape == (2, 4, 4)
    assert maps["layer4"].shape == (2, 2, 2)
    assert all(bool((value >= 0).all()) for value in maps.values())
    assert upsample(maps["layer4"], 64).shape == (2, 64, 64)


def test_randomize_top_changes_only_top_stage() -> None:
    model = build_model("resnet50", 3, pretrained=False)
    randomized = randomize_top(model)

    assert torch.equal(model.layer3[0].conv1.weight, randomized.layer3[0].conv1.weight)
    assert not torch.equal(model.fc.weight, randomized.fc.weight)
