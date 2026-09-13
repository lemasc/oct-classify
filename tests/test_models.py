import pytest

from oct_classify.models import build_model


def test_resnet50_replaces_classifier_head() -> None:
    model = build_model("resnet50", 2, pretrained=False)

    assert model.fc.out_features == 2


def test_model_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_model("unknown", 2, pretrained=False)
