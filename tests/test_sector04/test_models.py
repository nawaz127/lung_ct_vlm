import pytest
import torch

from models import create_model, model_info


@pytest.mark.parametrize("name", ["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
@pytest.mark.parametrize("head", ["gap_linear", "gap_dropout_linear"])
def test_scratch_models_forward(name, head):
    model = create_model(name, num_classes=3, head_type=head, dropout=0.2)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_model_info_and_freezing():
    model = create_model("resnet18", freeze_backbone=True)
    info = model_info(model)
    assert info.total_parameters > 0
    assert info.trainable_parameters > 0
    assert info.frozen_parameters > 0
    assert info.trainable_parameters + info.frozen_parameters == info.total_parameters


def test_invalid_head_and_input():
    with pytest.raises(ValueError):
        create_model("resnet18", head_type="bad")
    model = create_model("resnet18")
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 224, 224))
