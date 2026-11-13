import torch
import pytest

from models import create_model
from xai import (
    cosine_similarity,
    grad_cam,
    integrated_gradients,
    integrated_gradients_legacy,
    integrated_gradients_v2,
    paired_explanation,
    save_explanation,
    spearman_rank_correlation,
    ssim,
)
from xai.integrated_gradients.method import _quadrature_grid


@pytest.mark.parametrize("architecture", ["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
def test_gradcam_and_integrated_gradients_for_all_architectures(architecture):
    model = create_model(architecture, pretrained=False, head_type="gap_linear").eval()
    image = torch.randn(1, 3, 224, 224)
    heatmap, logits = grad_cam(model, image, 1)
    attribution, delta = integrated_gradients(model, image, 1, steps=2)
    assert heatmap.shape == (1, 224, 224)
    assert logits.shape == (1, 3)
    assert attribution.shape == image.shape
    assert delta.shape == (1,)
    assert torch.isfinite(heatmap).all() and torch.isfinite(attribution).all()


def test_gradcam_hooks_are_removed_and_target_is_explicit():
    model = create_model("resnet18", pretrained=False).eval()
    image = torch.randn(1, 3, 224, 224)
    grad_cam(model, image, 2)
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(ValueError):
        grad_cam(model, image, 3)


def test_stability_identical_different_constant_and_invalid_maps():
    first = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    second = torch.flip(first, (0,))
    assert ssim(first, first) == 1
    assert cosine_similarity(first, first) == pytest.approx(1)
    assert spearman_rank_correlation(first, first) == pytest.approx(1)
    assert torch.isfinite(torch.tensor([ssim(first, second), cosine_similarity(first, second)])).all()
    assert cosine_similarity(torch.ones(4, 4), torch.ones(4, 4)) == 1
    with pytest.raises(ValueError):
        ssim(torch.ones(4, 4), torch.ones(3, 3))


def test_paired_protocol_uses_clean_target_and_records_predictions(tmp_path):
    model = create_model("resnet18", pretrained=False).eval()
    clean, corrupted = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
    result = paired_explanation(
        model, clean, corrupted,
        {"source_relative_path": "class/a.JPG", "sample_id": "duplicate", "class_index": "1"},
        "gaussian_noise_s1", method="gradcam",
    )
    assert result["target_class"] == result["clean_prediction"]
    assert "corrupted_prediction" in result
    assert result["sample_identity"] == "class/a.JPG"
    path = save_explanation(result, tmp_path / "explanation.pt")
    assert path.is_file() and path.with_suffix(".json").is_file()


def test_quadrature_weights_and_endpoints():
    left_alphas, left_weights = _quadrature_grid(4, "left", torch.device("cpu"))
    trapezoid_alphas, trapezoid_weights = _quadrature_grid(4, "trapezoidal", torch.device("cpu"))
    midpoint_alphas, midpoint_weights = _quadrature_grid(4, "midpoint", torch.device("cpu"))
    assert torch.equal(left_alphas, torch.tensor([0.0, 0.25, 0.5, 0.75]))
    assert torch.equal(midpoint_alphas, torch.tensor([0.125, 0.375, 0.625, 0.875]))
    assert torch.equal(trapezoid_alphas, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert left_weights.sum() == pytest.approx(1.0)
    assert midpoint_weights.sum() == pytest.approx(1.0)
    assert trapezoid_weights.sum() == pytest.approx(1.0)
    assert trapezoid_weights[[0, -1]].tolist() == pytest.approx([0.125, 0.125])


def test_integrated_gradients_methods_are_fp32_deterministic_and_complete():
    class LinearModel(torch.nn.Module):
        num_classes = 2

        def forward(self, inputs):
            return torch.cat((inputs[:, :1].sum(1, keepdim=True),
                              -inputs[:, :1].sum(1, keepdim=True)), dim=1)

    model = LinearModel().eval()
    image = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    baseline = torch.zeros_like(image)
    expected = torch.tensor(2.0)
    for method in ("left", "trapezoidal", "midpoint"):
        attribution, delta = integrated_gradients(
            model, image, 0, steps=4, baseline=baseline,
            integration_method=method, gradient_batch_size=2,
        )
        assert attribution.dtype == torch.float32
        assert attribution.shape == image.shape
        assert attribution.sum() == pytest.approx(expected.item())
        assert delta.item() == pytest.approx(0.0)
        second, second_delta = integrated_gradients(
            model, image, 0, steps=4, integration_method=method,
        )
        assert torch.equal(attribution, second)
        assert torch.equal(delta, second_delta)


def test_v2_matches_finite_difference_gradient_and_legacy_is_available():
    class QuadraticModel(torch.nn.Module):
        num_classes = 2

        def forward(self, inputs):
            value = (inputs[:, :1] ** 2).sum(1, keepdim=True)
            return torch.cat((value, -value), dim=1)

    model = QuadraticModel().eval()
    image = torch.tensor([[[[1.5]]]], dtype=torch.float32)
    attribution, delta = integrated_gradients_v2(
        model, image, 0, steps=512, integration_method="trapezoidal",
        gradient_batch_size=8,
    )
    epsilon = 1e-3
    plus = model(image + epsilon)[0, 0]
    minus = model(image - epsilon)[0, 0]
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic_gradient = 2 * image[0, 0, 0, 0]
    assert finite_difference == pytest.approx(analytic_gradient.item(), rel=1e-4)
    assert attribution.sum() == pytest.approx(2.25, rel=1e-4)
    assert delta.item() == pytest.approx(0.0, abs=1e-5)
    legacy_attr, legacy_delta = integrated_gradients_legacy(model, image, 0, steps=4)
    assert legacy_attr.shape == image.shape
    assert legacy_delta.numel() == 1


def test_integrated_gradients_multi_image_batch_and_stability_channel_reduction():
    class QuadraticModel(torch.nn.Module):
        num_classes = 2

        def forward(self, inputs):
            value = (inputs ** 2).flatten(1).sum(1, keepdim=True)
            return torch.cat((value, -value), dim=1)

    model = QuadraticModel().eval()
    images = torch.tensor(
        [[[[1.0, 2.0], [0.5, -1.0]]], [[[2.0, -0.5], [1.5, 0.25]]]],
        dtype=torch.float32,
    )
    for method in ("left", "trapezoidal", "midpoint"):
        batched_attr, batched_delta = integrated_gradients(
            model, images, 0, steps=16, integration_method=method, gradient_batch_size=3,
        )
        single_attrs, single_deltas = [], []
        for idx in range(images.shape[0]):
            attr_i, delta_i = integrated_gradients(
                model, images[idx:idx + 1], 0, steps=16,
                integration_method=method, gradient_batch_size=1,
            )
            single_attrs.append(attr_i)
            single_deltas.append(delta_i)
        assert torch.allclose(batched_attr, torch.cat(single_attrs, dim=0), atol=1e-6)
        assert torch.allclose(batched_delta, torch.cat(single_deltas, dim=0), atol=1e-6)

    multi_channel = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]], [[-1.0, 2.0], [-3.0, 4.0]]]])
    reduced = multi_channel.abs().mean(1)
    assert cosine_similarity(multi_channel, reduced) == pytest.approx(1.0)
    assert ssim(multi_channel, reduced) == pytest.approx(1.0)
    assert spearman_rank_correlation(multi_channel, reduced) == pytest.approx(1.0)

