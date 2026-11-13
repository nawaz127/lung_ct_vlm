"""Versioned FP32 Integrated Gradients protocol."""

from .method import integrated_gradients


PROTOCOL_VERSION = "ig_v2_quadrature_fp32"


def integrated_gradients_v2(
    model, images, target_class, steps=32, baseline=None, return_delta=True,
    integration_method="trapezoidal", gradient_batch_size=8,
):
    return integrated_gradients(
        model, images, target_class, steps=steps, baseline=baseline,
        return_delta=return_delta, integration_method=integration_method,
        gradient_batch_size=gradient_batch_size,
    )
