from __future__ import annotations

from itertools import product

import pytest
import torch
from torch.nn import functional as F

from seis_interp.models.ccnet5d import CC3D2D


def _loop_reference(
    values: torch.Tensor,
    weight_3d: torch.Tensor,
    bias_3d: torch.Tensor,
    weight_2d: torch.Tensor,
    bias_2d: torch.Tensor,
    *,
    output_activation: str,
) -> torch.Tensor:
    batch, _, time, source_x, source_y, receiver_x, receiver_y = values.shape
    output = values.new_empty(
        (batch, weight_2d.shape[0], time, source_x, source_y, receiver_x, receiver_y)
    )
    for b in range(batch):
        features = values.new_empty(
            (weight_3d.shape[0], time, source_x, source_y, receiver_x, receiver_y)
        )
        for rx, ry in product(range(receiver_x), range(receiver_y)):
            block = F.conv3d(
                values[b, :, :, :, :, rx, ry].unsqueeze(0),
                weight_3d,
                bias_3d,
                padding=weight_3d.shape[-1] // 2,
            )
            features[:, :, :, :, rx, ry] = torch.relu(block[0])
        for t, sx, sy in product(range(time), range(source_x), range(source_y)):
            block = F.conv2d(
                features[:, t, sx, sy, :, :].unsqueeze(0),
                weight_2d,
                bias_2d,
                padding=weight_2d.shape[-1] // 2,
            )[0]
            output[b, :, t, sx, sy, :, :] = (
                torch.relu(block) if output_activation == "relu" else block
            )
    return output


@pytest.mark.parametrize("output_activation", ["linear", "relu"])
def test_cc3d2d_matches_independent_loop_forward_and_gradients(output_activation: str) -> None:
    generator = torch.Generator().manual_seed(214)
    module = CC3D2D(2, 3, 2, kernel_size=3, output_activation=output_activation).double()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float64, generator=generator))
    values = torch.randn(2, 2, 2, 3, 4, 5, 6, dtype=torch.float64, generator=generator)
    values.requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    actual = module(values)
    expected = _loop_reference(
        reference_values,
        module.conv3d.weight,
        module.conv3d.bias,
        module.conv2d.weight,
        module.conv2d.bias,
        output_activation=output_activation,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    probe = torch.randn(actual.shape, dtype=torch.float64, generator=generator)
    actual_gradients = torch.autograd.grad(actual, (values, *module.parameters()), probe)
    reference_gradients = torch.autograd.grad(
        expected, (reference_values, *module.parameters()), probe
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-10)


def test_cc3d2d_preserves_shape_batch_independence_and_convolution_contract() -> None:
    module = CC3D2D(2, 3, 2, kernel_size=3)
    values = torch.randn(2, 2, 2, 3, 4, 5, 6)
    output = module(values)
    changed = values.clone()
    changed[1] = -7.0

    assert output.shape == values.shape
    assert output.dtype == values.dtype
    assert output.device == values.device
    assert output.is_contiguous()
    torch.testing.assert_close(module(changed)[0], output[0], rtol=0.0, atol=0.0)
    for convolution, dimensions in ((module.conv3d, 3), (module.conv2d, 2)):
        assert convolution.bias is not None
        assert convolution.groups == 1
        assert convolution.kernel_size == (3,) * dimensions
        assert convolution.stride == convolution.dilation == (1,) * dimensions
        assert convolution.padding == (1,) * dimensions
        assert convolution.padding_mode == "zeros"


def test_cc3d2d_accepts_noncontiguous_input_without_reordering_values() -> None:
    module = CC3D2D(2, 3, 2, kernel_size=3, output_activation="linear")
    values = torch.randn(2, 2, 2, 3, 4, 5, 12)[..., ::2]

    assert not values.is_contiguous()
    torch.testing.assert_close(module(values), module(values.contiguous()))


def test_cc3d2d_linear_regime_matches_explicit_five_dimensional_cross_correlation() -> None:
    generator = torch.Generator().manual_seed(319)
    module = CC3D2D(2, 2, 2, kernel_size=3).double()
    with torch.no_grad():
        module.conv3d.weight.copy_(
            torch.rand(module.conv3d.weight.shape, dtype=torch.float64, generator=generator)
        )
        module.conv2d.weight.copy_(
            torch.rand(module.conv2d.weight.shape, dtype=torch.float64, generator=generator)
        )
        module.conv3d.bias.zero_()
        module.conv2d.bias.zero_()
    values = torch.rand(1, 2, 2, 1, 3, 2, 1, dtype=torch.float64, generator=generator)
    kernel = torch.einsum("oruv,rcijk->ocijkuv", module.conv2d.weight, module.conv3d.weight)
    padded = F.pad(values, (1, 1) * 5)
    expected = torch.empty_like(values)
    for b, out_channel in product(range(values.shape[0]), range(2)):
        for position in product(*(range(length) for length in values.shape[2:])):
            region = tuple(slice(start, start + 3) for start in position)
            expected[(b, out_channel, *position)] = (
                padded[(b, slice(None), *region)] * kernel[out_channel]
            ).sum()

    torch.testing.assert_close(module(values), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"in_channels": 0}, "in_channels"),
        ({"intermediate_channels": True}, "intermediate_channels"),
        ({"out_channels": -1}, "out_channels"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"kernel_size": 2}, "kernel_size"),
        ({"kernel_size": 1.5}, "kernel_size"),
        ({"output_activation": "tanh"}, "output_activation"),
    ],
)
def test_cc3d2d_rejects_invalid_constructor_values(changed: dict[str, object], match: str) -> None:
    arguments = {"in_channels": 2, "intermediate_channels": 3, "out_channels": 2} | changed

    with pytest.raises(ValueError, match=match):
        CC3D2D(**arguments)


@pytest.mark.parametrize("shape", [(2, 2, 3, 4, 5, 6), (2, 1, 2, 3, 4, 5, 6)])
def test_cc3d2d_rejects_wrong_input_rank_or_channels(shape: tuple[int, ...]) -> None:
    module = CC3D2D(2, 3, 2)

    with pytest.raises(ValueError, match="shape|channels"):
        module(torch.zeros(shape))
