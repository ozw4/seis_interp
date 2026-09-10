from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from seis_interp.models.nersi import FourierFeatureMapping, Nersi, NersiBlock


def _tiny_model(**changes: object) -> Nersi:
    options = {
        "fourier_components": 3,
        "frequency_base": 1.5,
        "encoder_width": 12,
        "latent_channels": 4,
        "decoder_channels": (4, 3, 2),
        "profile_shape": (8, 8),
        "kernel_size": 3,
    }
    options.update(changes)
    return Nersi(**options)


def test_fourier_frequency_buffer_uses_one_based_exponential_schedule() -> None:
    mapping = FourierFeatureMapping(fourier_components=4, frequency_base=1.5)
    expected = torch.tensor([math.pi * 1.5**exponent for exponent in range(1, 5)])

    torch.testing.assert_close(mapping.frequencies, expected)
    assert "frequencies" in dict(mapping.named_buffers())
    assert "frequencies" not in dict(mapping.named_parameters())
    assert "frequencies" in mapping.state_dict()


def test_fourier_features_are_axis_major_with_interleaved_cosine_and_sine() -> None:
    mapping = FourierFeatureMapping(
        input_features=3,
        fourier_components=2,
        frequency_base=2.0,
    ).double()
    coordinates = torch.tensor([[0.0, 0.25, 0.5]], dtype=torch.float64)
    frequencies = torch.tensor([2.0 * math.pi, 4.0 * math.pi], dtype=torch.float64)
    expected = torch.cat(
        [
            torch.stack(
                (torch.cos(axis * frequencies), torch.sin(axis * frequencies)), dim=-1
            ).reshape(-1)
            for axis in coordinates[0]
        ]
    ).unsqueeze(0)

    actual = mapping(coordinates)

    assert actual.shape == (1, 12)
    # The registered schedule is float32 by default, matching model parameters.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)


def test_nersi_block_doubles_both_profile_axes() -> None:
    block = NersiBlock(3, 5, kernel_size=3)
    values = torch.randn(2, 3, 4, 7)

    output = block(values)

    assert output.shape == (2, 5, 8, 14)
    assert block.convolution.out_channels == 5 * 2**2
    assert block.convolution.padding == (1, 1)
    assert isinstance(block.pixel_shuffle, nn.PixelShuffle)
    assert isinstance(block.activation, nn.GELU)


def test_three_decoder_blocks_produce_the_declared_profile_shape() -> None:
    model = _tiny_model()

    output = model(torch.rand(5, 3))

    assert output.shape == (5, 1, 8, 8)
    assert model.base_profile_shape == (1, 1)
    assert len(model.decoder) == 3
    assert all(isinstance(block, NersiBlock) for block in model.decoder)
    assert [block.upsample_scale for block in model.decoder] == [2, 2, 2]
    assert model.output_convolution.kernel_size == (1, 1)


def test_forward_backward_is_finite_for_every_trainable_parameter() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    coordinates = torch.rand(4, 3, requires_grad=True)

    output = model(coordinates)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_linear_output_can_represent_negative_signed_amplitudes() -> None:
    model = _tiny_model()
    with torch.no_grad():
        model.output_convolution.weight.zero_()
        model.output_convolution.bias.fill_(-2.0)

    output = model(torch.rand(2, 3))

    torch.testing.assert_close(output, -2.0 * torch.ones_like(output), rtol=0.0, atol=0.0)


def test_constructor_config_restores_structure_and_parameter_shapes() -> None:
    model = _tiny_model()
    expected = {
        "input_features": 3,
        "fourier_components": 3,
        "frequency_base": 1.5,
        "encoder_width": 12,
        "latent_channels": 4,
        "decoder_channels": (4, 3, 2),
        "profile_shape": (8, 8),
        "upsample_scales": (2, 2, 2),
        "kernel_size": 3,
        "activation": "gelu",
        "output_activation": "linear",
    }

    assert model.constructor_config() == expected
    restored = Nersi(**model.constructor_config())
    assert {name: tuple(parameter.shape) for name, parameter in restored.named_parameters()} == {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }
    assert restored.fourier_mapping.frequencies.shape == model.fourier_mapping.frequencies.shape


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"input_features": 4}, "input_features"),
        ({"fourier_components": 0}, "fourier_components"),
        ({"frequency_base": 1.0}, "frequency_base"),
        ({"frequency_base": float("inf")}, "frequency_base"),
        ({"encoder_width": 0}, "encoder_width"),
        ({"latent_channels": False}, "latent_channels"),
        ({"decoder_channels": (4, 3)}, "decoder_channels"),
        ({"decoder_channels": (4, 0, 2)}, "decoder_channels"),
        ({"profile_shape": (16, 7)}, "profile_shape"),
        ({"profile_shape": (8,)}, "profile_shape"),
        ({"upsample_scales": (2, 2, 4)}, "upsample_scales"),
        ({"kernel_size": 2}, "kernel_size"),
        ({"activation": "relu"}, "activation"),
        ({"output_activation": "tanh"}, "output_activation"),
    ],
)
def test_invalid_model_settings_are_rejected(changes: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _tiny_model(**changes)


def test_parameter_count_is_independent_of_profile_batch_count() -> None:
    model = _tiny_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    one = model(torch.rand(1, 3))
    seven = model(torch.rand(7, 3))

    assert one.shape[0] == 1
    assert seven.shape[0] == 7
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


@pytest.mark.parametrize("shape", [(3,), (2, 4), (2, 3, 1)])
def test_fourier_mapping_rejects_invalid_coordinate_shape(shape: tuple[int, ...]) -> None:
    mapping = FourierFeatureMapping(fourier_components=2)

    with pytest.raises(ValueError, match="coordinates must have shape"):
        mapping(torch.zeros(shape))
