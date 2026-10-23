"""Time-only NeRSI upsampling: the block, the decoder option and the FORGE profile helpers."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from seis_interp.forge_mvp_config import (
    profile_side,
    profile_time_multiple,
    validate_added_method_config,
)
from seis_interp.models.nersi import Nersi, NersiBlock, NersiTimeBlock
from seis_interp.training.forge_mvp_methods import (
    canvas_method_config,
    initialize_model,
    model_parameter_count,
)

# study_061: aligned FORGE canvas length and its 2001-sample, 32-receiver-point contract.
CANVAS = 4343
CONTRACT = {"time_sample_count": 2001, "spatial_shape": [6, 32, 8, 32], "model_seed": 7301}
DEEP_TIME_MODEL = {
    "fourier_components": 40,
    "frequency_base": [1.041056, 1.089643, 1.04985],
    "encoder_width": 40,
    "latent_channels": 28,
    "decoder_channels": [29, 14, 8],
    "time_upsample_channels": [28, 28, 28, 28],
    "time_kernel_size": 3,
}
RANK2_MODEL = {
    "fourier_components": 40,
    "frequency_base": [1.041056, 1.089643, 1.04985],
    "encoder_width": 30,
    "latent_channels": 8,
    "decoder_channels": [29, 14, 8],
    "latent_spatial_rank": 2,
}


def _conv(in_channels, out_channels, kernel_size):
    return in_channels * out_channels * kernel_size**2 + out_channels


def _linear(in_features, out_features):
    return in_features * out_features + out_features


def _expected_parameters(
    *,
    fourier,
    encoder,
    latent,
    base,
    decoder,
    kernel,
    time_channels=(),
    time_kernel=None,
    rank=None,
):
    """Encoder, time-only blocks (2x channels), isotropic blocks (4x) and the output conv."""
    latent_size = latent * base[0] * base[1] if rank is None else latent * rank * sum(base)
    time_inputs = (latent, *time_channels)[: len(time_channels)]
    iso_inputs = (time_channels[-1] if time_channels else latent, *decoder[:-1])
    return (
        _linear(3 * 2 * fourier, encoder)
        + _linear(encoder, latent_size)
        + sum(_conv(i, 2 * o, time_kernel) for i, o in zip(time_inputs, time_channels, strict=True))
        + sum(_conv(i, 4 * o, kernel) for i, o in zip(iso_inputs, decoder, strict=True))
        + _conv(decoder[-1], 1, 1)
    )


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _tiny_model(**changes):
    options = {
        "fourier_components": 3,
        "frequency_base": 1.5,
        "encoder_width": 12,
        "latent_channels": 4,
        "decoder_channels": (4, 3, 2),
        "profile_shape": (16, 8),
        "kernel_size": 3,
    }
    options.update(changes)
    return Nersi(**options)


def _time_model(**changes):
    """Two time-only stages: a (64, 16) profile decodes from a (2, 2) latent grid."""
    options = {
        "profile_shape": (64, 16),
        "time_upsample_channels": (5, 6),
        "time_kernel_size": 1,
    }
    options.update(changes)
    return _tiny_model(**options)


def _forge_config(model_changes):
    """A study_061 ``nersi_profile`` config with the candidate's model overrides."""
    model = {
        "input_features": 3,
        "fourier_components": 16,
        "frequency_base": [1.05894, 1.18685, 1.081445],
        "encoder_width": 150,
        "latent_channels": 1,
        "decoder_channels": [16, 8, 4],
        "profile_shape": [2008, 32],
        "upsample_scales": [2, 2, 2],
        "kernel_size": 5,
        "activation": "gelu",
        "output_activation": "linear",
    }
    model.update(deepcopy(model_changes))
    return {
        "method_id": "nersi_profile",
        "geometry_mode": "real",
        "model_seed": 7301,
        "loss": "masked_trace_relative_mse",
        "checkpoint_rule": "fixed_final_ema",
        "device": "cpu",
        "model": model,
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "max_steps": 20000,
            "weight_decay": 5.0e-05,
            "report_interval": 100,
            "learning_rate_schedule": "constant",
            "mixed_precision": "off",
            "supervised_traces_per_update": 64,
            "ema_decay": 0.9998,
        },
        "prediction": {"profile_batch_size": 64},
    }


# NersiTimeBlock


def test_time_block_doubles_time_only_with_a_two_channel_convolution():
    block = NersiTimeBlock(3, 5, kernel_size=3)

    output = block(torch.randn(2, 3, 4, 7))

    assert output.shape == (2, 5, 8, 7)
    assert block.convolution.in_channels == 3
    assert block.convolution.out_channels == 5 * 2
    assert block.convolution.kernel_size == (3, 3)
    assert block.convolution.padding == (1, 1)
    assert isinstance(block.activation, nn.GELU)
    assert NersiTimeBlock(3, 5).kernel_size == 3


def test_time_block_output_c_at_time_2h_plus_j_is_gelu_of_convolution_channel_2c_plus_j():
    torch.manual_seed(3)
    block = NersiTimeBlock(3, 4, kernel_size=3).double()
    values = torch.randn(2, 3, 5, 6, dtype=torch.float64)

    output = block(values)
    convolved = block.convolution(values)
    expected = torch.empty(2, 4, 10, 6, dtype=torch.float64)
    for b in range(2):
        for c in range(4):
            for h in range(5):
                for j in range(2):
                    for w in range(6):
                        expected[b, c, 2 * h + j, w] = convolved[b, 2 * c + j, h, w]

    torch.testing.assert_close(output, nn.functional.gelu(expected))


def test_identity_convolution_interleaves_input_channel_pairs_along_time():
    # Conv channel k copies input channel k, so output channel c alternates input
    # channels 2c and 2c + 1 sample by sample; a channel-major shuffle would not.
    block = NersiTimeBlock(4, 2, kernel_size=1).double()
    with torch.no_grad():
        block.convolution.weight.copy_(torch.eye(4, dtype=torch.float64)[:, :, None, None])
        block.convolution.bias.zero_()
    values = torch.arange(1, 4 * 3 * 2 + 1, dtype=torch.float64).reshape(1, 4, 3, 2) / 10

    output = block(values)

    gelu = nn.functional.gelu
    for c in range(2):
        rows = [values[0, 2 * c + j, h] for h in range(3) for j in range(2)]
        torch.testing.assert_close(output[0, c], gelu(torch.stack(rows)), rtol=0, atol=0)


@pytest.mark.parametrize(
    "arguments,message",
    [
        ((0, 2), "in_channels must be a positive integer"),
        ((True, 2), "in_channels must be a positive integer"),
        ((2, 0), "out_channels must be a positive integer"),
        ((2, 1.0), "out_channels must be a positive integer"),
    ],
)
def test_time_block_rejects_invalid_channels(arguments, message):
    with pytest.raises(ValueError, match=message):
        NersiTimeBlock(*arguments)


@pytest.mark.parametrize("kernel_size,message", [(2, "odd"), (0, "positive"), (True, "positive")])
def test_time_block_rejects_invalid_kernels(kernel_size, message):
    with pytest.raises(ValueError, match=message):
        NersiTimeBlock(2, 2, kernel_size=kernel_size)


@pytest.mark.parametrize("shape", [(1, 2, 4, 4), (3, 4, 4), (1, 3, 1, 4, 4)])
def test_time_block_rejects_values_of_another_shape(shape):
    with pytest.raises(ValueError, match=r"values must have shape \(B, 3, H, W\)"):
        NersiTimeBlock(3, 2)(torch.zeros(shape))


# Nersi with time-only stages


def test_time_stages_shorten_only_the_time_side_of_the_latent_grid():
    model = _time_model()

    output = model(torch.rand(3, 3))

    assert output.shape == (3, 1, 64, 16)
    assert torch.isfinite(output).all()
    # Time 64 / (8 * 2**2) and receiver 16 / 8.
    assert model.base_profile_shape == (2, 2)
    assert model.decoder_profile_shape == (64, 16)
    assert model.encoder[2].out_features == 4 * 2 * 2


def test_parameter_count_follows_encoder_time_isotropic_and_output_formula():
    model = _time_model(kernel_size=5, time_kernel_size=3)

    assert _parameter_count(model) == _expected_parameters(
        fourier=3,
        encoder=12,
        latent=4,
        base=(2, 2),
        decoder=(4, 3, 2),
        kernel=5,
        time_channels=(5, 6),
        time_kernel=3,
    )


def test_time_blocks_precede_the_isotropic_blocks_in_the_decoder():
    model = _time_model(kernel_size=3, time_kernel_size=1)
    shapes = []
    for block in model.decoder:
        block.register_forward_hook(lambda _, __, output: shapes.append(tuple(output.shape[1:])))

    model(torch.rand(2, 3))

    assert [type(block) for block in model.decoder] == [NersiTimeBlock] * 2 + [NersiBlock] * 3
    assert [(b.in_channels, b.out_channels) for b in model.decoder] == [
        (4, 5),
        (5, 6),
        (6, 4),
        (4, 3),
        (3, 2),
    ]
    assert [b.convolution.kernel_size for b in model.decoder] == [(1, 1)] * 2 + [(3, 3)] * 3
    # Time doubles alone twice from the (2, 2) latent, then both sides double.
    assert shapes == [(5, 4, 2), (6, 8, 2), (4, 16, 4), (3, 32, 8), (2, 64, 16)]
    assert [name for name in model.state_dict() if name.startswith("decoder.")] == [
        f"decoder.{index}.convolution.{kind}" for index in range(5) for kind in ("weight", "bias")
    ]


def test_time_kernel_size_defaults_to_the_decoder_kernel_size():
    model = _time_model(kernel_size=5, time_kernel_size=None)

    assert model.time_kernel_size == 5
    assert model.constructor_config()["time_kernel_size"] == 5
    assert all(block.kernel_size == 5 for block in model.decoder)


def test_constructor_config_round_trip_restores_the_same_function():
    torch.manual_seed(11)
    model = _time_model(time_upsample_channels=[5, 6], kernel_size=5, time_kernel_size=3)
    config = model.constructor_config()
    coordinates = torch.rand(4, 3)

    torch.manual_seed(11)
    rebuilt = Nersi(**config)
    restored = Nersi(**config)
    restored.load_state_dict(model.state_dict(), strict=True)

    assert config["time_upsample_channels"] == (5, 6)
    assert config["time_kernel_size"] == 3
    assert rebuilt.constructor_config() == config
    assert [(k, v.shape) for k, v in rebuilt.state_dict().items()] == [
        (k, v.shape) for k, v in model.state_dict().items()
    ]
    # Same seed, same construction order: identical initialization.
    assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in model.state_dict().items())
    torch.testing.assert_close(restored(coordinates), model(coordinates), rtol=0, atol=0)


def test_rank_two_latent_combines_with_time_stages():
    model = _time_model(latent_spatial_rank=2)
    coordinates = torch.rand(3, 3, requires_grad=True)

    output = model(coordinates)
    output.square().mean().backward()
    restored = Nersi(**model.constructor_config())

    assert output.shape == (3, 1, 64, 16)
    assert model.base_profile_shape == (2, 2)
    # Rank-2 separable latent: channels * rank * (base time + base receiver).
    assert model.encoder[2].out_features == 4 * 2 * (2 + 2)
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert restored.constructor_config() == model.constructor_config()
    assert _parameter_count(model) == _expected_parameters(
        fourier=3,
        encoder=12,
        latent=4,
        base=(2, 2),
        decoder=(4, 3, 2),
        kernel=3,
        time_channels=(5, 6),
        time_kernel=1,
        rank=2,
    )


def test_temporal_basis_decodes_a_time_side_divisible_by_the_time_scale():
    model = _time_model(
        profile_shape=(64, 8), time_upsample_channels=(5,), temporal_basis_components=32
    )

    assert model(torch.rand(2, 3)).shape == (2, 1, 64, 8)
    assert model.base_profile_shape == (2, 1)


def test_time_kernel_size_requires_time_stages():
    with pytest.raises(ValueError, match="time_kernel_size requires time_upsample_channels"):
        _tiny_model(time_kernel_size=3)


@pytest.mark.parametrize(
    "channels,message",
    [
        ((), "nonempty sequence"),
        ([], "nonempty sequence"),
        ("44", "nonempty sequence"),
        (4, "nonempty sequence"),
        ((4, 0), "positive integer"),
        ((4, -2), "positive integer"),
        ((True,), "positive integer"),
        ((2.0,), "positive integer"),
    ],
)
def test_time_upsample_channels_must_be_positive_integers(channels, message):
    with pytest.raises(ValueError, match=f"time_upsample_channels must be a {message}"):
        _time_model(time_upsample_channels=channels)


@pytest.mark.parametrize("kernel_size,message", [(2, "odd"), (0, "positive"), (False, "positive")])
def test_time_kernel_size_must_be_a_positive_odd_integer(kernel_size, message):
    with pytest.raises(ValueError, match=message):
        _time_model(time_kernel_size=kernel_size)


@pytest.mark.parametrize(
    "profile_shape,channels,temporal_basis",
    [
        ((32, 8), (4, 4, 4), None),  # 32 % (8 * 2**3)
        ((48, 8), (4, 4), None),  # 48 % (8 * 2**2)
        ((24, 8), (4,), None),  # 24 % (8 * 2**1)
        ((64, 8), (4,), 24),  # decoded time 24 % 16
    ],
)
def test_decoded_time_must_be_divisible_by_the_time_upsample_scale(
    profile_shape, channels, temporal_basis
):
    with pytest.raises(ValueError, match="divisible by the total time upsample scale"):
        _time_model(
            profile_shape=profile_shape,
            time_upsample_channels=channels,
            temporal_basis_components=temporal_basis,
        )


def test_receiver_side_still_needs_only_the_isotropic_scale():
    model = _time_model(profile_shape=(32, 8), time_upsample_channels=(4, 4))

    assert model(torch.rand(1, 3)).shape == (1, 1, 32, 8)
    assert model.base_profile_shape == (1, 1)


# Backward compatibility without time-only stages


def test_default_model_keeps_the_isotropic_module_tree_and_parameter_names():
    model = _tiny_model()

    expected = [
        ("fourier_mapping.frequencies", (3,)),
        ("encoder.0.weight", (12, 18)),
        ("encoder.0.bias", (12,)),
        ("encoder.2.weight", (8, 12)),
        ("encoder.2.bias", (8,)),
        ("decoder.0.convolution.weight", (16, 4, 3, 3)),
        ("decoder.0.convolution.bias", (16,)),
        ("decoder.1.convolution.weight", (12, 4, 3, 3)),
        ("decoder.1.convolution.bias", (12,)),
        ("decoder.2.convolution.weight", (8, 3, 3, 3)),
        ("decoder.2.convolution.bias", (8,)),
        ("output_convolution.weight", (1, 2, 1, 1)),
        ("output_convolution.bias", (1,)),
    ]
    assert [(k, tuple(v.shape)) for k, v in model.state_dict().items()] == expected
    assert _parameter_count(model) == 1595
    assert _parameter_count(model) == _expected_parameters(
        fourier=3, encoder=12, latent=4, base=(2, 1), decoder=(4, 3, 2), kernel=3
    )
    assert [type(block) for block in model.decoder] == [NersiBlock] * 3
    assert not any(isinstance(module, NersiTimeBlock) for module in model.modules())
    assert model.base_profile_shape == (2, 1)
    assert model.time_upsample_channels is None and model.time_kernel_size is None


@pytest.mark.parametrize("changes", [{}, {"latent_spatial_rank": 2}, {"kernel_size": 5}])
def test_default_constructor_config_has_no_time_keys_and_matches_explicit_none(changes):
    torch.manual_seed(5)
    model = _tiny_model(**changes)
    torch.manual_seed(5)
    explicit = _tiny_model(**changes, time_upsample_channels=None, time_kernel_size=None)

    config = model.constructor_config()
    assert "time_upsample_channels" not in config
    assert "time_kernel_size" not in config
    assert explicit.constructor_config() == config
    assert list(explicit.state_dict()) == list(model.state_dict())
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in model.state_dict().items())


# FORGE profile helpers


@pytest.mark.parametrize(
    "length,multiple,expected",
    [
        (1, 8, 8),
        (8, 8, 8),
        (9, 8, 16),
        (2001, 8, 2008),
        (2001, 32, 2016),
        (2001, 128, 2048),
        (4343, 8, 4344),
        (4343, 128, 4352),
        (4352, 128, 4352),
    ],
)
def test_profile_side_is_the_smallest_covering_multiple(length, multiple, expected):
    assert profile_side(length, multiple) == expected


def test_profile_side_defaults_to_the_isotropic_multiple_of_eight():
    assert profile_side(4343) == 4344
    assert profile_side(30) == 32


@pytest.mark.parametrize(
    "model,expected",
    [
        ({}, 8),
        ({"time_upsample_channels": None}, 8),
        ({"time_upsample_channels": [5]}, 16),
        ({"time_upsample_channels": [28, 28, 28, 28]}, 128),
    ],
)
def test_profile_time_multiple_adds_one_factor_two_per_time_stage(model, expected):
    assert profile_time_multiple(model) == expected


@pytest.mark.parametrize(
    "model_changes,time_side,base_profile_shape",
    [({}, 4344, (543, 4)), (DEEP_TIME_MODEL, 4352, (34, 4))],
)
def test_canvas_config_pads_the_time_side_to_the_decoder_time_multiple(
    model_changes, time_side, base_profile_shape
):
    config = _forge_config(model_changes)
    original = deepcopy(config)

    resolved = canvas_method_config(config, CANVAS)

    assert resolved["model"]["profile_shape"] == [time_side, 32]
    assert {k: v for k, v in resolved["model"].items() if k != "profile_shape"} == {
        k: v for k, v in config["model"].items() if k != "profile_shape"
    }
    assert {k: v for k, v in resolved.items() if k != "model"} == {
        k: v for k, v in config.items() if k != "model"
    }
    assert config == original
    resolved["model"]["profile_shape"][1] = -1
    resolved["model"].get("time_upsample_channels", []).append(-1)
    assert config == original
    assert initialize_model(canvas_method_config(config, CANVAS)).base_profile_shape == (
        base_profile_shape
    )


def test_deep_time_model_rejects_the_isotropic_canvas_side():
    model = {**_forge_config(DEEP_TIME_MODEL)["model"], "profile_shape": [4344, 32]}

    with pytest.raises(ValueError, match="divisible by the total time upsample scale"):
        Nersi(**model)


@pytest.mark.parametrize(
    "model_changes,expected",
    [
        # Center of study_061 (no time stages); its README records 360,239.
        ({}, 360239),
        (DEEP_TIME_MODEL, 355653),
        (RANK2_MODEL, 353755),
    ],
)
def test_study_061_parameter_counts_at_the_aligned_canvas(model_changes, expected):
    assert model_parameter_count(_forge_config(model_changes), CANVAS) == expected


def test_deep_time_parameter_count_follows_the_formula():
    assert model_parameter_count(_forge_config(DEEP_TIME_MODEL), CANVAS) == _expected_parameters(
        fourier=40,
        encoder=40,
        latent=28,
        base=(34, 4),
        decoder=(29, 14, 8),
        kernel=5,
        time_channels=(28, 28, 28, 28),
        time_kernel=3,
    )
    assert model_parameter_count(_forge_config(RANK2_MODEL), CANVAS) == _expected_parameters(
        fourier=40, encoder=30, latent=8, base=(543, 4), decoder=(29, 14, 8), kernel=5, rank=2
    )


@pytest.mark.parametrize(
    "model_changes,profile_shape",
    [
        ({}, [2008, 32]),
        (RANK2_MODEL, [2008, 32]),
        ({"time_upsample_channels": [8, 8]}, [2016, 32]),
        (DEEP_TIME_MODEL, [2048, 32]),
    ],
)
def test_added_method_accepts_the_profile_shape_of_its_time_multiple(model_changes, profile_shape):
    config = _forge_config({**model_changes, "profile_shape": profile_shape})

    validate_added_method_config(config, CONTRACT)


@pytest.mark.parametrize(
    "model_changes,profile_shape",
    [
        ({}, [2016, 32]),
        ({"time_upsample_channels": [8, 8]}, [2008, 32]),
        (DEEP_TIME_MODEL, [2008, 32]),
        (DEEP_TIME_MODEL, [2016, 32]),
        (DEEP_TIME_MODEL, [2048, 40]),
    ],
)
def test_added_method_rejects_a_profile_shape_of_another_multiple(model_changes, profile_shape):
    config = _forge_config({**model_changes, "profile_shape": profile_shape})

    with pytest.raises(ValueError, match="complete traces and receiver lines"):
        validate_added_method_config(config, CONTRACT)
