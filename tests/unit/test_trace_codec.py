"""Contract tests for the per-trace temporal encoder and decoder."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from seis_interp.models import (
    TraceNodeDecoder as RootTraceNodeDecoder,
)
from seis_interp.models import (
    TraceNodeEncoder as RootTraceNodeEncoder,
)
from seis_interp.models.trace_codec import TraceNodeDecoder, TraceNodeEncoder
from seis_interp.models.trace_graph_interpolator import (
    TraceGraphInterpolator,
)
from seis_interp.models.trace_graph_interpolator import (
    TraceNodeDecoder as GraphModuleTraceNodeDecoder,
)
from seis_interp.models.trace_graph_interpolator import (
    TraceNodeEncoder as GraphModuleTraceNodeEncoder,
)

WIDTH = 8
TIME = 20
FACTOR = 5
FRAMES = TIME // FACTOR


def _encoder() -> TraceNodeEncoder:
    return TraceNodeEncoder(
        WIDTH,
        stem_kernel_size=3,
        time_downsample_factor=FACTOR,
    )


def _decoder() -> TraceNodeDecoder:
    return TraceNodeDecoder(WIDTH, time_downsample_factor=FACTOR)


def _small_graph_model() -> TraceGraphInterpolator:
    return TraceGraphInterpolator(
        width=WIDTH,
        message_passing_rounds=2,
        time_downsample_factor=FACTOR,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        temporal_dilations=(1, 2),
        spatial_kernel_size=3,
        attention_width=4,
    )


def test_new_old_and_package_root_imports_are_identical() -> None:
    assert GraphModuleTraceNodeEncoder is TraceNodeEncoder
    assert RootTraceNodeEncoder is TraceNodeEncoder
    assert GraphModuleTraceNodeDecoder is TraceNodeDecoder
    assert RootTraceNodeDecoder is TraceNodeDecoder


def test_encoder_constructor_attributes_shape_and_input_immutability() -> None:
    signature = inspect.signature(TraceNodeEncoder)
    assert list(signature.parameters) == [
        "width",
        "stem_kernel_size",
        "time_downsample_factor",
    ]
    assert signature.parameters["width"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["stem_kernel_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["time_downsample_factor"].kind is inspect.Parameter.KEYWORD_ONLY

    encoder = _encoder()
    waveforms = torch.randn(6, 1, TIME)
    unchanged = waveforms.clone()
    latents = encoder(waveforms)

    assert encoder.width == WIDTH
    assert encoder.stem_kernel_size == 3
    assert encoder.time_downsample_factor == FACTOR
    assert latents.shape == (6, WIDTH, FRAMES)
    assert torch.equal(waveforms, unchanged)


def test_decoder_constructor_attributes_shape_and_input_immutability() -> None:
    signature = inspect.signature(TraceNodeDecoder)
    assert list(signature.parameters) == ["width", "time_downsample_factor"]
    assert signature.parameters["width"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["time_downsample_factor"].kind is inspect.Parameter.KEYWORD_ONLY

    decoder = _decoder()
    latents = torch.randn(6, WIDTH, FRAMES)
    unchanged = latents.clone()
    traces = decoder(latents)

    assert decoder.width == WIDTH
    assert decoder.time_downsample_factor == FACTOR
    assert traces.shape == (6, TIME)
    assert torch.equal(latents, unchanged)


def test_codec_maps_twenty_samples_to_four_frames_and_back() -> None:
    latents = _encoder()(torch.randn(3, 1, TIME))
    traces = _decoder()(latents)

    assert latents.shape == (3, WIDTH, 4)
    assert traces.shape == (3, 20)


def test_decoder_final_projection_is_exactly_zero_initialized() -> None:
    decoder = _decoder()
    final_projection = decoder.head[-1]

    assert isinstance(final_projection, nn.Conv1d)
    assert torch.equal(final_projection.weight, torch.zeros_like(final_projection.weight))
    assert torch.equal(final_projection.bias, torch.zeros_like(final_projection.bias))


@pytest.mark.parametrize("width", [0, 4, 12])
def test_codec_rejects_invalid_width(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        TraceNodeEncoder(width, stem_kernel_size=3, time_downsample_factor=FACTOR)
    with pytest.raises(ValueError, match="width"):
        TraceNodeDecoder(width, time_downsample_factor=FACTOR)


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_encoder_rejects_non_positive_or_even_stem_kernel(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="stem_kernel_size"):
        TraceNodeEncoder(
            WIDTH,
            stem_kernel_size=kernel_size,
            time_downsample_factor=FACTOR,
        )


@pytest.mark.parametrize("factor", [0, -1])
def test_codec_rejects_non_positive_downsample_factor(factor: int) -> None:
    with pytest.raises(ValueError, match="time_downsample_factor"):
        TraceNodeEncoder(WIDTH, stem_kernel_size=3, time_downsample_factor=factor)
    with pytest.raises(ValueError, match="time_downsample_factor"):
        TraceNodeDecoder(WIDTH, time_downsample_factor=factor)


@pytest.mark.parametrize("shape", [(2, TIME), (2, 2, TIME), (2, 1, 1, TIME)])
def test_encoder_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="waveforms must have shape"):
        _encoder()(torch.randn(shape))


def test_encoder_rejects_non_floating_input_and_indivisible_time() -> None:
    with pytest.raises(TypeError, match="floating-point dtype"):
        _encoder()(torch.ones(2, 1, TIME, dtype=torch.int64))
    with pytest.raises(ValueError, match="divisible"):
        _encoder()(torch.randn(2, 1, TIME + 1))


@pytest.mark.parametrize("shape", [(2, FRAMES), (2, WIDTH, FRAMES, 1)])
def test_decoder_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="latents must have shape"):
        _decoder()(torch.randn(shape))


def test_decoder_rejects_non_floating_input_and_wrong_channel_count() -> None:
    with pytest.raises(TypeError, match="floating-point dtype"):
        _decoder()(torch.ones(2, WIDTH, FRAMES, dtype=torch.int64))
    with pytest.raises(ValueError, match="latents must have shape"):
        _decoder()(torch.randn(2, WIDTH // 2, FRAMES))


def test_codec_state_dict_key_order_is_stable() -> None:
    assert list(_encoder().state_dict()) == [
        "stem.weight",
        "stem.bias",
        "norm.weight",
        "norm.bias",
        "downsample.weight",
        "downsample.bias",
    ]
    assert list(_decoder().state_dict()) == [
        "upsample.weight",
        "upsample.bias",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
        "head.4.weight",
        "head.4.bias",
    ]


def test_graph_model_preserves_codec_state_dict_prefixes() -> None:
    codec_keys = [
        name
        for name in _small_graph_model().state_dict()
        if name.startswith(("encoder.", "decoder."))
    ]
    assert codec_keys == [
        "encoder.stem.weight",
        "encoder.stem.bias",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "encoder.downsample.weight",
        "encoder.downsample.bias",
        "decoder.upsample.weight",
        "decoder.upsample.bias",
        "decoder.head.0.weight",
        "decoder.head.0.bias",
        "decoder.head.2.weight",
        "decoder.head.2.bias",
        "decoder.head.4.weight",
        "decoder.head.4.bias",
    ]


def test_same_seed_produces_exactly_equal_codec_parameters() -> None:
    torch.manual_seed(17)
    first = (_encoder(), _decoder())
    torch.manual_seed(17)
    second = (_encoder(), _decoder())

    for first_module, second_module in zip(first, second, strict=True):
        assert list(first_module.state_dict()) == list(second_module.state_dict())
        for name, first_tensor in first_module.state_dict().items():
            assert torch.equal(first_tensor, second_module.state_dict()[name])
