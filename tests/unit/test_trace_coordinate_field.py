"""The implicit trace field returns a complete trace and discards no output sample.

This is the property the field exists for: the NeRSI trace decoder builds an
eight-column canvas through three pixel-shuffle stages and returns one column, so
seven eighths of its decoder output never reaches a trace. Here the output layer
emits exactly ``time_sample_count`` values.
"""

import pytest
import torch
from torch import nn

from seis_interp.models.trace_coordinate_field import (
    TraceCoordinateField,
    field_parameter_count,
    matched_hidden_width,
)

TARGET = 363269
TOLERANCE = 0.05
CANVASES = ((384, 249, 362679), (2172, 126, 364296))
UPSAMPLING = (
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.PixelShuffle,
    nn.Upsample,
)


def _field(**overrides):
    settings = {
        "time_sample_count": 24,
        "fourier_components": 3,
        "hidden_width": 8,
        "hidden_layers": 2,
    }
    return TraceCoordinateField(**(settings | overrides))


def test_four_coordinates_map_to_the_whole_trace():
    model = _field(time_sample_count=17)
    output = model(torch.rand(5, 4))
    assert output.shape == (5, 17)
    assert model.output_layer.out_features == 17


def test_no_output_sample_is_discarded_and_no_canvas_is_built():
    model = _field()
    assert not [layer for layer in model.modules() if isinstance(layer, UPSAMPLING)]
    assert sum(1 for layer in model.modules() if isinstance(layer, nn.Linear)) == 3


def test_every_parameter_reaches_the_returned_trace():
    model = _field()
    model(torch.rand(4, 4)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_the_fourier_encoding_is_parameter_free_with_the_declared_width():
    model = _field(fourier_components=7)
    assert model.encoding.output_features == 4 * 2 * 7
    assert list(model.encoding.parameters()) == []


def test_parameter_count_matches_the_closed_form():
    settings = {
        "time_sample_count": 2172,
        "fourier_components": 40,
        "hidden_width": 126,
        "hidden_layers": 4,
    }
    model = TraceCoordinateField(**settings)
    assert sum(p.numel() for p in model.parameters()) == field_parameter_count(**settings)


@pytest.mark.parametrize("base", [1.05, 1.075, 1.1, 1.15, 1.25])
def test_the_fourier_band_never_changes_the_parameter_count(base):
    model = _field(frequency_base=base)
    assert sum(p.numel() for p in model.parameters()) == field_parameter_count(
        time_sample_count=24, fourier_components=3, hidden_width=8, hidden_layers=2
    )


@pytest.mark.parametrize("canvas,width,count", CANVASES)
def test_the_matched_width_meets_the_budget_on_each_canvas(canvas, width, count):
    solved, solved_count = matched_hidden_width(
        time_sample_count=canvas,
        fourier_components=40,
        hidden_layers=4,
        parameter_count_target=TARGET,
    )
    assert (solved, solved_count) == (width, count)
    assert abs(count / TARGET - 1) <= TOLERANCE


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"input_features": 3}, "four coordinates"),
        ({"input_features": 5}, "four coordinates"),
        ({"activation": "relu"}, "GELU and linear"),
        ({"output_activation": "tanh"}, "GELU and linear"),
        ({"time_sample_count": 0}, "time_sample_count"),
        ({"hidden_width": 0}, "hidden_width"),
        ({"hidden_layers": 0}, "hidden_layers"),
        ({"hidden_layers": True}, "hidden_layers"),
    ],
)
def test_rejected_constructor_arguments(overrides, message):
    with pytest.raises(ValueError, match=message):
        _field(**overrides)
