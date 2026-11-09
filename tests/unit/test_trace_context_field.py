"""The conditioned field reads its neighbours and is not confined to a subspace.

Both properties answer measured failures of the unconditional field: it held the
wavefield in its weights, and its single linear output layer confined every
prediction to a subspace of its own width.
"""

import pytest
import torch
from torch import nn

from seis_interp.models.trace_context_field import TraceContextField

TARGET = 363269
TIME = 24
SETTINGS = {
    "fourier_components": 4,
    "width": 8,
    "stem_kernel_size": 3,
    "time_downsample_factor": 2,
}


def _field(**overrides):
    return TraceContextField(time_sample_count=TIME, **(SETTINGS | overrides))


def _batch(model, batch=5, context=3, time_count=TIME, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, 4, generator=generator),
        torch.randn(batch, context, time_count, generator=generator),
        torch.rand(batch, context, 4, generator=generator),
    )


def test_a_query_and_its_neighbours_decode_the_whole_trace():
    model = _field()
    assert model(*_batch(model)).shape == (5, TIME)


def test_the_parameter_count_does_not_depend_on_the_time_axis():
    counts = {
        sum(p.numel() for p in TraceContextField(time_sample_count=time, **SETTINGS).parameters())
        for time in (24, 384, 2172)
    }
    assert len(counts) == 1


def test_the_production_width_meets_the_matched_budget_on_every_canvas():
    for canvas in (384, 2172):
        model = TraceContextField(time_sample_count=canvas, fourier_components=40, width=120)
        count = sum(p.numel() for p in model.parameters())
        assert count == 362642
        assert abs(count / TARGET - 1) <= 0.05


@pytest.fixture(scope="module")
def trained():
    """Briefly fit a tiny field so the zero-initialized decoder head is no longer zero."""
    model = _field()
    coordinates, values, context = _batch(model, batch=TIME + 8)
    target = torch.randn(coordinates.shape[0], TIME, generator=torch.Generator().manual_seed(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        (model(coordinates, values, context) - target).square().mean().backward()
        optimizer.step()
    model.eval()
    return model, (coordinates, values, context)


def test_the_output_is_not_confined_to_the_hidden_width(trained):
    """A convolutional decoder leaves the output rank free; a linear head would not."""
    model, batch = trained
    assert not [
        layer
        for layer in model.decoder.modules()
        if isinstance(layer, nn.Linear) and layer.out_features == TIME
    ]
    with torch.no_grad():
        rank = torch.linalg.matrix_rank(model(*batch).double())
    assert int(rank) > model.width


def test_the_decoder_starts_at_the_zero_predictor_and_then_trains():
    model = _field()
    coordinates, values, context = _batch(model)
    assert model(coordinates, values, context).abs().max() == 0
    target = torch.randn(coordinates.shape[0], TIME, generator=torch.Generator().manual_seed(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    live = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(coordinates, values, context) - target).square().mean().backward()
        live.append(sum(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        optimizer.step()
    total = sum(1 for _ in model.parameters())
    assert live[0] == 2 and live[1] == total


@pytest.mark.parametrize("perturbed", ["values", "context", "coordinates"])
def test_the_prediction_depends_on_every_input(trained, perturbed):
    model, (coordinates, values, context) = trained
    with torch.no_grad():
        base = model(coordinates, values, context)
        changed = {
            "values": (coordinates, values.flip(0), context),
            "context": (coordinates, values, context.flip(0)),
            "coordinates": (coordinates.flip(0), values, context),
        }[perturbed]
        assert not torch.allclose(model(*changed), base)


@pytest.mark.parametrize(
    "overrides,arguments,message",
    [
        ({"input_features": 3}, None, "four coordinates"),
        ({"activation": "relu"}, None, "GELU and linear"),
        ({"output_activation": "tanh"}, None, "GELU and linear"),
        ({"time_downsample_factor": 5}, None, "divisible by time_downsample_factor"),
        (None, "batch", "every query needs its own context"),
        (None, "context", "same neighbours"),
        (None, "time", "span the declared time axis"),
        (None, "rank", "expected"),
    ],
)
def test_rejected_shapes_and_arguments(overrides, arguments, message):
    if overrides is not None:
        with pytest.raises(ValueError, match=message):
            _field(**overrides)
        return
    model = _field()
    coordinates, values, context = _batch(model)
    broken = {
        "batch": (coordinates[:2], values, context),
        "context": (coordinates, values[:, :2], context),
        "time": (coordinates, values[:, :, :-2], context),
        "rank": (coordinates, values[0], context),
    }[arguments]
    with pytest.raises(ValueError, match=message):
        model(*broken)
