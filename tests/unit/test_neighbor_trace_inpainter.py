from __future__ import annotations

import pytest
import torch
from torch import nn

from seis_interp.models import (
    TEMPORAL_DILATIONS,
    NeighborTraceInpainter,
    TemporalResidualBlock,
)


def test_successful_architecture_contract_and_parameter_count() -> None:
    model = NeighborTraceInpainter(neighbor_count=104)

    assert model.width == 128
    assert model.input_channels == 212
    assert model.dilations == (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
    assert model.dilations is TEMPORAL_DILATIONS
    assert len(model.blocks) == 11
    assert [block.dilation for block in model.blocks] == list(model.dilations)
    assert model.stem.kernel_size == (15,)
    assert model.stem.padding == (7,)
    assert all(block.depthwise.kernel_size == (7,) for block in model.blocks)
    assert all(block.depthwise.groups == 128 for block in model.blocks)
    assert [block.depthwise.dilation for block in model.blocks] == [
        (dilation,) for dilation in model.dilations
    ]
    assert [block.depthwise.padding for block in model.blocks] == [
        (3 * dilation,) for dilation in model.dilations
    ]
    assert isinstance(model.head[-1], nn.Conv1d)
    assert model.head[-1].out_channels == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 983_041


def test_forward_preserves_batch_and_time_and_supports_boolean_availability() -> None:
    torch.manual_seed(0)
    model = NeighborTraceInpainter(neighbor_count=4, width=16)
    neighbors = torch.randn(3, 4, 29)
    availability = torch.tensor([[True, True, False, True], [False, True, True, False], [True] * 4])
    target_coordinates = torch.randn(3, 3)

    output = model(neighbors, availability, target_coordinates)

    assert output.shape == (3, 29)
    assert output.dtype == neighbors.dtype
    assert torch.isfinite(output).all()


def test_stem_receives_neighbors_availability_geometry_and_time_in_order() -> None:
    model = NeighborTraceInpainter(neighbor_count=2, width=8)
    neighbors = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    availability = torch.tensor([[True, False]])
    target_coordinates = torch.tensor([[0.25, -0.5, 0.75]])
    captured: list[torch.Tensor] = []

    handle = model.stem.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        model(neighbors, availability, target_coordinates)
    finally:
        handle.remove()

    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [-0.5, -0.5, -0.5],
                [0.75, 0.75, 0.75],
                [-1.0, 0.0, 1.0],
            ]
        ]
    )
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)


def test_gradients_reach_every_parameter_and_all_inputs() -> None:
    torch.manual_seed(1)
    model = NeighborTraceInpainter(neighbor_count=2, width=8).double()
    neighbors = torch.randn(2, 2, 17, dtype=torch.float64, requires_grad=True)
    availability = torch.rand(2, 2, dtype=torch.float64, requires_grad=True)
    target_coordinates = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    model(neighbors, availability, target_coordinates).square().mean().backward()

    assert neighbors.grad is not None and torch.isfinite(neighbors.grad).all()
    assert availability.grad is not None and torch.isfinite(availability.grad).all()
    assert target_coordinates.grad is not None and torch.isfinite(target_coordinates.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_temporal_residual_block_preserves_shape() -> None:
    block = TemporalResidualBlock(width=16, dilation=4)
    traces = torch.randn(2, 16, 23, requires_grad=True)

    output = block(traces)

    assert output.shape == traces.shape
    output.mean().backward()
    assert traces.grad is not None


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: NeighborTraceInpainter(neighbor_count=0), "neighbor_count"),
        (lambda: NeighborTraceInpainter(neighbor_count=True), "neighbor_count"),
        (lambda: NeighborTraceInpainter(neighbor_count=2, width=0), "width"),
        (lambda: NeighborTraceInpainter(neighbor_count=2, width=10), "divisible by 8"),
        (lambda: TemporalResidualBlock(width=8, dilation=0), "dilation"),
        (lambda: TemporalResidualBlock(width=7, dilation=1), "divisible by 8"),
    ],
)
def test_constructor_validation(constructor: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("neighbors", "availability", "coordinates", "error", "message"),
    [
        (torch.randn(2, 3), torch.ones(2, 3), torch.ones(2, 3), ValueError, "neighbors"),
        (torch.randn(2, 2, 5), torch.ones(2, 3), torch.ones(2, 3), ValueError, "channels"),
        (
            torch.randn(2, 3, 5),
            torch.ones(2, 3, 1),
            torch.ones(2, 3),
            ValueError,
            "availability",
        ),
        (
            torch.randn(2, 3, 5),
            torch.ones(2, 3),
            torch.ones(2, 4),
            ValueError,
            "target_coordinates",
        ),
        (
            torch.ones(2, 3, 5, dtype=torch.int64),
            torch.ones(2, 3),
            torch.ones(2, 3),
            TypeError,
            "floating-point",
        ),
        (
            torch.randn(2, 3, 5),
            torch.ones(2, 3, dtype=torch.int64),
            torch.ones(2, 3),
            TypeError,
            "availability",
        ),
        (
            torch.randn(2, 3, 5, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3, dtype=torch.float32),
            TypeError,
            "share the neighbors dtype",
        ),
        (
            torch.randn(2, 3, 5, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.float64),
            TypeError,
            "share the neighbors dtype",
        ),
    ],
)
def test_forward_validation(
    neighbors: torch.Tensor,
    availability: torch.Tensor,
    coordinates: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    model = NeighborTraceInpainter(neighbor_count=3, width=8)

    with pytest.raises(error, match=message):
        model(neighbors, availability, coordinates)


def test_temporal_residual_block_validates_forward_input() -> None:
    block = TemporalResidualBlock(width=8, dilation=1)

    with pytest.raises(ValueError, match="shape"):
        block(torch.randn(8, 7))
    with pytest.raises(ValueError, match="8 channels"):
        block(torch.randn(2, 7, 9))
    with pytest.raises(TypeError, match="floating-point"):
        block(torch.ones(2, 8, 9, dtype=torch.int64))
