from __future__ import annotations

import pytest
import torch
from torch import nn

from seis_interp.models import (
    TEMPORAL_DILATIONS,
    NeighborTraceInpainter,
    TemporalResidualBlock,
)
from seis_interp.models.neighbor_trace_inpainter import (
    _availability_masked_softmax_gates,
)


def test_successful_architecture_contract_and_parameter_count() -> None:
    model = NeighborTraceInpainter(neighbor_count=104)

    assert model.width == 128
    assert model.target_coordinate_count == 3
    assert model.coordinate_conditioning == "stem"
    assert model.neighbor_gating == "none"
    assert model.neighbor_gate_projection is None
    assert model.stem_kernel_size == 15
    assert model.residual_kernel_size == 7
    assert model.temporal_dilations == TEMPORAL_DILATIONS
    assert model.input_channels == 212
    assert model.dilations == (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
    assert model.dilations is TEMPORAL_DILATIONS
    assert len(model.blocks) == 11
    assert len(model.coordinate_modulations) == 0
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


def test_custom_architecture_preserves_configured_shapes_and_provenance() -> None:
    model = NeighborTraceInpainter(
        neighbor_count=2,
        width=8,
        target_coordinate_count=5,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1, 3, 2),
    )

    output = model(
        torch.randn(4, 2, 19),
        torch.ones(4, 2, dtype=torch.bool),
        torch.randn(4, 5),
    )

    assert output.shape == (4, 19)
    assert model.input_channels == 10
    assert model.target_coordinate_count == 5
    assert model.stem_kernel_size == 5
    assert model.residual_kernel_size == 3
    assert model.temporal_dilations == (1, 3, 2)
    assert model.dilations == model.temporal_dilations
    assert model.stem.kernel_size == (5,)
    assert model.stem.padding == (2,)
    assert [block.kernel_size for block in model.blocks] == [3, 3, 3]
    assert [block.dilation for block in model.blocks] == [1, 3, 2]
    assert [block.depthwise.padding for block in model.blocks] == [(1,), (3,), (2,)]


def test_film_conditioning_preserves_shapes_and_adds_one_projection_per_block() -> None:
    model = NeighborTraceInpainter(
        neighbor_count=2,
        width=8,
        target_coordinate_count=4,
        temporal_dilations=(1, 3),
        coordinate_conditioning="film",
    )

    output = model(
        torch.randn(3, 2, 17),
        torch.ones(3, 2, dtype=torch.bool),
        torch.randn(3, 4),
    )

    assert output.shape == (3, 17)
    assert model.coordinate_conditioning == "film"
    assert model.input_channels == 9
    assert model.stem.in_channels == 9
    assert len(model.coordinate_modulations) == 2
    assert all(projection.in_features == 4 for projection in model.coordinate_modulations)
    assert all(projection.out_features == 16 for projection in model.coordinate_modulations)


def test_film_conditioning_is_neutral_at_initialization() -> None:
    torch.manual_seed(5)
    stem_model = NeighborTraceInpainter(
        neighbor_count=2,
        width=8,
        temporal_dilations=(1,),
    )
    torch.manual_seed(5)
    film_model = NeighborTraceInpainter(
        neighbor_count=2,
        width=8,
        temporal_dilations=(1,),
        coordinate_conditioning="film",
    )
    projection = film_model.coordinate_modulations[0]
    traces = torch.randn(3, 8, 13)
    neighbors = torch.randn(3, 2, 13)
    availability = torch.ones(3, 2, dtype=torch.bool)
    coordinates = torch.randn(3, 3)

    modulation = projection(coordinates)

    assert torch.count_nonzero(projection.weight) == 0
    assert torch.count_nonzero(projection.bias) == 0
    assert torch.count_nonzero(modulation) == 0
    torch.testing.assert_close(
        film_model.blocks[0](traces, modulation),
        film_model.blocks[0](traces),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        film_model(neighbors, availability, coordinates),
        stem_model(neighbors, availability, coordinates),
        rtol=0.0,
        atol=0.0,
    )


def test_film_projection_parameters_receive_gradients() -> None:
    torch.manual_seed(7)
    model = NeighborTraceInpainter(
        neighbor_count=2,
        width=8,
        temporal_dilations=(1, 2),
        coordinate_conditioning="film",
    ).double()
    neighbors = torch.randn(2, 2, 17, dtype=torch.float64, requires_grad=True)
    availability = torch.rand(2, 2, dtype=torch.float64, requires_grad=True)
    coordinates = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    model(neighbors, availability, coordinates).square().mean().backward()

    for projection in model.coordinate_modulations:
        assert projection.weight.grad is not None
        assert projection.bias.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.isfinite(projection.bias.grad).all()
        assert torch.count_nonzero(projection.weight.grad) > 0
        assert torch.count_nonzero(projection.bias.grad) > 0


def test_masked_softmax_neighbor_gate_is_neutral_at_initialization() -> None:
    torch.manual_seed(23)
    plain = NeighborTraceInpainter(neighbor_count=3, width=8, temporal_dilations=(1,))
    torch.manual_seed(23)
    gated = NeighborTraceInpainter(
        neighbor_count=3,
        width=8,
        temporal_dilations=(1,),
        neighbor_gating="target_coordinate_masked_softmax",
    )
    availability = torch.tensor([[True, False, True], [True, True, True]])
    neighbors = torch.randn(2, 3, 11) * availability[..., None]
    coordinates = torch.randn(2, 3)

    assert gated.neighbor_gate_projection is not None
    assert torch.count_nonzero(gated.neighbor_gate_projection.weight) == 0
    assert torch.count_nonzero(gated.neighbor_gate_projection.bias) == 0
    for name, tensor in plain.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[name], tensor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gated(neighbors, availability, coordinates),
        plain(neighbors, availability, coordinates),
        rtol=0.0,
        atol=0.0,
    )


def test_masked_softmax_neighbor_gates_mask_and_preserve_available_mean() -> None:
    logits = torch.tensor([[2.0, -3.0, 1.0], [7.0, 8.0, 9.0]], requires_grad=True)
    availability = torch.tensor([[True, False, True], [False, False, False]])

    gates = _availability_masked_softmax_gates(logits, availability)

    assert torch.isfinite(gates).all()
    assert gates[0, 1] == 0.0
    assert float(gates[0].sum().detach()) == pytest.approx(2.0)
    torch.testing.assert_close(gates[1], torch.zeros(3))
    gates.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_neighbor_gate_changes_only_amplitude_channels_and_handles_no_neighbors() -> None:
    model = NeighborTraceInpainter(
        neighbor_count=3,
        width=8,
        temporal_dilations=(1,),
        neighbor_gating="target_coordinate_masked_softmax",
    )
    assert model.neighbor_gate_projection is not None
    with torch.no_grad():
        model.neighbor_gate_projection.weight.copy_(torch.eye(3))
    neighbors = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    availability = torch.tensor([[True, False, True], [False, False, False]])
    coordinates = torch.tensor([[2.0, 9.0, 0.0], [1.0, 2.0, 3.0]])
    captured: list[torch.Tensor] = []
    handle = model.stem.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        output = model(neighbors, availability, coordinates)
    finally:
        handle.remove()

    expected_gates = _availability_masked_softmax_gates(coordinates, availability)
    torch.testing.assert_close(captured[0][:, :3], neighbors * expected_gates[..., None])
    torch.testing.assert_close(
        captured[0][:, 3:6],
        availability.to(dtype=neighbors.dtype)[..., None].expand(-1, -1, 2),
    )
    assert torch.isfinite(captured[0]).all()
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.neighbor_gate_projection.weight.grad is not None
    assert torch.isfinite(model.neighbor_gate_projection.weight.grad).all()
    assert torch.count_nonzero(model.neighbor_gate_projection.weight.grad) > 0


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
    block = TemporalResidualBlock(width=16, dilation=4, kernel_size=5)
    traces = torch.randn(2, 16, 23, requires_grad=True)

    output = block(traces)

    assert output.shape == traces.shape
    assert block.kernel_size == 5
    assert block.depthwise.padding == (8,)
    output.mean().backward()
    assert traces.grad is not None


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: NeighborTraceInpainter(neighbor_count=0), "neighbor_count"),
        (lambda: NeighborTraceInpainter(neighbor_count=True), "neighbor_count"),
        (lambda: NeighborTraceInpainter(neighbor_count=2, width=0), "width"),
        (lambda: NeighborTraceInpainter(neighbor_count=2, width=10), "divisible by 8"),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, target_coordinate_count=0),
            "target_coordinate_count",
        ),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, target_coordinate_count=True),
            "target_coordinate_count",
        ),
        (lambda: NeighborTraceInpainter(neighbor_count=2, stem_kernel_size=0), "stem_kernel_size"),
        (lambda: NeighborTraceInpainter(neighbor_count=2, stem_kernel_size=4), "odd"),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, residual_kernel_size=2),
            "odd",
        ),
        (lambda: NeighborTraceInpainter(neighbor_count=2, temporal_dilations=()), "not be empty"),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, temporal_dilations=(1, 0)),
            "temporal_dilations\\[1\\]",
        ),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, temporal_dilations=(True,)),
            "temporal_dilations\\[0\\]",
        ),
        (
            lambda: NeighborTraceInpainter(
                neighbor_count=2,
                coordinate_conditioning="concatenate",
            ),
            "coordinate_conditioning",
        ),
        (
            lambda: NeighborTraceInpainter(
                neighbor_count=2,
                coordinate_conditioning=1,  # type: ignore[arg-type]
            ),
            "coordinate_conditioning",
        ),
        (
            lambda: NeighborTraceInpainter(neighbor_count=2, neighbor_gating="sigmoid"),
            "neighbor_gating",
        ),
        (
            lambda: NeighborTraceInpainter(
                neighbor_count=2,
                neighbor_gating=1,  # type: ignore[arg-type]
            ),
            "neighbor_gating",
        ),
        (lambda: TemporalResidualBlock(width=8, dilation=0), "dilation"),
        (lambda: TemporalResidualBlock(width=8, dilation=1, kernel_size=2), "odd"),
        (lambda: TemporalResidualBlock(width=8, dilation=1, kernel_size=True), "kernel_size"),
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


def test_temporal_residual_block_validates_optional_modulation() -> None:
    block = TemporalResidualBlock(width=8, dilation=1)
    traces = torch.randn(2, 8, 9)

    with pytest.raises(TypeError, match="torch.Tensor"):
        block(traces, modulation=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        block(traces, modulation=torch.ones(2, 16, dtype=torch.int64))
    with pytest.raises(ValueError, match=r"shape \(2, 16\)"):
        block(traces, modulation=torch.randn(2, 8))
    with pytest.raises(TypeError, match="share the traces dtype"):
        block(traces, modulation=torch.randn(2, 16, dtype=torch.float64))
