"""Unit tests for the trace-node graph gather interpolator."""

from __future__ import annotations

import pytest
import torch

from seis_interp.models.shot_gather_inpainter import inverse_distance_reference
from seis_interp.models.trace_graph_interpolator import (
    SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE,
    TRACE_LATTICE_GRAPH_MODE,
    TraceGraphInterpolator,
    TraceNodeDecoder,
    TraceNodeEncoder,
)

BATCH = 2
SOURCES = 3
TIME = 20


def _small_model(graph_mode: str) -> TraceGraphInterpolator:
    return TraceGraphInterpolator(
        width=8,
        graph_mode=graph_mode,
        message_passing_rounds=2,
        time_downsample_factor=5,
        stem_kernel_size=3,
        temporal_kernel_size=3,
        temporal_dilations=(1, 2),
        spatial_kernel_size=3,
        attention_width=4,
    )


def _inputs(
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if generator is None:
        generator = torch.Generator().manual_seed(7)
    neighbors = torch.randn(BATCH, SOURCES, 8, 68, TIME, generator=generator)
    availability = torch.rand(BATCH, SOURCES, 8, 68, generator=generator) > 0.3
    source_deltas = torch.randn(BATCH, SOURCES, 2, generator=generator) * 100.0 + 200.0
    target_coordinates = torch.rand(BATCH, 2, generator=generator)
    return neighbors, availability, source_deltas, target_coordinates


@pytest.mark.parametrize(
    "graph_mode",
    [TRACE_LATTICE_GRAPH_MODE, SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE],
)
def test_forward_shape_and_zero_initialization(graph_mode: str) -> None:
    torch.manual_seed(0)
    model = _small_model(graph_mode)
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    output = model(neighbors, availability, source_deltas, target_coordinates)
    assert output.shape == (BATCH, 8, 68, TIME)
    reference = inverse_distance_reference(neighbors, availability, source_deltas)
    assert torch.allclose(output, reference)


@pytest.mark.parametrize(
    "graph_mode",
    [TRACE_LATTICE_GRAPH_MODE, SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE],
)
def test_forward_is_source_permutation_invariant(graph_mode: str) -> None:
    torch.manual_seed(0)
    model = _small_model(graph_mode)
    _train_one_step(model)
    model.eval()
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        direct = model(neighbors, availability, source_deltas, target_coordinates)
        permuted = model(
            neighbors[:, permutation],
            availability[:, permutation],
            source_deltas[:, permutation],
            target_coordinates,
        )
    assert torch.allclose(direct, permuted, atol=1.0e-5)


@pytest.mark.parametrize(
    "graph_mode",
    [TRACE_LATTICE_GRAPH_MODE, SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE],
)
def test_masked_neighbor_amplitudes_cannot_influence_output(graph_mode: str) -> None:
    torch.manual_seed(0)
    model = _small_model(graph_mode)
    _train_one_step(model)
    model.eval()
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    corrupted = neighbors.clone()
    corrupted[~availability[..., None].expand_as(corrupted)] = 1.0e6
    with torch.no_grad():
        clean_output = model(neighbors, availability, source_deltas, target_coordinates)
        corrupted_output = model(corrupted, availability, source_deltas, target_coordinates)
    assert torch.allclose(clean_output, corrupted_output, atol=1.0e-5)


@pytest.mark.parametrize(
    "graph_mode",
    [TRACE_LATTICE_GRAPH_MODE, SOURCE_RECEIVER_BIPARTITE_GRAPH_MODE],
)
def test_training_step_updates_produce_nonreference_output(graph_mode: str) -> None:
    torch.manual_seed(0)
    model = _small_model(graph_mode)
    _train_one_step(model)
    _train_one_step(model)
    model.eval()
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    with torch.no_grad():
        output = model(neighbors, availability, source_deltas, target_coordinates)
        reference = inverse_distance_reference(neighbors, availability, source_deltas)
    assert not torch.allclose(output, reference)


def test_encoder_and_decoder_are_exact_time_inverses_in_shape() -> None:
    encoder = TraceNodeEncoder(8, stem_kernel_size=3, time_downsample_factor=5)
    decoder = TraceNodeDecoder(8, time_downsample_factor=5)
    waveforms = torch.randn(6, 1, TIME)
    latents = encoder(waveforms)
    assert latents.shape == (6, 8, TIME // 5)
    residual = decoder(latents)
    assert residual.shape == (6, TIME)


def test_rejects_time_not_divisible_by_downsample_factor() -> None:
    model = _small_model(TRACE_LATTICE_GRAPH_MODE)
    generator = torch.Generator().manual_seed(3)
    neighbors = torch.randn(1, SOURCES, 8, 68, 21, generator=generator)
    availability = torch.ones(1, SOURCES, 8, 68, dtype=torch.bool)
    source_deltas = torch.randn(1, SOURCES, 2, generator=generator) + 50.0
    target_coordinates = torch.rand(1, 2, generator=generator)
    with pytest.raises(ValueError, match="divisible"):
        model(neighbors, availability, source_deltas, target_coordinates)


def test_rejects_non_boolean_availability() -> None:
    model = _small_model(TRACE_LATTICE_GRAPH_MODE)
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    with pytest.raises(TypeError, match="torch.bool"):
        model(neighbors, availability.float(), source_deltas, target_coordinates)


def test_rejects_wrong_receiver_grid() -> None:
    model = _small_model(TRACE_LATTICE_GRAPH_MODE)
    generator = torch.Generator().manual_seed(3)
    neighbors = torch.randn(1, SOURCES, 4, 68, TIME, generator=generator)
    availability = torch.ones(1, SOURCES, 4, 68, dtype=torch.bool)
    source_deltas = torch.randn(1, SOURCES, 2, generator=generator) + 50.0
    target_coordinates = torch.rand(1, 2, generator=generator)
    with pytest.raises(ValueError, match="receiver dimensions"):
        model(neighbors, availability, source_deltas, target_coordinates)


def test_rejects_unknown_graph_mode() -> None:
    with pytest.raises(ValueError, match="graph_mode"):
        TraceGraphInterpolator(width=8, graph_mode="fully_connected")


def test_rejects_dilation_count_mismatch() -> None:
    with pytest.raises(ValueError, match="message_passing_rounds"):
        TraceGraphInterpolator(
            width=8,
            message_passing_rounds=3,
            temporal_dilations=(1, 2),
        )


def test_rejects_width_not_divisible_by_group_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TraceGraphInterpolator(width=12)


def _train_one_step(model: TraceGraphInterpolator) -> None:
    generator = torch.Generator().manual_seed(11)
    neighbors, availability, source_deltas, target_coordinates = _inputs(generator)
    targets = torch.randn(BATCH, 8, 68, TIME, generator=generator)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model.train()
    prediction = model(neighbors, availability, source_deltas, target_coordinates)
    loss = torch.mean(torch.square(prediction - targets))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
