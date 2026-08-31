from __future__ import annotations

import torch

from seis_interp.models.shared_offset_attention_inpainter import (
    DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE,
    SharedOffsetAttentionInpainter,
)
from seis_interp.processing.multiline_neighbor_geometry import multiline_neighbor_offsets

_OFFSETS = (
    (0, 0, 0, -1),
    (0, 0, 0, 1),
    (2, 0, 0, 0),
)


def _model() -> SharedOffsetAttentionInpainter:
    return SharedOffsetAttentionInpainter(
        _OFFSETS,
        width=8,
        neighbor_feature_width=4,
        attention_width=4,
        target_coordinate_count=4,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1,),
        coarse_shift_samples_per_relative_receiver_y_index=2,
        attention_geometry_prior_scale=0.5,
    )


def test_k274_geometry_offsets_and_coarse_shifts_are_stored_exactly() -> None:
    offsets = multiline_neighbor_offsets(2, 0, 4, 5)

    model = SharedOffsetAttentionInpainter(
        offsets,
        width=8,
        neighbor_feature_width=4,
        attention_width=4,
        temporal_dilations=(1,),
    )

    assert len(offsets) == 274
    assert model.neighbor_offsets.tolist() == [list(offset) for offset in offsets]
    assert model.coarse_sample_shifts.tolist() == [3 * offset[3] for offset in offsets]


def test_initial_prediction_is_exact_masked_distance_prior_shifted_reference() -> None:
    model = _model()
    neighbors = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        ]
    )
    availability = torch.tensor([[True, True, True]])
    target_coordinates = torch.tensor([[0.1, -0.2, 0.3, -0.4]])

    prediction = model(neighbors, availability, target_coordinates)

    aligned = torch.tensor(
        [
            [
                [3.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 20.0, 30.0, 40.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        ]
    )
    valid = aligned != 0.0
    logits = torch.tensor([-0.5, -0.5, -2.0])[None, :, None].expand(1, -1, 6)
    weights = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=1)
    expected = (aligned * weights).sum(dim=1)
    torch.testing.assert_close(prediction, expected)
    assert model.prediction_reference == DISTANCE_PRIOR_SHIFTED_NEIGHBOR_REFERENCE
    final_projection = model.head[-1]
    assert isinstance(final_projection, torch.nn.Conv1d)
    torch.testing.assert_close(final_projection.weight, torch.zeros_like(final_projection.weight))
    torch.testing.assert_close(final_projection.bias, torch.zeros_like(final_projection.bias))


def test_masked_attention_ignores_unavailable_values_and_handles_empty_rows() -> None:
    torch.manual_seed(7)
    model = _model()
    final_projection = model.head[-1]
    assert isinstance(final_projection, torch.nn.Conv1d)
    torch.nn.init.normal_(final_projection.weight)
    torch.nn.init.normal_(final_projection.bias)
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [False, False, False]])
    target_coordinates = torch.randn(2, 4)

    expected = model(neighbors, availability, target_coordinates)
    changed = neighbors.clone()
    changed[0, 1] = 1.0e6
    changed[1] = -1.0e6
    actual = model(changed, availability, target_coordinates)

    torch.testing.assert_close(actual, expected)
    aligned, aligned_availability = model._coarse_align(neighbors, availability.float())
    encoded = model._encode_neighbors(aligned)
    attention = model._masked_attention_weights(
        encoded,
        aligned_availability,
        target_coordinates,
    )
    assert bool(torch.isfinite(attention).all())
    expected_sum = aligned_availability[0].any(dim=0).to(dtype=attention.dtype)
    torch.testing.assert_close(attention[0].sum(dim=0), expected_sum)
    torch.testing.assert_close(attention[0, 1], torch.zeros(9))
    torch.testing.assert_close(attention[1], torch.zeros(3, 9))


def test_zero_padded_edges_are_excluded_and_attention_is_renormalized() -> None:
    model = _model()
    neighbors = torch.ones(1, 3, 6)
    availability = torch.ones(1, 3, dtype=torch.bool)
    target_coordinates = torch.zeros(1, 4)

    aligned, aligned_availability = model._coarse_align(neighbors, availability.float())
    encoded = model._encode_neighbors(aligned)
    attention = model._masked_attention_weights(
        encoded,
        aligned_availability,
        target_coordinates,
    )

    assert not bool(aligned_availability[0, 1, 0])
    assert float(attention[0, 1, 0]) == 0.0
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(1, 6))
    torch.testing.assert_close((attention * aligned).sum(dim=1), torch.ones(1, 6))


def test_offset_target_time_attention_and_residual_receive_gradients() -> None:
    torch.manual_seed(13)
    model = _model()
    neighbors = torch.randn(3, 3, 11)
    availability = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])
    target_coordinates = torch.randn(3, 4)
    target = torch.randn(3, 11)

    torch.square(model(neighbors, availability, target_coordinates) - target).mean().backward()

    assert model.offset_key_projection.weight.grad is not None
    assert bool(torch.isfinite(model.offset_key_projection.weight.grad).all())
    assert float(model.offset_key_projection.weight.grad.abs().sum()) > 0.0
    final_projection = model.head[-1]
    assert isinstance(final_projection, torch.nn.Conv1d)
    assert final_projection.weight.grad is not None
    assert float(final_projection.weight.grad.abs().sum()) > 0.0


def test_two_optimizer_steps_reach_every_learned_branch() -> None:
    torch.manual_seed(19)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    neighbors = torch.randn(3, 3, 11)
    availability = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])
    target_coordinates = torch.randn(3, 4)
    target = torch.randn(3, 11)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.square(model(neighbors, availability, target_coordinates) - target).mean().backward()
        optimizer.step()

    selected = {
        "shared_encoder.0.weight",
        "offset_feature_modulation.weight",
        "offset_key_projection.weight",
        "target_query_projection.weight",
        "time_query_projection.weight",
        "content_score_projection.weight",
        "stem.weight",
        "coordinate_modulations.0.weight",
        "head.2.weight",
        "head.4.weight",
    }
    parameters = dict(model.named_parameters())
    for name in selected:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_permuting_offsets_and_matching_neighbor_axis_preserves_output() -> None:
    torch.manual_seed(23)
    model = _model()
    with torch.no_grad():
        torch.nn.init.normal_(model.offset_feature_modulation.weight)
        torch.nn.init.normal_(model.offset_feature_modulation.bias)
        torch.nn.init.normal_(model.offset_key_projection.weight)
        torch.nn.init.normal_(model.content_score_projection.weight)
        torch.nn.init.normal_(model.content_score_projection.bias)
        final_projection = model.head[-1]
        assert isinstance(final_projection, torch.nn.Conv1d)
        torch.nn.init.normal_(final_projection.weight)
        torch.nn.init.normal_(final_projection.bias)
    permutation = torch.tensor([2, 0, 1])
    permuted = SharedOffsetAttentionInpainter(
        tuple(_OFFSETS[index] for index in permutation.tolist()),
        width=8,
        neighbor_feature_width=4,
        attention_width=4,
        target_coordinate_count=4,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1,),
        coarse_shift_samples_per_relative_receiver_y_index=2,
        attention_geometry_prior_scale=0.5,
    )
    for target_parameter, source_parameter in zip(
        permuted.parameters(),
        model.parameters(),
        strict=True,
    ):
        target_parameter.data.copy_(source_parameter.data)
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [True, True, True]])
    target_coordinates = torch.randn(2, 4)

    expected = model(neighbors, availability, target_coordinates)
    actual = permuted(
        neighbors[:, permutation],
        availability[:, permutation],
        target_coordinates,
    )

    torch.testing.assert_close(actual, expected)


def test_forward_supports_double_precision_and_all_unavailable_zero_reference() -> None:
    model = _model().double()
    neighbors = torch.randn(2, 3, 7, dtype=torch.float64)
    availability = torch.zeros(2, 3, dtype=torch.bool)
    target_coordinates = torch.randn(2, 4, dtype=torch.float64)

    prediction = model(neighbors, availability, target_coordinates)

    assert prediction.dtype == torch.float64
    assert bool(torch.isfinite(prediction).all())
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))
