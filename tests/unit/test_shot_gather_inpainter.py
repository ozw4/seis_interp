from __future__ import annotations

import pytest
import torch
from torch import nn

from seis_interp.models.shot_gather_inpainter import (
    RECEIVER_X_COUNT,
    RECEIVER_Y_COUNT,
    SHOT_GATHER_INPUT_FEATURE_NAMES,
    FactorizedGatherResidualBlock,
    ShotGatherInpainter,
    inverse_distance_reference,
)


def _inputs(
    *,
    batch_size: int = 2,
    source_count: int = 3,
    time_count: int = 7,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbors = torch.randn(
        batch_size,
        source_count,
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
        time_count,
        dtype=dtype,
    )
    availability = torch.ones(
        batch_size,
        source_count,
        RECEIVER_X_COUNT,
        RECEIVER_Y_COUNT,
        dtype=torch.bool,
    )
    source_deltas = (
        torch.tensor(
            [[float(index + 1), float(index % 2)] for index in range(source_count)],
            dtype=dtype,
        )
        .expand(batch_size, -1, -1)
        .clone()
    )
    target_coordinates = torch.randn(batch_size, 2, dtype=dtype)
    return neighbors, availability, source_deltas, target_coordinates


def test_inverse_distance_reference_is_receiver_wise_and_masks_missing_sources() -> None:
    neighbors = torch.empty(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 3)
    neighbors[:, 0].fill_(2.0)
    neighbors[:, 1].fill_(6.0)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    availability[:, 1, 0, 0] = False
    availability[:, :, 0, 1] = False
    source_deltas = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])

    reference = inverse_distance_reference(neighbors, availability, source_deltas)

    torch.testing.assert_close(reference[:, 1:, 1:], torch.full_like(reference[:, 1:, 1:], 3.0))
    torch.testing.assert_close(reference[0, 0, 0], torch.full((3,), 2.0))
    torch.testing.assert_close(reference[0, 0, 1], torch.zeros(3))


def test_inverse_distance_reference_is_finite_for_low_precision_close_sources() -> None:
    neighbors = torch.tensor([1.0, 3.0], dtype=torch.float16).view(1, 2, 1, 1, 1)
    neighbors = neighbors.expand(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 1)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor([[[1.0e-5, 0.0], [2.0e-5, 0.0]]], dtype=torch.float16)

    reference = inverse_distance_reference(neighbors, availability, source_deltas)

    assert reference.dtype == torch.float16
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(
        reference.float(),
        torch.full_like(reference.float(), 5.0 / 3.0),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_zero_initialized_model_returns_exact_inverse_distance_reference() -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1, 2))
    neighbors, availability, source_deltas, target_coordinates = _inputs()
    availability[:, 1, 2:4, 7:11] = False
    expected = inverse_distance_reference(neighbors, availability, source_deltas)

    output = model(neighbors, availability, source_deltas, target_coordinates)

    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    assert isinstance(model.head[-1], nn.Conv3d)
    assert torch.count_nonzero(model.head[-1].weight) == 0
    assert torch.count_nonzero(model.head[-1].bias) == 0
    assert model.spatial_y_dilations == (1, 1)
    assert all(block.spatial.dilation == (1, 1, 1) for block in model.blocks)
    assert all(block.spatial.padding == (1, 1, 0) for block in model.blocks)


def test_default_spatial_y_dilations_match_explicit_ones_exactly() -> None:
    torch.manual_seed(37)
    default_model = ShotGatherInpainter(width=8, temporal_dilations=(1, 2))
    torch.manual_seed(37)
    explicit_model = ShotGatherInpainter(
        width=8,
        temporal_dilations=(1, 2),
        spatial_y_dilations=(1, 1),
    )

    assert default_model.spatial_y_dilations == (1, 1)
    for name, expected in default_model.state_dict().items():
        torch.testing.assert_close(
            explicit_model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_stem_feature_contract_includes_signed_moments_and_receiver_coordinates() -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,)).double()
    neighbors = torch.tensor(
        [[1.0, 3.0], [5.0, 7.0]],
        dtype=torch.float64,
    ).view(1, 2, 1, 1, 2)
    neighbors = neighbors.expand(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 2)
    availability = torch.ones(1, 2, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, dtype=torch.bool)
    source_deltas = torch.tensor(
        [[[-1.0, 0.0], [0.6, 0.8]]],
        dtype=torch.float64,
    )
    target_coordinates = torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    captured_features: list[torch.Tensor] = []

    def capture_stem_input(_module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
        captured_features.append(inputs[0].detach().clone())

    handle = model.stem.register_forward_pre_hook(capture_stem_input)
    try:
        output = model(neighbors, availability, source_deltas, target_coordinates)
    finally:
        handle.remove()

    expected_feature_names = (
        "inverse_distance_reference",
        "weighted_absolute_deviation",
        "source_direction_x_waveform_moment",
        "source_direction_y_waveform_moment",
        "availability_fraction",
        "weighted_source_direction_x",
        "weighted_source_direction_y",
        "target_coordinate_x",
        "target_coordinate_y",
        "receiver_coordinate_x",
        "receiver_coordinate_y",
    )
    assert expected_feature_names == SHOT_GATHER_INPUT_FEATURE_NAMES
    assert model.input_feature_names == expected_feature_names
    assert model.input_channels == 11
    assert model.stem.in_channels == 11
    assert len(captured_features) == 1
    features = captured_features[0]
    assert features.shape == (1, 11, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 2)
    assert features.dtype == neighbors.dtype
    assert features.device == neighbors.device
    channel = {name: index for index, name in enumerate(model.input_feature_names)}
    reference = torch.tensor([3.0, 5.0], dtype=torch.float64)
    torch.testing.assert_close(output, reference.view(1, 1, 1, 2).expand_as(output))
    torch.testing.assert_close(
        features[0, channel["inverse_distance_reference"]],
        reference.view(1, 1, 2).expand(RECEIVER_X_COUNT, RECEIVER_Y_COUNT, -1),
    )
    for name, value in (
        ("weighted_absolute_deviation", 2.0),
        ("source_direction_x_waveform_moment", 1.6),
        ("source_direction_y_waveform_moment", 0.8),
        ("availability_fraction", 1.0),
        ("weighted_source_direction_x", -0.2),
        ("weighted_source_direction_y", 0.4),
        ("target_coordinate_x", 0.25),
        ("target_coordinate_y", -0.5),
    ):
        torch.testing.assert_close(
            features[0, channel[name]],
            torch.full_like(features[0, channel[name]], value),
        )
    expected_receiver_x = torch.linspace(
        -1.0,
        1.0,
        RECEIVER_X_COUNT,
        dtype=torch.float64,
    )[:, None].expand(-1, RECEIVER_Y_COUNT)
    expected_receiver_y = torch.linspace(
        -1.0,
        1.0,
        RECEIVER_Y_COUNT,
        dtype=torch.float64,
    )[None, :].expand(RECEIVER_X_COUNT, -1)
    torch.testing.assert_close(
        features[0, channel["receiver_coordinate_x"], :, :, 0],
        expected_receiver_x,
    )
    torch.testing.assert_close(
        features[0, channel["receiver_coordinate_y"], :, :, 0],
        expected_receiver_y,
    )
    torch.testing.assert_close(
        features[0, channel["receiver_coordinate_x"], :, :, 1],
        expected_receiver_x,
    )
    torch.testing.assert_close(
        features[0, channel["receiver_coordinate_y"], :, :, 1],
        expected_receiver_y,
    )


def test_custom_architecture_preserves_shape_and_factorized_block_contract() -> None:
    model = ShotGatherInpainter(
        width=16,
        temporal_dilations=(1, 3, 2),
        spatial_y_dilations=(1, 2, 4),
        stem_kernel_size=5,
        residual_kernel_size=3,
    )
    inputs = _inputs(batch_size=1, source_count=2, time_count=11)

    output = model(*inputs)

    assert output.shape == (1, RECEIVER_X_COUNT, RECEIVER_Y_COUNT, 11)
    assert model.width == 16
    assert model.input_channels == len(SHOT_GATHER_INPUT_FEATURE_NAMES) == 11
    assert model.stem.in_channels == 11
    assert model.temporal_dilations == (1, 3, 2)
    assert model.spatial_y_dilations == (1, 2, 4)
    assert model.stem.kernel_size == (1, 1, 5)
    assert [block.temporal.dilation for block in model.blocks] == [
        (1, 1, 1),
        (1, 1, 3),
        (1, 1, 2),
    ]
    assert all(block.temporal.groups == 16 for block in model.blocks)
    assert all(block.spatial.kernel_size == (3, 3, 1) for block in model.blocks)
    assert all(block.spatial.groups == 16 for block in model.blocks)
    assert [block.spatial_y_dilation for block in model.blocks] == [1, 2, 4]
    assert [block.spatial.dilation for block in model.blocks] == [
        (1, 1, 1),
        (1, 2, 1),
        (1, 4, 1),
    ]
    assert [block.spatial.padding for block in model.blocks] == [
        (1, 1, 0),
        (1, 2, 0),
        (1, 4, 0),
    ]


@pytest.mark.parametrize(
    ("spatial_y_dilations", "match"),
    [
        ((1,), "length must equal"),
        ((1, 0), r"spatial_y_dilations\[1\].*positive integer"),
        ("1,2", "must be an iterable"),
    ],
)
def test_model_rejects_invalid_spatial_y_dilations(
    spatial_y_dilations: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ShotGatherInpainter(
            width=8,
            temporal_dilations=(1, 2),
            spatial_y_dilations=spatial_y_dilations,
        )


def test_synthetic_forward_backward_reaches_reference_and_residual_parameters() -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,)).double()
    neighbors, availability, source_deltas, target_coordinates = _inputs(
        batch_size=1,
        source_count=2,
        time_count=5,
        dtype=torch.float64,
    )
    neighbors.requires_grad_()

    model(neighbors, availability, source_deltas, target_coordinates).square().mean().backward()

    assert neighbors.grad is not None
    assert torch.isfinite(neighbors.grad).all()
    assert torch.count_nonzero(neighbors.grad) > 0
    final_projection = model.head[-1]
    assert final_projection.weight.grad is not None
    assert final_projection.bias.grad is not None
    assert torch.isfinite(final_projection.weight.grad).all()
    assert torch.count_nonzero(final_projection.weight.grad) > 0


def test_backward_is_finite_when_each_receiver_has_only_one_available_source() -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,)).double()
    with torch.no_grad():
        model.head[-1].weight.fill_(0.1)
    neighbors, availability, source_deltas, target_coordinates = _inputs(
        batch_size=1,
        source_count=2,
        time_count=5,
        dtype=torch.float64,
    )
    availability[:, 1] = False
    neighbors.requires_grad_()

    model(neighbors, availability, source_deltas, target_coordinates).square().mean().backward()

    assert neighbors.grad is not None
    assert torch.isfinite(neighbors.grad).all()
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()


@pytest.mark.parametrize(
    ("input_index", "replacement", "error", "match"),
    [
        (
            0,
            torch.randn(2, 3, 7, RECEIVER_Y_COUNT, 7),
            ValueError,
            "receiver dimensions",
        ),
        (
            1,
            torch.ones(2, 3, RECEIVER_X_COUNT, RECEIVER_Y_COUNT),
            TypeError,
            "torch.bool",
        ),
        (2, torch.randn(2, 3, 2, dtype=torch.float64), TypeError, "share the neighbors dtype"),
        (3, torch.randn(2, 3), ValueError, "target_coordinates must have shape"),
    ],
)
def test_model_rejects_invalid_shapes_and_dtypes(
    input_index: int,
    replacement: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    inputs = list(_inputs())
    inputs[input_index] = replacement

    with pytest.raises(error, match=match):
        model(*inputs)


def test_model_rejects_mixed_devices_before_computation() -> None:
    model = ShotGatherInpainter(width=8, temporal_dilations=(1,))
    neighbors, availability, source_deltas, _target_coordinates = _inputs()
    target_coordinates = torch.empty(2, 2, device="meta")

    with pytest.raises(ValueError, match="share a device"):
        model(neighbors, availability, source_deltas, target_coordinates)


def test_available_source_requires_nonzero_delta_but_missing_source_does_not() -> None:
    neighbors, availability, source_deltas, _target_coordinates = _inputs(source_count=2)
    source_deltas[:, 0] = 0.0

    with pytest.raises(ValueError, match="non-zero source delta"):
        inverse_distance_reference(neighbors, availability, source_deltas)

    availability[:, 0] = False
    reference = inverse_distance_reference(neighbors, availability, source_deltas)
    torch.testing.assert_close(reference, neighbors[:, 1])


def test_factorized_block_validates_receiver_shape() -> None:
    block = FactorizedGatherResidualBlock(width=8, temporal_dilation=2)

    with pytest.raises(ValueError, match="features must have shape"):
        block(torch.randn(1, 8, RECEIVER_X_COUNT, RECEIVER_Y_COUNT - 1, 5))
