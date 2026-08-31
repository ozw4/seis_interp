from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from seis_interp.models import NeighborTraceInpainter, SharedOffsetAttentionInpainter
from seis_interp.models.neighbor_trace_inpainter import (
    SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE,
)
from seis_interp.training.neighbor_inpainter_checkpoints import (
    load_neighbor_inpainter_checkpoint,
    save_neighbor_inpainter_checkpoint,
)


def test_neighbor_inpainter_checkpoint_round_trip_preserves_model_and_metadata(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = NeighborTraceInpainter(
        neighbor_count=3,
        width=8,
        target_coordinate_count=2,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1, 3, 2),
        coordinate_conditioning="film",
        neighbor_gating="target_coordinate_masked_softmax",
        neighbor_alignment_kernel_size=3,
        prediction_reference="masked_aligned_neighbor_mean",
    )
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [True, True, False]])
    coordinates = torch.randn(2, 2)
    expected = model(neighbors, availability, coordinates).detach()
    checkpoint_path = tmp_path / "nested" / "best.pt"

    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=2300,
        best_validation_global_snr_db=16.182,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_neighbor_inpainter_checkpoint(checkpoint_path, device="cpu")

    assert payload["model_type"] == "neighbor_trace_inpainter"
    assert payload["model_config"] == {
        "neighbor_count": 3,
        "width": 8,
        "target_coordinate_count": 2,
        "stem_kernel_size": 5,
        "residual_kernel_size": 3,
        "temporal_dilations": [1, 3, 2],
        "coordinate_conditioning": "film",
        "neighbor_gating": "target_coordinate_masked_softmax",
        "neighbor_alignment_kernel_size": 3,
        "prediction_reference": "masked_aligned_neighbor_mean",
    }
    assert "neighbor_offsets" not in payload["model_state_dict"]
    assert "coarse_sample_shifts" not in payload["model_state_dict"]
    assert all(tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values())
    assert payload["amplitude_scaling"] == "per_trace_rms"
    assert payload["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert loaded.model.neighbor_count == 3
    assert loaded.model.width == 8
    assert loaded.model.target_coordinate_count == 2
    assert loaded.model.stem_kernel_size == 5
    assert loaded.model.residual_kernel_size == 3
    assert loaded.model.temporal_dilations == (1, 3, 2)
    assert loaded.model.coordinate_conditioning == "film"
    assert loaded.model.neighbor_gating == "target_coordinate_masked_softmax"
    assert loaded.model.neighbor_gate_projection is not None
    assert loaded.model.neighbor_alignment_kernel_size == 3
    assert loaded.model.neighbor_alignment is not None
    assert loaded.model.prediction_reference == "masked_aligned_neighbor_mean"
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    assert loaded.amplitude_scaling == "per_trace_rms"
    assert loaded.validation_metric_domain == "oracle_per_trace_unit_rms"
    assert loaded.best_step == 2300
    assert loaded.best_validation_global_snr_db == pytest.approx(16.182)
    torch.testing.assert_close(loaded.model(neighbors, availability, coordinates), expected)


def test_bracketing_reference_checkpoint_round_trip_uses_last_channel(
    tmp_path: Path,
) -> None:
    model = NeighborTraceInpainter(
        neighbor_count=3,
        width=8,
        temporal_dilations=(1,),
        prediction_reference="same_line_exact_receiver_linear_bracketing",
    )
    neighbors = torch.randn(2, 3, 9)
    availability = torch.ones(2, 3, dtype=torch.bool)
    coordinates = torch.randn(2, 3)
    expected = model(neighbors, availability, coordinates)
    checkpoint_path = tmp_path / "bracketing.pt"

    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=10,
        best_validation_global_snr_db=5.5,
    )
    loaded = load_neighbor_inpainter_checkpoint(checkpoint_path)

    assert loaded.model.prediction_reference == ("same_line_exact_receiver_linear_bracketing")
    assert loaded.model.local_neighbor_count == 2
    torch.testing.assert_close(
        loaded.model(neighbors, availability, coordinates),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_bracketing_channels_checkpoint_round_trip_uses_weighted_last_two_channels(
    tmp_path: Path,
) -> None:
    model = NeighborTraceInpainter(
        neighbor_count=4,
        width=8,
        temporal_dilations=(1,),
        prediction_reference=(SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE),
    )
    neighbors = torch.randn(2, 4, 9)
    availability = torch.tensor(((1.0, 1.0, 0.25, 0.75), (1.0, 0.0, 1.0, 0.0)))
    coordinates = torch.randn(2, 3)
    expected = model(neighbors, availability, coordinates)
    checkpoint_path = tmp_path / "bracketing-channels.pt"

    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=10,
        best_validation_global_snr_db=5.5,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_neighbor_inpainter_checkpoint(checkpoint_path)

    assert payload["model_config"]["prediction_reference"] == (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    assert loaded.model.prediction_reference == (
        SAME_LINE_EXACT_RECEIVER_LINEAR_BRACKETING_CHANNELS_REFERENCE
    )
    assert loaded.model.neighbor_count == 4
    assert loaded.model.reference_neighbor_count == 2
    assert loaded.model.local_neighbor_count == 2
    torch.testing.assert_close(
        loaded.model(neighbors, availability, coordinates),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_neighbor_trace_coarse_alignment_checkpoint_round_trip_preserves_exact_offsets(
    tmp_path: Path,
) -> None:
    offsets = ((0, 0, 0, -1), (0, 0, 0, 1), (1, 0, 0, 0))
    model = NeighborTraceInpainter(
        neighbor_count=3,
        width=8,
        target_coordinate_count=4,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1,),
        coordinate_conditioning="film",
        neighbor_gating="target_coordinate_masked_softmax",
        neighbor_alignment_kernel_size=3,
        coarse_shift_samples_per_relative_receiver_y_index=2,
        neighbor_offsets=offsets,
    )
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [True, True, False]])
    coordinates = torch.randn(2, 4)
    expected = model(neighbors, availability, coordinates).detach()
    checkpoint_path = tmp_path / "coarse" / "best.pt"

    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=50,
        best_validation_global_snr_db=14.5,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_neighbor_inpainter_checkpoint(checkpoint_path)

    assert payload["model_config"]["neighbor_offsets"] == [list(offset) for offset in offsets]
    assert payload["model_config"]["coarse_shift_samples_per_relative_receiver_y_index"] == 2
    torch.testing.assert_close(
        payload["model_state_dict"]["neighbor_offsets"],
        torch.tensor(offsets),
    )
    torch.testing.assert_close(
        payload["model_state_dict"]["coarse_sample_shifts"],
        torch.tensor([-2, 2, 0]),
    )
    assert isinstance(loaded.model, NeighborTraceInpainter)
    assert loaded.model.neighbor_offsets is not None
    assert loaded.model.coarse_sample_shifts is not None
    assert loaded.model.neighbor_offsets.tolist() == [list(offset) for offset in offsets]
    assert loaded.model.coarse_sample_shifts.tolist() == [-2, 2, 0]
    assert loaded.model.coarse_shift_samples_per_relative_receiver_y_index == 2
    torch.testing.assert_close(loaded.model(neighbors, availability, coordinates), expected)


def test_neighbor_trace_checkpoint_rejects_corrupted_derived_coarse_shift_buffer(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "corrupt-coarse-shift.pt"
    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        NeighborTraceInpainter(
            neighbor_count=2,
            width=8,
            temporal_dilations=(1,),
            coarse_shift_samples_per_relative_receiver_y_index=3,
            neighbor_offsets=((0, 0, 0, -1), (0, 0, 0, 1)),
        ),
        best_step=1,
        best_validation_global_snr_db=1.0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["coarse_sample_shifts"] = torch.tensor([-2, 2])
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="derived coarse alignment buffer 'coarse_sample_shifts'"):
        load_neighbor_inpainter_checkpoint(checkpoint_path)


def test_shared_offset_attention_checkpoint_round_trip_preserves_exact_offsets(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    offsets = ((0, 0, 0, -1), (0, 0, 0, 1), (1, 0, 0, 0))
    model = SharedOffsetAttentionInpainter(
        offsets,
        width=8,
        neighbor_feature_width=4,
        attention_width=6,
        target_coordinate_count=4,
        stem_kernel_size=5,
        residual_kernel_size=3,
        temporal_dilations=(1, 2),
        coarse_shift_samples_per_relative_receiver_y_index=3,
        attention_geometry_prior_scale=0.75,
    )
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [True, True, False]])
    coordinates = torch.randn(2, 4)
    expected = model(neighbors, availability, coordinates).detach()
    checkpoint_path = tmp_path / "shared" / "best.pt"

    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=100,
        best_validation_global_snr_db=25.5,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_neighbor_inpainter_checkpoint(checkpoint_path)

    assert payload["model_type"] == "shared_offset_attention_inpainter"
    assert payload["model_config"] == {
        "neighbor_offsets": [list(offset) for offset in offsets],
        "width": 8,
        "neighbor_feature_width": 4,
        "attention_width": 6,
        "target_coordinate_count": 4,
        "stem_kernel_size": 5,
        "residual_kernel_size": 3,
        "temporal_dilations": [1, 2],
        "coarse_shift_samples_per_relative_receiver_y_index": 3,
        "attention_geometry_prior_scale": 0.75,
    }
    torch.testing.assert_close(
        payload["model_state_dict"]["neighbor_offsets"],
        torch.tensor(offsets),
    )
    assert isinstance(loaded.model, SharedOffsetAttentionInpainter)
    assert tuple(tuple(row) for row in loaded.model.neighbor_offsets.tolist()) == offsets
    assert loaded.model.coarse_shift_samples_per_relative_receiver_y_index == 3
    assert loaded.model.attention_geometry_prior_scale == pytest.approx(0.75)
    assert loaded.best_step == 100
    assert loaded.best_validation_global_snr_db == pytest.approx(25.5)
    torch.testing.assert_close(loaded.model(neighbors, availability, coordinates), expected)


@pytest.mark.parametrize(
    "buffer_name",
    [
        "neighbor_offsets",
        "normalized_neighbor_offsets",
        "attention_geometry_prior",
        "coarse_sample_shifts",
    ],
)
def test_shared_checkpoint_rejects_derived_buffers_inconsistent_with_config(
    tmp_path: Path,
    buffer_name: str,
) -> None:
    model = SharedOffsetAttentionInpainter(
        ((0, 0, 0, -1), (0, 0, 0, 1)),
        width=8,
        neighbor_feature_width=4,
        attention_width=4,
        temporal_dilations=(1,),
    )
    checkpoint_path = tmp_path / f"corrupt-{buffer_name}.pt"
    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=1,
        best_validation_global_snr_db=1.0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["model_state_dict"][buffer_name] = payload["model_state_dict"][buffer_name].clone()
    payload["model_state_dict"][buffer_name].view(-1)[0] += 1
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=f"derived buffer {buffer_name!r}"):
        load_neighbor_inpainter_checkpoint(checkpoint_path)


def test_neighbor_inpainter_checkpoint_loads_study017_legacy_model_config(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    model = NeighborTraceInpainter(neighbor_count=3, width=8)
    neighbors = torch.randn(2, 3, 9)
    availability = torch.tensor([[True, False, True], [True, True, False]])
    coordinates = torch.randn(2, 3)
    expected = model(neighbors, availability, coordinates).detach()
    checkpoint_path = tmp_path / "best.pt"
    legacy_checkpoint_path = tmp_path / "legacy.pt"
    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        model,
        best_step=2500,
        best_validation_global_snr_db=18.111870025656728,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for field in (
        "target_coordinate_count",
        "stem_kernel_size",
        "residual_kernel_size",
        "temporal_dilations",
        "coordinate_conditioning",
        "neighbor_gating",
        "neighbor_alignment_kernel_size",
        "prediction_reference",
    ):
        payload["model_config"].pop(field)
    torch.save(payload, legacy_checkpoint_path)

    loaded = load_neighbor_inpainter_checkpoint(legacy_checkpoint_path)

    assert loaded.model.target_coordinate_count == 3
    assert loaded.model.stem_kernel_size == 15
    assert loaded.model.residual_kernel_size == 7
    assert loaded.model.temporal_dilations == (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
    assert loaded.model.coordinate_conditioning == "stem"
    assert loaded.model.neighbor_gating == "none"
    assert loaded.model.neighbor_gate_projection is None
    assert loaded.model.neighbor_alignment_kernel_size == 1
    assert loaded.model.neighbor_alignment is None
    assert loaded.model.prediction_reference == "none"
    torch.testing.assert_close(loaded.model(neighbors, availability, coordinates), expected)


def test_neighbor_inpainter_checkpoint_loads_weights_strictly(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        NeighborTraceInpainter(neighbor_count=1, width=8),
        best_step=1,
        best_validation_global_snr_db=2.0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["unexpected.weight"] = torch.ones(1)
    torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_neighbor_inpainter_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("amplitude_scaling", "train_global_rms", "amplitude_scaling"),
        ("validation_metric_domain", "train_global_rms", "validation_metric_domain"),
        ("model_type", "siren", "model_type"),
    ],
)
def test_neighbor_inpainter_checkpoint_rejects_incompatible_contract_metadata(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    save_neighbor_inpainter_checkpoint(
        checkpoint_path,
        NeighborTraceInpainter(neighbor_count=1, width=8),
        best_step=1,
        best_validation_global_snr_db=2.0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        load_neighbor_inpainter_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    ("best_step", "validation_snr", "message"),
    [
        (0, 1.0, "best_step"),
        (True, 1.0, "best_step"),
        (1, math.nan, "finite"),
        (1, math.inf, "finite"),
        (1, True, "finite"),
    ],
)
def test_neighbor_inpainter_checkpoint_rejects_invalid_selection_metadata(
    tmp_path: Path,
    best_step: object,
    validation_snr: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        save_neighbor_inpainter_checkpoint(
            tmp_path / "best.pt",
            NeighborTraceInpainter(neighbor_count=1, width=8),
            best_step=best_step,  # type: ignore[arg-type]
            best_validation_global_snr_db=validation_snr,  # type: ignore[arg-type]
        )
