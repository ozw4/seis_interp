from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from seis_interp.models import NeighborTraceInpainter
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
    }
    assert all(tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values())
    assert payload["amplitude_scaling"] == "per_trace_rms"
    assert payload["validation_metric_domain"] == "oracle_per_trace_unit_rms"
    assert loaded.model.neighbor_count == 3
    assert loaded.model.width == 8
    assert loaded.model.target_coordinate_count == 2
    assert loaded.model.stem_kernel_size == 5
    assert loaded.model.residual_kernel_size == 3
    assert loaded.model.temporal_dilations == (1, 3, 2)
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    assert loaded.amplitude_scaling == "per_trace_rms"
    assert loaded.validation_metric_domain == "oracle_per_trace_unit_rms"
    assert loaded.best_step == 2300
    assert loaded.best_validation_global_snr_db == pytest.approx(16.182)
    torch.testing.assert_close(loaded.model(neighbors, availability, coordinates), expected)


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
    ):
        payload["model_config"].pop(field)
    torch.save(payload, legacy_checkpoint_path)

    loaded = load_neighbor_inpainter_checkpoint(legacy_checkpoint_path)

    assert loaded.model.target_coordinate_count == 3
    assert loaded.model.stem_kernel_size == 15
    assert loaded.model.residual_kernel_size == 7
    assert loaded.model.temporal_dilations == (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1)
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
