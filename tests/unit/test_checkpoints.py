from __future__ import annotations

from pathlib import Path

import pytest
import torch

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.training.checkpoints import load_siren_checkpoint, save_siren_checkpoint


def _normalization() -> NormalizationParameters:
    return NormalizationParameters(
        coordinate_order=MODEL_COORDINATE_ORDER,
        coordinate_min=(-1.0, -2.0, -3.0, -4.0, -1.0, -1.0),
        coordinate_max=(1.0, 2.0, 3.0, 4.0, 1.0, 1.0),
        amplitude_rms=2.5,
    )


def test_checkpoint_round_trip_restores_function_config_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = Siren(
        input_features=6,
        hidden_width=7,
        hidden_layers=2,
        omega_0=13.0,
        hidden_omega=17.0,
    )
    coordinates = torch.randn(5, 6)
    expected = model(coordinates).detach()
    checkpoint_path = tmp_path / "nested" / "best.pt"

    save_siren_checkpoint(
        checkpoint_path,
        model,
        _normalization(),
        epoch=3,
        global_step=21,
        validation_median_trace_snr_db=4.5,
        validation_global_snr_db=11.25,
    )
    loaded = load_siren_checkpoint(checkpoint_path)

    torch.testing.assert_close(loaded.model(coordinates), expected)
    assert loaded.model.input_features == 6
    assert loaded.model.hidden_width == 7
    assert loaded.model.hidden_layers == 2
    assert loaded.model.output_features == 1
    assert loaded.model.omega_0 == 13.0
    assert loaded.model.hidden_omega == 17.0
    assert loaded.normalization == _normalization()
    assert loaded.amplitude_scaling == "train_global_rms"
    assert loaded.validation_metric_domain == "train_global_rms"
    assert (
        loaded.epoch,
        loaded.global_step,
        loaded.validation_median_trace_snr_db,
        loaded.validation_global_snr_db,
    ) == (3, 21, 4.5, 11.25)


def test_checkpoint_persists_hidden_omega_and_loads_legacy_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = Siren(hidden_width=7, hidden_layers=2)
    coordinates = torch.randn(5, model.input_features)
    expected = model(coordinates).detach()
    save_siren_checkpoint(
        checkpoint_path,
        model,
        _normalization(),
        epoch=1,
        global_step=2,
        validation_median_trace_snr_db=3.0,
        validation_global_snr_db=4.0,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["model_config"]["hidden_omega"] == 1.0
    payload["model_config"].pop("hidden_omega")
    payload.pop("amplitude_scaling")
    payload.pop("validation_metric_domain")
    legacy_checkpoint_path = tmp_path / "legacy.pt"
    torch.save(payload, legacy_checkpoint_path)

    loaded = load_siren_checkpoint(legacy_checkpoint_path)

    assert loaded.model.hidden_omega == 1.0
    assert loaded.amplitude_scaling == "train_global_rms"
    assert loaded.validation_metric_domain == "train_global_rms"
    torch.testing.assert_close(loaded.model(coordinates), expected)


def test_checkpoint_allows_global_only_validation_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = Siren(hidden_width=7, hidden_layers=1)

    save_siren_checkpoint(
        checkpoint_path,
        model,
        _normalization(),
        epoch=2,
        global_step=6,
        validation_median_trace_snr_db=None,
        validation_global_snr_db=8.5,
    )

    loaded = load_siren_checkpoint(checkpoint_path)

    assert loaded.validation_median_trace_snr_db is None
    assert loaded.validation_global_snr_db == 8.5


def test_checkpoint_records_per_trace_rms_target_scaling(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"

    save_siren_checkpoint(
        checkpoint_path,
        Siren(hidden_width=7, hidden_layers=1),
        _normalization(),
        amplitude_scaling="per_trace_rms",
        epoch=2,
        global_step=6,
        validation_median_trace_snr_db=None,
        validation_global_snr_db=8.5,
    )

    loaded = load_siren_checkpoint(checkpoint_path)
    assert loaded.amplitude_scaling == "per_trace_rms"
    assert loaded.validation_metric_domain == "oracle_per_trace_unit_rms"


def test_checkpoint_rejects_a_metric_domain_that_mislabels_target_scaling(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    save_siren_checkpoint(
        checkpoint_path,
        Siren(hidden_width=7, hidden_layers=1),
        _normalization(),
        amplitude_scaling="per_trace_rms",
        epoch=1,
        global_step=2,
        validation_median_trace_snr_db=None,
        validation_global_snr_db=3.0,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["validation_metric_domain"] = "train_global_rms"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="does not match amplitude_scaling"):
        load_siren_checkpoint(checkpoint_path)
