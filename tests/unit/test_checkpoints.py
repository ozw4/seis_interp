from __future__ import annotations

from pathlib import Path

import pytest
import torch

from seis_interp.data.trace_schema import MODEL_COORDINATE_ORDER
from seis_interp.models.siren import Siren
from seis_interp.processing.normalization import NormalizationParameters
from seis_interp.processing.training_coordinates import (
    CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES,
    CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES,
    CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
    model_coordinate_parameters,
)
from seis_interp.training.checkpoints import (
    load_fixed_step_siren_checkpoint,
    load_siren_checkpoint,
    save_fixed_step_siren_checkpoint,
    save_siren_checkpoint,
)


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
    assert loaded.model_coordinates is None
    assert loaded.time_coordinate_scale == 1.0
    assert loaded.amplitude_scaling == "train_global_rms"
    assert loaded.validation_metric_domain == "train_global_rms"
    assert (
        loaded.epoch,
        loaded.global_step,
        loaded.validation_median_trace_snr_db,
        loaded.validation_global_snr_db,
    ) == (3, 21, 4.5, 11.25)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "model_type",
        "model_config",
        "model_state_dict",
        "normalization",
        "amplitude_scaling",
        "validation_metric_domain",
        "training",
    }


@pytest.mark.parametrize(
    ("schedule", "skip"), [(None, None), ("exponential", None), ("exponential", "dense")]
)
def test_fixed_final_checkpoint_round_trip(
    tmp_path: Path, schedule: str | None, skip: str | None
) -> None:
    torch.manual_seed(39)
    model = Siren(
        hidden_width=7,
        hidden_layers=3,
        omega_0=5.0,
        hidden_omega=40.0,
        layer_omega_schedule=schedule,
        skip_connections=skip,
    )
    coordinates = torch.randn(7, 6)
    expected = model(coordinates).detach()
    normalization = _normalization()
    parameters = model_coordinate_parameters("cmp_offset_azimuth", normalization)
    path = tmp_path / "final.pt"
    save_fixed_step_siren_checkpoint(
        path, model, normalization, parameters, global_step=23, final_batch_loss=0.125
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    loaded = load_fixed_step_siren_checkpoint(path, device="cpu")

    torch.testing.assert_close(loaded.model(coordinates), expected, rtol=0, atol=0)
    assert loaded.normalization == normalization
    assert loaded.model_coordinates == parameters
    assert loaded.global_step == 23
    assert loaded.final_batch_loss == 0.125
    assert payload["model_type"] == "siren"
    assert payload["checkpoint_role"] == "fixed_step_final"
    assert payload["method_variant"] == "per_volume_internal_learning_fixed_steps"
    assert payload["amplitude_scaling"] == "train_global_rms"
    assert payload["training_domain"] == "benchmark_observed_samples"
    assert payload["training"] == {"global_step": 23, "final_batch_loss": 0.125}
    assert set(payload) == {
        "model_type",
        "checkpoint_role",
        "method_variant",
        "model_config",
        "model_state_dict",
        "normalization",
        "model_coordinates",
        "amplitude_scaling",
        "training_domain",
        "training",
    }
    expected_config = {
        "input_features": 6,
        "hidden_width": 7,
        "hidden_layers": 3,
        "output_features": 1,
        "omega_0": 5.0,
        "hidden_omega": 40.0,
        "layer_omega_schedule": schedule,
        "skip_connections": skip,
    }
    assert payload["model_config"] == expected_config
    for name, tensor in payload["model_state_dict"].items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(loaded.model.state_dict()[name], tensor, rtol=0, atol=0)
    assert loaded.model.layer_omega_schedule == schedule
    assert loaded.model.skip_connections == skip
    assert loaded.model.layer_omegas == model.layer_omegas


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("model_type", "other", "model_type"),
        ("checkpoint_role", "best", "checkpoint_role"),
        ("method_variant", "other", "method_variant"),
        ("normalization", {}, "normalization parameters"),
        ("model_coordinates", {}, "model coordinates"),
        ("training", {}, "training.*required"),
        ("training", {"global_step": True, "final_batch_loss": 0.1}, "global_step"),
        ("training", {"global_step": 1, "final_batch_loss": float("nan")}, "final_batch_loss"),
    ],
)
def test_fixed_final_checkpoint_rejects_corrupt_metadata(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    path = tmp_path / "final.pt"
    save_fixed_step_siren_checkpoint(
        path,
        Siren(hidden_width=4, hidden_layers=1),
        _normalization(),
        model_coordinate_parameters("cmp_offset_azimuth", _normalization()),
        global_step=1,
        final_batch_loss=0.5,
    )
    payload = torch.load(path, weights_only=True)
    payload[field] = replacement
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_fixed_step_siren_checkpoint(path)


@pytest.mark.parametrize("schedule", [None, "exponential"])
def test_fixed_final_checkpoint_requires_complete_constructor_and_strict_weights(
    tmp_path: Path,
    schedule: str | None,
) -> None:
    path = tmp_path / "final.pt"
    save_fixed_step_siren_checkpoint(
        path,
        Siren(
            hidden_width=4, hidden_layers=2, layer_omega_schedule=schedule, skip_connections="dense"
        ),
        _normalization(),
        model_coordinate_parameters("cmp_offset_azimuth", _normalization()),
        global_step=1,
        final_batch_loss=0.5,
    )
    original = torch.load(path, weights_only=True)
    for key in original:
        incomplete = dict(original)
        del incomplete[key]
        torch.save(incomplete, path)
        with pytest.raises(ValueError, match="missing required fields"):
            load_fixed_step_siren_checkpoint(path)
    for key in original["model_config"]:
        incomplete = dict(original)
        incomplete["model_config"] = dict(original["model_config"])
        del incomplete["model_config"][key]
        torch.save(incomplete, path)
        with pytest.raises(ValueError, match="constructor fields"):
            load_fixed_step_siren_checkpoint(path)
    original["model_state_dict"].pop(next(iter(original["model_state_dict"])))
    torch.save(original, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fixed_step_siren_checkpoint(path)


def test_fixed_final_checkpoint_checks_coordinate_width_on_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "final.pt"
    model = Siren(hidden_width=4, hidden_layers=1)
    parameters = model_coordinate_parameters("cmp_cartesian_half_offset", _normalization())
    with pytest.raises(ValueError, match="coordinate width"):
        save_fixed_step_siren_checkpoint(
            path, model, _normalization(), parameters, global_step=1, final_batch_loss=0.5
        )
    assert not path.exists()
    save_fixed_step_siren_checkpoint(
        path,
        model,
        _normalization(),
        model_coordinate_parameters("cmp_offset_azimuth", _normalization()),
        global_step=1,
        final_batch_loss=0.5,
    )
    payload = torch.load(path, weights_only=True)
    payload["model_coordinates"] = parameters.to_dict()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="coordinate width"):
        load_fixed_step_siren_checkpoint(path)


@pytest.mark.parametrize(("step", "loss"), [(0, 0.0), (True, 0.0), (1, -1.0), (1, float("inf"))])
def test_fixed_final_checkpoint_rejects_invalid_training_values_on_save(
    tmp_path: Path, step: int, loss: float
) -> None:
    path = tmp_path / "final.pt"
    with pytest.raises(ValueError, match="global_step|final_batch_loss"):
        save_fixed_step_siren_checkpoint(
            path,
            Siren(hidden_width=4, hidden_layers=1),
            _normalization(),
            model_coordinate_parameters("cmp_offset_azimuth", _normalization()),
            global_step=step,
            final_batch_loss=loss,
        )
    assert not path.exists()


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
    assert "layer_omega_schedule" not in payload["model_config"]
    assert "skip_connections" not in payload["model_config"]
    payload["model_config"].pop("hidden_omega")
    payload.pop("amplitude_scaling")
    payload.pop("validation_metric_domain")
    legacy_checkpoint_path = tmp_path / "legacy.pt"
    torch.save(payload, legacy_checkpoint_path)

    loaded = load_siren_checkpoint(legacy_checkpoint_path)

    assert loaded.model.hidden_omega == 1.0
    assert loaded.model.layer_omega_schedule is None
    assert loaded.model.skip_connections is None
    assert loaded.amplitude_scaling == "train_global_rms"
    assert loaded.validation_metric_domain == "train_global_rms"
    torch.testing.assert_close(loaded.model(coordinates), expected)


def test_checkpoint_round_trip_restores_exponential_layer_omega_schedule(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = Siren(
        hidden_width=7,
        hidden_layers=4,
        omega_0=5.0,
        hidden_omega=50.0,
        layer_omega_schedule="exponential",
    )
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
    loaded = load_siren_checkpoint(checkpoint_path)
    assert payload["model_config"]["layer_omega_schedule"] == "exponential"
    assert loaded.model.layer_omega_schedule == "exponential"
    assert loaded.model.layer_omegas == pytest.approx(
        tuple(5.0 * 10.0 ** (index / 3.0) for index in range(4))
    )
    torch.testing.assert_close(loaded.model(coordinates), expected)


def test_checkpoint_round_trip_restores_dense_skip_architecture(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    model = Siren(
        hidden_width=7,
        hidden_layers=4,
        omega_0=5.0,
        hidden_omega=50.0,
        layer_omega_schedule="exponential",
        skip_connections="dense",
    )
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
    loaded = load_siren_checkpoint(checkpoint_path)
    assert payload["model_config"]["skip_connections"] == "dense"
    assert loaded.model.skip_connections == "dense"
    assert loaded.model.sine_layer_input_features == (6, 7, 14, 21)
    assert loaded.model.final_input_features == 28
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


@pytest.mark.parametrize(
    ("coordinate_features", "input_features"),
    [
        (CMP_CARTESIAN_HALF_OFFSET_COORDINATE_FEATURES, 5),
        (CMP_CARTESIAN_HALF_OFFSET_RADIUS_COORDINATE_FEATURES, 6),
    ],
)
def test_checkpoint_records_cartesian_coordinate_mode_and_scales(
    tmp_path: Path,
    coordinate_features: str,
    input_features: int,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    coordinates = model_coordinate_parameters(
        coordinate_features,
        _normalization(),
    )

    save_siren_checkpoint(
        checkpoint_path,
        Siren(input_features=input_features, hidden_width=7, hidden_layers=1),
        _normalization(),
        model_coordinates=coordinates,
        epoch=2,
        global_step=6,
        validation_median_trace_snr_db=None,
        validation_global_snr_db=8.5,
    )

    loaded = load_siren_checkpoint(checkpoint_path)
    assert loaded.model.input_features == input_features
    assert loaded.model_coordinates == coordinates
    assert loaded.model_coordinates.half_offset_scale_m == 2.0


def test_checkpoint_round_trip_exposes_nondefault_time_coordinate_scale(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    coordinates = model_coordinate_parameters(
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        _normalization(),
        time_coordinate_scale=4.0,
    )

    save_siren_checkpoint(
        checkpoint_path,
        Siren(hidden_width=7, hidden_layers=1),
        _normalization(),
        model_coordinates=coordinates,
        epoch=2,
        global_step=6,
        validation_median_trace_snr_db=3.0,
        validation_global_snr_db=8.5,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_siren_checkpoint(checkpoint_path)
    assert payload["model_coordinates"]["time_coordinate_scale"] == 4.0
    assert loaded.model_coordinates == coordinates
    assert loaded.time_coordinate_scale == 4.0


@pytest.mark.parametrize("invalid_scale", [1.0e-50, 1.0e40, 10**400])
def test_checkpoint_rejects_unusable_or_overflowing_time_coordinate_scale(
    tmp_path: Path,
    invalid_scale: object,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    coordinates = model_coordinate_parameters(
        CMP_OFFSET_AZIMUTH_COORDINATE_FEATURES,
        _normalization(),
        time_coordinate_scale=4.0,
    )
    save_siren_checkpoint(
        checkpoint_path,
        Siren(hidden_width=7, hidden_layers=1),
        _normalization(),
        model_coordinates=coordinates,
        epoch=2,
        global_step=6,
        validation_median_trace_snr_db=3.0,
        validation_global_snr_db=8.5,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["model_coordinates"]["time_coordinate_scale"] = invalid_scale
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="positive finite number representable as float32"):
        load_siren_checkpoint(checkpoint_path)


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
