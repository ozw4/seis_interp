from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.models.nersi import Nersi
from seis_interp.training.c3_volume_nersi_data import (
    PROFILE_AXIS_ORDER,
    PROFILE_COORDINATE_ORDER,
    C3VolumeNersiData,
)
from seis_interp.training.nersi_checkpoints import (
    AMPLITUDE_SCALING,
    FIXED_STEP_FINAL_CHECKPOINT_ROLE,
    NERSI_METHOD_VARIANT,
    TRAINING_DOMAIN,
    load_fixed_step_nersi_checkpoint,
    nersi_checkpoint_input_binding,
    save_fixed_step_nersi_checkpoint,
    validate_fixed_step_nersi_checkpoint_input_binding,
)


def _inputs_lock() -> dict[str, object]:
    return {
        "benchmark_case": {
            "case_id": "case-a",
            "sha256": "a" * 64,
        },
        "benchmark_volume": {
            "volume_id": "volume-a",
            "files": {
                "volume.json": {"sha256": "b" * 64},
                "volume_index.parquet": {"sha256": "c" * 64},
            },
        },
    }


def _model() -> Nersi:
    torch.manual_seed(13)
    return Nersi(
        fourier_components=2,
        frequency_base=1.5,
        encoder_width=8,
        latent_channels=2,
        decoder_channels=(3, 2, 2),
        profile_shape=(8, 8),
        kernel_size=1,
    )


def _data(*, amplitude_scale: float = 2.5) -> C3VolumeNersiData:
    mask = np.array(
        [
            [True, False, True, False, False, False, False, False],
            [False, True, False, False, False, False, False, False],
        ],
        dtype=np.bool_,
    )
    return C3VolumeNersiData(
        normalized_coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        normalized_profiles=np.zeros((2, 1, 8, 8), dtype=np.float32),
        observed_trace_mask=mask,
        training_profile_indices=np.array([0, 1], dtype=np.int64),
        amplitude_scale=amplitude_scale,
        spatial_shape=(2, 1, 1, 8),
        profile_shape=(8, 8),
    )


def _save(path: Path) -> tuple[Nersi, C3VolumeNersiData]:
    model, data = _model(), _data()
    save_fixed_step_nersi_checkpoint(
        path,
        model,
        data,
        global_step=7,
        final_batch_loss=0.125,
        input_binding=nersi_checkpoint_input_binding(_inputs_lock()),
        model_initialization_seed=101,
        sampling_seed=201,
    )
    return model, data


def test_cpu_round_trip_restores_function_constructor_and_preprocessing(tmp_path: Path) -> None:
    path = tmp_path / "final.pt"
    model, data = _save(path)
    coordinates = torch.tensor(data.normalized_coordinates)
    expected = model(coordinates).detach()

    loaded = load_fixed_step_nersi_checkpoint(path)

    torch.testing.assert_close(loaded.model(coordinates), expected, rtol=0, atol=0)
    assert loaded.model.constructor_config() == model.constructor_config()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], tensor, rtol=0, atol=0)
    assert loaded.coordinate_order == PROFILE_COORDINATE_ORDER
    assert loaded.profile_axis_order == PROFILE_AXIS_ORDER
    assert loaded.coordinate_bounds == data.coordinate_bounds == ((0, 1), (0, 0), (0, 0))
    assert loaded.spatial_shape == data.spatial_shape
    assert loaded.profile_shape == data.profile_shape
    assert loaded.amplitude_scaling == AMPLITUDE_SCALING
    assert loaded.amplitude_scale == data.amplitude_scale
    assert (loaded.global_step, loaded.final_batch_loss) == (7, 0.125)
    assert loaded.input_binding == nersi_checkpoint_input_binding(_inputs_lock())
    validate_fixed_step_nersi_checkpoint_input_binding(loaded, _inputs_lock(), data)


def test_payload_is_cpu_snapshot_without_resume_or_target_information(tmp_path: Path) -> None:
    path = tmp_path / "final.pt"
    model, _ = _save(path)
    saved_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {
        "model_type",
        "checkpoint_role",
        "method_variant",
        "training_domain",
        "input_binding",
        "model_config",
        "model_state_dict",
        "preprocessing",
        "training",
    }
    assert payload["model_type"] == "nersi"
    assert payload["checkpoint_role"] == FIXED_STEP_FINAL_CHECKPOINT_ROLE
    assert payload["method_variant"] == NERSI_METHOD_VARIANT
    assert payload["training_domain"] == TRAINING_DOMAIN
    assert payload["input_binding"] == nersi_checkpoint_input_binding(_inputs_lock())
    assert payload["preprocessing"] == {
        "coordinate_order": list(PROFILE_COORDINATE_ORDER),
        "coordinate_normalization": "fixed_analysis_domain_index_bounds_to_unit_interval",
        "coordinate_bounds": [[0, 1], [0, 0], [0, 0]],
        "profile_axis_order": list(PROFILE_AXIS_ORDER),
        "spatial_shape": [2, 1, 1, 8],
        "profile_shape": [8, 8],
        "amplitude_scaling": AMPLITUDE_SCALING,
        "amplitude_scale": 2.5,
    }
    assert payload["training"] == {
        "global_step": 7,
        "final_batch_loss": 0.125,
        "model_initialization_seed": 101,
        "sampling_seed": 201,
    }
    assert "optimizer" not in payload
    assert "rng" not in payload
    assert "evaluation" not in payload
    assert "best" not in repr(payload).lower()
    assert "target" not in repr(payload).lower()
    for name, tensor in payload["model_state_dict"].items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, saved_state[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_can_load_directly_to_cuda(tmp_path: Path) -> None:
    path = tmp_path / "final.pt"
    _save(path)

    loaded = load_fixed_step_nersi_checkpoint(path, device="cuda:0")

    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cuda"}


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_type", "other"),
        ("checkpoint_role", "best"),
        ("method_variant", "legacy_nersi"),
        ("training_domain", "evaluation_target"),
    ],
)
def test_rejects_wrong_checkpoint_identity(tmp_path: Path, field: str, replacement: object) -> None:
    path = tmp_path / "final.pt"
    _save(path)
    payload = torch.load(path, weights_only=True)
    payload[field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=field):
        load_fixed_step_nersi_checkpoint(path)


@pytest.mark.parametrize("key", ["model_initialization_seed", "sampling_seed"])
@pytest.mark.parametrize("value", [None, True, -1, 1.5])
def test_checkpoint_requires_explicit_nonnegative_seed_integers(tmp_path, key, value):
    path = tmp_path / "final.pt"
    _save(path)
    payload = torch.load(path, weights_only=True)
    payload["training"][key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=key):
        load_fixed_step_nersi_checkpoint(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_config", "model_config"),
        ("shape", "profile_shape"),
        ("coordinate_order", "coordinate_order"),
        ("coordinate_bounds", "coordinate_bounds"),
        ("profile_order", "profile_axis_order"),
        ("scale", "amplitude_scale"),
        ("state", "model_state_dict"),
    ],
)
def test_rejects_incomplete_or_inconsistent_payload(
    tmp_path: Path, change: str, message: str
) -> None:
    path = tmp_path / "final.pt"
    _save(path)
    payload = torch.load(path, weights_only=True)
    if change == "missing_config":
        del payload["model_config"]
    elif change == "shape":
        payload["preprocessing"]["profile_shape"] = [16, 8]
    elif change == "coordinate_order":
        payload["preprocessing"]["coordinate_order"] = ["wrong"]
    elif change == "coordinate_bounds":
        payload["preprocessing"]["coordinate_bounds"] = [[0, 2], [0, 0], [0, 0]]
    elif change == "profile_order":
        payload["preprocessing"]["profile_axis_order"] = ["receiver_y", "time"]
    elif change == "scale":
        payload["preprocessing"]["amplitude_scale"] = 0.0
    else:
        del payload["model_state_dict"][next(iter(payload["model_state_dict"]))]
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_fixed_step_nersi_checkpoint(path)


def test_rejects_wrong_case_volume_or_preprocessing_when_pairing_current_inputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "final.pt"
    _save(path)
    loaded = load_fixed_step_nersi_checkpoint(path)
    wrong_inputs = copy.deepcopy(_inputs_lock())
    wrong_inputs["benchmark_volume"]["volume_id"] = "volume-b"  # type: ignore[index]

    with pytest.raises(ValueError, match="case/volume input binding"):
        validate_fixed_step_nersi_checkpoint_input_binding(loaded, wrong_inputs, _data())
    with pytest.raises(ValueError, match="preprocessing"):
        validate_fixed_step_nersi_checkpoint_input_binding(
            loaded,
            _inputs_lock(),
            _data(amplitude_scale=3.0),
        )


def test_rejects_malformed_checkpoint_input_binding(tmp_path: Path) -> None:
    path = tmp_path / "final.pt"
    _save(path)
    payload = torch.load(path, weights_only=True)
    payload["input_binding"]["input_hashes"]["benchmark_case"] = "not-a-sha256"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="SHA-256"):
        load_fixed_step_nersi_checkpoint(path)


@pytest.mark.parametrize(
    ("step", "loss", "message"),
    [
        (0, 0.1, "global_step"),
        (True, 0.1, "global_step"),
        (1, -1.0, "final_batch_loss"),
        (1, float("nan"), "final_batch_loss"),
    ],
)
def test_save_rejects_invalid_final_training_metadata(
    tmp_path: Path, step: object, loss: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        save_fixed_step_nersi_checkpoint(
            tmp_path / "bad.pt",
            _model(),
            _data(),
            global_step=step,
            final_batch_loss=loss,
            input_binding=nersi_checkpoint_input_binding(_inputs_lock()),
            model_initialization_seed=101,
            sampling_seed=201,
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True])
def test_save_rejects_invalid_amplitude_scale(tmp_path: Path, scale: object) -> None:
    with pytest.raises(ValueError, match="amplitude_scale"):
        save_fixed_step_nersi_checkpoint(
            tmp_path / "bad.pt",
            _model(),
            _data(amplitude_scale=scale),  # type: ignore[arg-type]
            global_step=1,
            final_batch_loss=0.0,
            input_binding=nersi_checkpoint_input_binding(_inputs_lock()),
            model_initialization_seed=101,
            sampling_seed=201,
        )
