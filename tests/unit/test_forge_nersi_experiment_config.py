import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seis_interp.forge_nersi_experiment_config import (
    NERSI_TRACE_DEFAULTS,
    REJECTED_KEYS,
    budget_comparison,
    config_differences,
    ema_initial_weight_fraction,
    fourier_max_angular_frequencies,
    load_experiment_config,
    load_experiment_inputs,
    resolve_run_config,
    validate_against_preparation,
    validate_device_available,
)
from seis_interp.processing.nersi_trace_coordinates import midpoint_offset_transform

WORKSPACE = Path(__file__).resolve().parents[2] / "studies/study_forge_nersi_temp"
# The user edits config.yaml; the traps and diffs below use the pinned reference defaults.
CONFIG_TEXT = (WORKSPACE / "config.default.yaml").read_text()
# Spec table of rejected keys with a distinctive fragment of each reason.
REJECTED_REASON_FRAGMENTS = {
    "training.optimizer": "coupled L2",
    "training.learning_rate_schedule": "constant learning rate",
    "training.mixed_precision": "no AMP",
    "loss": "masked_trace_relative_mse",
    "model.input_features": "fixed by NersiTrace",
    "model.upsample_scales": "fixed by NersiTrace",
    "model.activation": "fixed by NersiTrace",
    "model.output_activation": "fixed by NersiTrace",
    "model.time_sample_count": "derived from the preparation contract",
    "checkpoint_rule": "derived from training.ema_decay",
    "model.coordinate_transform": "derived from model.coordinate_frame",
    "model.moveout_samples_per_radius": "use model.moveout_velocity_m_per_s",
    "model.head_samples_per_radius": "use model.head_slowness_ms_per_m",
    "model.residual_slowness_samples_per_radius": "use model.residual_slowness_ms_per_m",
    "model.output_window": "derived from model.output_window_mask",
    "model.spatial_shape": "derived from the preparation geometry",
    "model.sample_interval_s": "derived from the preparation contract",
    "model.axis_frequency_limits": "not forwarded by NersiTrace",
    "model.decoder_convolution": "not forwarded by NersiTrace",
    **{
        key: "choose another preparation in inputs.yaml"
        for key in (
            "method_id",
            "geometry_mode",
            "alignment",
            "amplitude_scaling",
            "trace_scale_hash",
            "mask_hash",
            "mask_id",
            "mask_seed",
            "global_rms",
            "preparation",
        )
    },
}
SHAPE = (2, 3, 2, 3)
CONTRACT = {"time_sample_count": 9, "sample_interval_us": 2000}
SEALED = {
    "method_id": "nersi_real",
    "geometry_mode": "real",
    "model_seed": 7301,
    "loss": "masked_trace_relative_mse",
    "checkpoint_rule": "fixed_final_ema",
    "device": "cuda:1",
    "source_config": "studies/x/nersi.yaml",
    "source_config_sha256": "0" * 64,
    "changes_from_source": ["tuned"],
    "model": {"encoder_width": 999, "time_sample_count": 9},
    "training": {"learning_rate": 0.001},
    "prediction": {"batch_size": 256},
    "alignment": {"velocity_m_per_s": 500.0, "max_delay_samples": 6},
    "amplitude_scaling": "trace_oracle",
    "trace_scale_hash": "a" * 64,
    "mask_hash": "b" * 64,
    "mask_id": "m1_random80_mvp_v1",
    "mask_seed": 510101,
    "global_rms": 0.01,
}


def _config():
    return yaml.safe_load(CONFIG_TEXT)


def _write(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(config if isinstance(config, str) else yaml.safe_dump(config))
    return path


def _errors(tmp_path, config):
    with pytest.raises(ValueError) as caught:
        load_experiment_config(_write(tmp_path, config))
    return str(caught.value)


def _geometry():
    rng = np.random.default_rng(3)
    local = rng.uniform(0, 1, size=(12, 4))
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    geometry = pd.DataFrame(local * (np.array(SHAPE) - 1), columns=columns)
    geometry[["source_x_m", "source_y_m"]] = local[:, :2] @ [[40, 5], [3, 60]]
    geometry[["receiver_x_m", "receiver_y_m"]] = local[:, 2:] @ [[35, -4], [2, 55]] + [30, 10]
    return geometry


def test_committed_workspace_files_are_valid_without_preparation_or_gpu():
    # config.yaml is user-editable: only its validity is checked, never its values.
    load_experiment_config(WORKSPACE / "config.yaml")
    assert load_experiment_config(WORKSPACE / "config.default.yaml")["name"] == "baseline"
    inputs = load_experiment_inputs(WORKSPACE / "inputs.yaml")
    assert set(inputs) == {"preparation", "preparation_manifest_sha256"}
    assert not Path(inputs["preparation"]).is_absolute()


@pytest.mark.parametrize(
    "path",
    [
        "model.kernel_size",
        "model.head_slowness_ms_per_m",
        "model.output_window_mask",
        "training.weight_decay_after",
        "training.ema_decay",
        "prediction.batch_size",
    ],
)
def test_missing_key_is_reported(tmp_path, path):
    config = _config()
    section, key = path.split(".")
    del config[section][key]
    assert f"missing required key {path}" in _errors(tmp_path, config)


def test_unknown_key_suggests_the_allowed_key(tmp_path):
    config = _config()
    config["training"]["lr"] = config["training"].pop("learning_rate")
    message = _errors(tmp_path, config)
    assert "unknown key training.lr; did you mean training.learning_rate?" in message
    assert "allowed: training.learning_rate" in message


def test_rejected_key_table_matches_the_spec():
    assert set(REJECTED_KEYS) == set(REJECTED_REASON_FRAGMENTS)


@pytest.mark.parametrize("path", sorted(REJECTED_REASON_FRAGMENTS))
def test_fixed_derived_and_unsupported_keys_are_rejected(tmp_path, path):
    config = _config()
    *parents, key = path.split(".")
    (config[parents[0]] if parents else config)[key] = "anything"
    message = _errors(tmp_path, config)
    assert f"{path}: {REJECTED_KEYS[path]}" in message
    assert REJECTED_REASON_FRAGMENTS[path] in REJECTED_KEYS[path]


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("learning_rate: 0.001", "learning_rate: 1e-3", "write 0.001 or 1.0e-3"),
        ("learning_rate: 0.001", "learning_rate: 1.0e3", "write 1000 or 1.0e+3"),
        ("learning_rate: 0.001", "learning_rate: nan", "must be a finite number, got str 'nan'"),
        ("max_steps: 20000", "max_steps: 2e4", "training.max_steps: got string '2e4'"),
        ("max_steps: 20000", 'max_steps: "20000"', "quoted numbers stay strings, write 20000"),
        ("max_steps: 20000", "max_steps: 25e-1", "must be an integer, got str '25e-1'"),
        ("model_seed: 7301", "model_seed: 07301", "model_seed: '07301' is not a plain decimal"),
        ("max_steps: 20000", "max_steps: 20:00", "PyYAML reads it as 1200"),
        ("encoder_width: 150", "encoder_width: 0x96", "PyYAML reads it as 150"),
        ("max_steps: 20000", "max_steps: 20000.0", "training.max_steps: must be an integer"),
        ("batch_size: 64", "batch_size: true", "training.batch_size: must be an integer"),
        ("encoder_width: 150", "encoder_width: 0", "model.encoder_width: must be >= 1"),
        ("kernel_size: 5", "kernel_size: 4", "model.kernel_size: must be an odd"),
        ("decoder_channels: [16, 8, 4]", "decoder_channels: [16, 8]", "exactly 3 positive"),
        ("frequency_base: [1.05, 1.05, 1.1, 1.1]", "frequency_base: 1.0", "finite number > 1"),
        ("frequency_base: [1.05, 1.05, 1.1, 1.1]", "frequency_base: [1.1, 1.1, 1.1]", "3 items"),
        ("ema_decay: 0.9998", "ema_decay: 0", "use null to disable EMA"),
        ("ema_decay: 0.9998", "ema_decay: 1.0", "training.ema_decay: ema_decay must be"),
        ("report_interval: 100", "report_interval: 0", "training.report_interval: must be >= 1"),
        ("max_steps: 20000", "max_steps: 0", "training.max_steps: must be >= 1"),
        ("batch_size: 256", "batch_size: 0", "prediction.batch_size: must be >= 1"),
        ("learning_rate: 0.001", "learning_rate: -0.001", "training.learning_rate: must be > 0"),
        ("weight_decay: 5.0e-05", "weight_decay: -1.0", "training.weight_decay: must be >= 0"),
        ("polar_offset: true", 'polar_offset: "yes"', "must be true or false, got str 'yes'"),
        ("name: baseline", 'name: "Bad/Name"', "name: must match"),
        ("name: baseline", "name: latest", "'latest' is reserved"),
        ("device: cuda:0", "device: cuda", "give an explicit index"),
        ("device: cuda:0", "device: gpu:0", "device: must be cpu or cuda:<index>"),
        ("name: baseline", 'name: "baseline\\n"', "(lowercase snake_case), got 'baseline\\n'"),
        ("device: cuda:0", 'device: "cuda:0\\n"', "cuda:<index>, got 'cuda:0\\n'"),
    ],
)
def test_yaml_traps_are_reported_by_key_path(tmp_path, old, new, expected):
    assert old in CONFIG_TEXT
    assert expected in _errors(tmp_path, CONFIG_TEXT.replace(old, new, 1))


def test_integer_hint_suggests_only_the_integer_spelling(tmp_path):
    message = _errors(tmp_path, CONFIG_TEXT.replace("max_steps: 20000", "max_steps: 2e4", 1))
    assert "write 20000" in message and "2.0e+4" not in message


def test_duplicate_keys_are_reported_with_both_lines(tmp_path):
    text = CONFIG_TEXT.replace(
        "  latent_channels: 4", "  encoder_width: 300\n  latent_channels: 4", 1
    )
    first = text.splitlines().index(next(ln for ln in text.splitlines() if "width: 150" in ln))
    second = text.splitlines().index("  encoder_width: 300")
    message = _errors(tmp_path, text)
    assert f"model.encoder_width: duplicate key (lines {first + 1} and {second + 1})" in message


def test_yaml_syntax_errors_name_the_file(tmp_path):
    text = CONFIG_TEXT.replace("  kernel_size: 5", "   kernel_size: 5", 1)
    message = _errors(tmp_path, text)
    assert message.startswith(f"{tmp_path / 'config.yaml'}: YAML syntax error")
    inputs = tmp_path / "inputs.yaml"
    inputs.write_text("preparation: [unclosed\n")
    with pytest.raises(ValueError, match="inputs.yaml: YAML syntax error"):
        load_experiment_inputs(inputs)


def test_several_errors_are_reported_together(tmp_path):
    text = CONFIG_TEXT.replace("kernel_size: 5", "kernel_size: 4").replace(
        "learning_rate: 0.001", "learning_rate: 1e-3"
    )
    message = _errors(tmp_path, text)
    assert message.startswith(f"{tmp_path / 'config.yaml'}: invalid NeRSI experiment config")
    assert "model.kernel_size" in message and "training.learning_rate" in message


def test_cross_field_rules(tmp_path):
    config = _config()
    config["model"]["coordinate_frame"] = "source_receiver"
    assert "polar_offset needs the midpoint_offset frame" in _errors(tmp_path, config)
    config = _config()
    config["model"].update(polar_offset=False, moveout_velocity_m_per_s=700.0)
    assert "moveout requires polar_offset: true" in _errors(tmp_path, config)


def test_device_override_leaves_inputs_unchanged(tmp_path):
    path = _write(tmp_path, CONFIG_TEXT)
    requested = load_experiment_config(path, device_override="cuda:1")
    assert requested["device"] == "cuda:1"
    assert yaml.safe_load(path.read_text())["device"] == "cuda:0"
    with pytest.raises(ValueError, match="--device: bare 'cuda'"):
        load_experiment_config(path, device_override="cuda")
    sealed = deepcopy(SEALED)
    resolve_run_config(requested, SEALED, CONTRACT, _geometry(), SHAPE)
    assert sealed == SEALED
    assert requested == load_experiment_config(path, device_override="cuda:1")


def test_inputs_require_relative_path_and_hash(tmp_path):
    path = tmp_path / "inputs.yaml"
    for inputs, message in (
        ({"preparation": "runs/x"}, "exactly"),
        ({"preparation": "../x", "preparation_manifest_sha256": "a" * 64}, "without '..'"),
        ({"preparation": "/abs", "preparation_manifest_sha256": "a" * 64}, "relative"),
        ({"preparation": "runs/x", "preparation_manifest_sha256": "A" * 64}, "lowercase hex"),
        (
            {"preparation": "runs/x", "preparation_manifest_sha256": "a" * 64 + "\n"},
            "lowercase hex",
        ),
    ):
        path.write_text(yaml.safe_dump(inputs))
        with pytest.raises(ValueError, match=message):
            load_experiment_inputs(path)
    path.write_text(
        f"preparation: runs/x\npreparation: runs/y\npreparation_manifest_sha256: {'a' * 64}\n"
    )
    with pytest.raises(ValueError, match=r"preparation: duplicate key \(lines 1 and 2\)"):
        load_experiment_inputs(path)


def test_resolution_injects_fixed_and_derived_values():
    requested = _config()
    geometry = _geometry()
    resolved = resolve_run_config(requested, SEALED, CONTRACT, geometry, SHAPE)
    assert resolved["loss"] == "masked_trace_relative_mse"
    assert resolved["model"]["time_sample_count"] == CONTRACT["time_sample_count"]
    assert resolved["model"]["input_features"] == 4
    assert resolved["model"]["upsample_scales"] == [2, 2, 2]
    assert resolved["training"]["optimizer"] == "adam"
    assert resolved["checkpoint_rule"] == "fixed_final_ema"
    np.testing.assert_array_equal(
        resolved["model"]["coordinate_transform"],
        midpoint_offset_transform(geometry, SHAPE, polar=True),
    )
    assert resolved["model"]["moveout_samples_per_radius"] == 0.0
    for key in ("alignment", "amplitude_scaling", "trace_scale_hash", "mask_hash", "global_rms"):
        assert resolved[key] == SEALED[key]
    for key in ("source_config", "source_config_sha256", "changes_from_source"):
        assert key not in resolved
    assert resolved["model"]["encoder_width"] == requested["model"]["encoder_width"]

    requested["training"]["ema_decay"] = None
    requested["model"].update(coordinate_frame="source_receiver", polar_offset=False)
    resolved = resolve_run_config(requested, SEALED, CONTRACT, geometry, SHAPE)
    assert resolved["checkpoint_rule"] == "fixed_final"
    assert resolved["model"]["coordinate_transform"] is None

    requested["model"].update(coordinate_frame="midpoint_offset", polar_offset=False)
    resolved = resolve_run_config(requested, SEALED, CONTRACT, geometry, SHAPE)
    np.testing.assert_array_equal(
        resolved["model"]["coordinate_transform"],
        midpoint_offset_transform(geometry, SHAPE, polar=False),
    )

    requested["model"].update(polar_offset=True, moveout_velocity_m_per_s=700.0)
    resolved = resolve_run_config(requested, SEALED, CONTRACT, geometry, SHAPE)
    source = geometry[["source_x_m", "source_y_m"]].to_numpy()
    receiver = geometry[["receiver_x_m", "receiver_y_m"]].to_numpy()
    radius = np.linalg.norm((receiver - source) / 2, axis=1).max()
    assert resolved["model"]["moveout_samples_per_radius"] == pytest.approx(
        2 * radius / (700.0 * 0.002), rel=1e-12
    )


def test_resolution_rejects_another_method():
    with pytest.raises(ValueError, match="nersi_real"):
        resolve_run_config(
            _config(), {**SEALED, "method_id": "regsi_real"}, CONTRACT, _geometry(), SHAPE
        )


def test_budget_comparison_flags_band_and_equalities():
    run_config = {"training": {"max_steps": 20000, "learning_rate": 0.001, "batch_size": 64}}
    budget = {
        "max_steps": 20000,
        "learning_rate": 0.001,
        "loss_traces_per_update": 64,
        "parameter_count_target": 1000,
        "parameter_count_tolerance": 0.05,
    }
    assert budget_comparison(run_config, 1000, None) is None
    inside = budget_comparison(run_config, 1020, budget)
    assert inside["parameter_ratio"] == pytest.approx(0.02)
    assert inside["budget_matched"] is True
    # The band edges follow the same float ratio test the matrix enforces.
    low, high = inside["parameter_band"]
    assert 950 <= low <= 951 and 1049 <= high <= 1050
    for count, within in ((low, True), (high, True), (low - 1, False), (high + 1, False)):
        flags = budget_comparison(run_config, count, budget)
        assert flags["parameters_within_tolerance"] is within
        assert flags["budget_matched"] is within
    run_config["training"]["batch_size"] = 32
    changed = budget_comparison(run_config, 1000, budget)
    assert changed["batch_equal"] is False
    assert changed["budget_matched"] is False
    older = {k: v for k, v in budget.items() if k != "loss_traces_per_update"}
    with pytest.raises(ValueError, match="lacks loss_traces_per_update"):
        budget_comparison(run_config, 1000, {**older, "nersi_batch_size": 64})


def test_ema_fraction_and_fourier_maxima():
    assert ema_initial_weight_fraction({"ema_decay": 0.9998, "max_steps": 20000}) == pytest.approx(
        0.9998**19999
    )
    assert ema_initial_weight_fraction({"ema_decay": None, "max_steps": 20000}) is None
    maxima = fourier_max_angular_frequencies(_config()["model"])
    np.testing.assert_allclose(maxima, [math.pi * b**16 for b in (1.05, 1.05, 1.1, 1.1)])
    assert (
        fourier_max_angular_frequencies({"frequency_base": 2.0, "fourier_components": 2})
        == [4 * math.pi] * 4
    )


def test_batch_and_device_are_checked_against_the_preparation(monkeypatch):
    with pytest.raises(ValueError, match="64 exceeds the 7 observed traces"):
        validate_against_preparation({"training": {"batch_size": 64}}, 7)
    validate_against_preparation({"training": {"batch_size": 7}}, 7)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    validate_device_available("cpu")
    validate_device_available("cuda:1")
    with pytest.raises(ValueError, match="must be cpu or cuda:<index>"):
        validate_device_available("cuda:1\n")
    with pytest.raises(ValueError, match=r"device: cuda:2 not available \(2 CUDA devices\)"):
        validate_device_available("cuda:2")


def test_config_differences_list_changed_keys_and_default_moveout():
    resolved = resolve_run_config(_config(), SEALED, CONTRACT, _geometry(), SHAPE)
    differences = config_differences(resolved, SEALED)
    assert "model.encoder_width: 999 -> 150" in differences
    assert any(item.startswith("model.polar_offset: False -> True") for item in differences)
    assert any(item.startswith("model.coordinate_transform: null -> 4x5") for item in differences)
    assert not any(item.startswith("model.moveout_samples_per_radius") for item in differences)
    assert not any(item.startswith(("device", "prediction.batch_size")) for item in differences)
    assert config_differences(resolved, resolved) == []


SLANT = {
    "head_slowness_ms_per_m": [0.3, 2.0, 3.1],
    "static_shift_samples": 6.0,
    "residual_slowness_ms_per_m": 0.2,
    "output_band_hz": [4, 25],
    "output_window_mask": True,
}


def _slant_config():
    config = _config()
    config["model"].update(deepcopy(SLANT))
    config["training"].update(weight_decay=0.0, weight_decay_after={"step": 2000, "value": 2.0e-4})
    return config


def _maximum_half_offset(geometry):
    source = geometry[["source_x_m", "source_y_m"]].to_numpy()
    receiver = geometry[["receiver_x_m", "receiver_y_m"]].to_numpy()
    return np.linalg.norm((receiver - source) / 2, axis=1).max()


def test_committed_defaults_turn_every_decoder_prior_off():
    model, training = _config()["model"], _config()["training"]
    assert model["head_slowness_ms_per_m"] is None
    assert model["static_shift_samples"] == 0 and model["residual_slowness_ms_per_m"] == 0
    assert model["output_band_hz"] is None and model["output_window_mask"] is False
    assert training["weight_decay_after"] is None


def test_decoder_prior_config_is_valid(tmp_path):
    load_experiment_config(_write(tmp_path, _slant_config()))


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"model.head_slowness_ms_per_m": []}, "nonempty list of slownesses"),
        ({"model.head_slowness_ms_per_m": 2.0}, "must be null or a nonempty list"),
        ({"model.head_slowness_ms_per_m": [0.3, 0.0]}, "finite number > 0 (ms/m)"),
        ({"model.head_slowness_ms_per_m": [0.3, "1e-1"]}, "write 0.1 or 1.0e-1"),
        ({"model.head_slowness_ms_per_m": [0.3, 0.3]}, "must be distinct"),
        ({"model.static_shift_samples": -1.0}, "model.static_shift_samples: must be >= 0"),
        ({"model.static_shift_samples": None}, "model.static_shift_samples: must be a finite"),
        (
            {"model.residual_slowness_ms_per_m": True},
            "residual_slowness_ms_per_m: must be a finite",
        ),
        ({"model.output_band_hz": [25, 4]}, "needs finite 2 < lo < hi"),
        ({"model.output_band_hz": [2, 25]}, "needs finite 2 < lo < hi"),
        ({"model.output_band_hz": [4]}, "must be null or [lo, hi]"),
        ({"model.output_window_mask": "yes"}, "model.output_window_mask: must be true or false"),
        ({"training.weight_decay_after": {"step": 2000}}, "exactly step and value"),
        ({"training.weight_decay_after": 2000}, "exactly step and value"),
        (
            {"training.weight_decay_after": {"step": 0, "value": 1.0e-4}},
            "weight_decay_after: step must be >= 1",
        ),
        (
            {"training.weight_decay_after": {"step": 2000, "value": "2e-4"}},
            "weight_decay_after: value got string '2e-4'",
        ),
        (
            {"training.weight_decay_after": {"step": 20000, "value": 1.0e-4}},
            "step must be < training.max_steps (20000), got 20000",
        ),
        (
            {"model.polar_offset": False, "model.output_window_mask": False},
            "slowness heads require coordinate_frame midpoint_offset and polar_offset: true",
        ),
        (
            {"model.moveout_velocity_m_per_s": 700.0},
            "exclusive with model.moveout_velocity_m_per_s",
        ),
        (
            {"model.head_slowness_ms_per_m": None},
            "model.static_shift_samples: requires model.head_slowness_ms_per_m",
        ),
        (
            {"model.head_slowness_ms_per_m": None, "model.static_shift_samples": 0.0},
            "model.residual_slowness_ms_per_m: requires model.head_slowness_ms_per_m",
        ),
        (
            {"model.polar_offset": False},
            "model.output_window_mask: the window requires polar_offset: true",
        ),
    ],
)
def test_decoder_prior_errors_are_reported_by_key_path(tmp_path, changes, expected):
    config = _slant_config()
    for path, value in changes.items():
        section, key = path.split(".")
        config[section][key] = value
    assert expected in _errors(tmp_path, config)


def test_decoder_priors_resolve_from_the_preparation_geometry():
    geometry = _geometry()
    resolved = resolve_run_config(_slant_config(), SEALED, CONTRACT, geometry, SHAPE)
    model = resolved["model"]
    radius = _maximum_half_offset(geometry)
    # p_align = 1 / 500 m/s = 2 ms/m; dt = 2 ms; m_k = (p_align - p_k) * 2 R_max / dt.
    expected = [(2.0e-3 - p * 1e-3) * 2 * radius / 0.002 for p in (0.3, 2.0, 3.1)]
    np.testing.assert_allclose(model["head_samples_per_radius"], expected, rtol=1e-12)
    assert model["head_samples_per_radius"][1] == 0.0
    assert model["static_shift_samples"] == 6.0
    assert model["residual_slowness_samples_per_radius"] == pytest.approx(
        0.2e-3 * 2 * radius / 0.002, rel=1e-12
    )
    assert model["spatial_shape"] == list(SHAPE)
    assert model["output_band_hz"] == [4, 25] and model["sample_interval_s"] == 0.002
    assert model["output_window"] == {
        "max_delay_samples": 6,
        "window_samples": 9,
        "delay_samples_per_radius": pytest.approx(2 * radius / (500.0 * 0.002), rel=1e-12),
    }
    assert model["moveout_samples_per_radius"] == 0.0
    assert resolved["training"]["weight_decay_after"] == {"step": 2000, "value": 2.0e-4}

    unaligned = {key: value for key, value in SEALED.items() if key != "alignment"}
    config = _slant_config()
    config["model"]["output_window_mask"] = False
    raw = resolve_run_config(config, unaligned, CONTRACT, geometry, SHAPE)["model"]
    np.testing.assert_allclose(
        raw["head_samples_per_radius"],
        [-p * 1e-3 * 2 * radius / 0.002 for p in (0.3, 2.0, 3.1)],
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="model.output_window_mask: needs a preparation aligned"):
        resolve_run_config(_slant_config(), unaligned, CONTRACT, geometry, SHAPE)
    config = _slant_config()
    config["model"]["output_band_hz"] = [4, 246]
    with pytest.raises(ValueError, match="model.output_band_hz: hi \\+ 5 Hz must be below the 250"):
        resolve_run_config(config, SEALED, CONTRACT, geometry, SHAPE)


def test_off_priors_resolve_to_the_nersi_trace_defaults():
    resolved = resolve_run_config(_config(), SEALED, CONTRACT, _geometry(), SHAPE)
    for key in (
        "head_samples_per_radius",
        "static_shift_samples",
        "residual_slowness_samples_per_radius",
        "output_band_hz",
        "output_window",
    ):
        assert resolved["model"][key] == NERSI_TRACE_DEFAULTS[key], key
    assert "spatial_shape" not in resolved["model"]
    assert "sample_interval_s" not in resolved["model"]
    assert resolved["training"]["weight_decay_after"] is None


def test_config_differences_list_the_decoder_priors():
    resolved = resolve_run_config(_slant_config(), SEALED, CONTRACT, _geometry(), SHAPE)
    differences = config_differences(resolved, SEALED)
    prefixes = (
        "model.head_samples_per_radius: null -> [",
        "model.static_shift_samples: 0 -> 6",
        "model.residual_slowness_samples_per_radius: 0 -> ",
        "model.output_band_hz: null -> [4, 25]",
        "model.output_window: null -> {",
        "model.spatial_shape: null -> [2, 3, 2, 3]",
        "training.weight_decay_after: null -> {'step': 2000, 'value': 0.0002}",
    )
    for prefix in prefixes:
        assert any(item.startswith(prefix) for item in differences), prefix
    default = config_differences(
        resolve_run_config(_config(), SEALED, CONTRACT, _geometry(), SHAPE), SEALED
    )
    assert not any("head_samples" in item or "weight_decay_after" in item for item in default)
