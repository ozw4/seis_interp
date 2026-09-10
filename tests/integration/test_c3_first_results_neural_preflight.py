"""Small suite checks for disposable neural measurements and authorized teachers."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest
import torch
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_supervised_source
from seis_interp.data.c3_benchmark_suite import (
    c3_suite_case,
    load_c3_benchmark_input_manifest,
)
from seis_interp.data.c3_first_results_training_regions import (
    resolve_c3_first_results_ccnet_regions,
)
from seis_interp.pipelines import train_ccnet5d as ccnet_pipeline
from seis_interp.pipelines.preflight_c3_first_results_neural import (
    run_c3_first_results_neural_preflight,
)
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (2, 3), (4, 2, 2, 2, 4))
NERSI_DIMENSIONS = C3BenchmarkDimensions((0, 8), (2, 3), (8, 2, 2, 2, 8))
METHODS = REPOSITORY_ROOT / "studies/study_028_c3_first_results/methods"
NERSI_METHODS = REPOSITORY_ROOT / "studies/study_034_c3_nersi_baseline/methods"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("neural-preflight")
    config = synthetic_benchmark_config()
    config["c3_benchmark"]["sail_lines"].update(requested_inclusive=[2, 3], index_range=[2, 4])
    config["benchmark_volume"]["selection"]["source_line"] = [2, 4]
    interim = make_benchmark_interim(root / "source")
    output = root / "suite"
    prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=config,
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=DIMENSIONS,
    )
    return output


@pytest.fixture(scope="module")
def nersi_suite(tmp_path_factory):
    """Build a profile shape compatible with NeRSI's three 2x decoder blocks."""
    root = tmp_path_factory.mktemp("nersi-neural-preflight")
    config = synthetic_benchmark_config()
    config["c3_benchmark"]["shape"] = list(NERSI_DIMENSIONS.shape)
    config["c3_benchmark"]["sail_lines"].update(requested_inclusive=[2, 3], index_range=[2, 4])
    config["c3_benchmark"]["training"]["time_samples"] = [0, 8]
    config["benchmark_volume"]["selection"].update(time=[0, 8], source_line=[2, 4])
    interim = make_benchmark_interim(root / "source")
    output = root / "suite"
    prepare_c3_benchmark_artifacts(
        interim,
        output,
        config=config,
        inputs={"cases": synthetic_benchmark_cases()},
        dimensions=NERSI_DIMENSIONS,
    )
    return output


def _regions(suite, **kwargs):
    return resolve_c3_first_results_ccnet_regions(
        suite,
        dimensions=DIMENSIONS,
        fit_source_line_range=(0, 1),
        selection_source_line_range=kwargs.get("selection_source_line_range", (1, 2)),
        spatial_lengths=(2, 2, 4),
    )


def _config(suite, action):
    filename = {
        "siren": "siren",
        "nersi": "nersi_preflight",
        "ccnet-train": "ccnet_train",
        "gnn-preflight": "gnn_train",
    }[action]
    method_directory = NERSI_METHODS if action == "nersi" else METHODS
    config = load_resolved_config(method_directory / f"{filename}.yaml")
    config.update(project={"random_seed": 42}, data={"dataset_id": "synthetic_c3"})
    config["training"]["device"] = "cpu"
    if action == "ccnet-train":
        regions = _regions(suite)
        config["supervision"].update(
            {name: regions[name] for name in ("fit_region", "selection_region")}
        )
        config["patches"].update(shape=[2, 1, 1, 2, 2], fit_count=2, selection_count=1)
        config["model"].update(hidden_channels=2, intermediate_channels=2)
    elif action == "gnn-preflight":
        config["model"].update(width=8, attention_width=8)
        config["training_data"]["time_samples"] = [0, 4]
        config["training"]["query_batch_size"] = 2
        config["evaluation"]["query_batch_size"] = 2
    elif action == "nersi":
        config["model"].update(
            fourier_components=2,
            encoder_width=8,
            latent_channels=2,
            decoder_channels=[2, 2, 2],
            kernel_size=1,
        )
        config["training"].update(
            profiles_per_step=1,
            max_steps=20,
            report_interval=10,
        )
        config["prediction"]["batch_size"] = 3
    else:
        config["model"].update(hidden_width=8, hidden_layers=2)
        config["training"]["batch_size"] = 8
        config["prediction"]["batch_size"] = 8
    return config


def _nested_mapping_keys(value):
    keys = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(_nested_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_mapping_keys(child))
    return keys


def test_regions_resolve_geometry_and_reject_nontrain_overlap_or_wrong_time(suite):
    result = _regions(suite)
    assert result["fit_region"]["source_line"] == [0, 1]
    assert result["selection_region"]["source_line"] == [1, 2]
    assert result["fit_trace_count"] == result["selection_trace_count"] == 16
    assert result["time_samples"] == [0, 4]
    for lines, message in (((2, 3), "authorized"), ((0, 1), "disjoint")):
        with pytest.raises(ValueError, match=message):
            _regions(suite, selection_source_line_range=lines)
    wrong_time = deepcopy(result["fit_region"])
    wrong_time["time"] = [0, 8]
    with pytest.raises(ValueError, match="authorized training time"):
        load_c3_benchmark_supervised_source(
            suite,
            fit_region=wrong_time,
            selection_region=result["selection_region"],
            dimensions=DIMENSIONS,
        )


@pytest.mark.parametrize(
    "action,amplitude_scaling",
    [
        ("siren", "train_global_rms"),
        ("siren", "per_trace_rms"),
        ("ccnet-train", "train_global_rms"),
        ("gnn-preflight", "train_global_rms"),
    ],
)
def test_neural_smoke_measures_tiny_fixed_suite_without_full_prediction(
    suite, tmp_path, action, amplitude_scaling
):
    config = _config(suite, action)
    if amplitude_scaling == "per_trace_rms":
        config["training"]["amplitude_scaling"] = amplitude_scaling
        config["prediction"]["scale_interpolation"] = {
            "neighbors": 2,
            "power": 2.0,
            "distance_scales_m": [1.0] * 4,
        }
    if action == "gnn-preflight":
        config["training_data"]["max_abs_amplitude"] = 10000.0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prediction_path = tmp_path / "prediction.yaml"
    prediction_path.write_text(yaml.safe_dump({"prediction": {"core_shape": [2, 1, 1, 2, 2]}}))
    output = tmp_path / "smoke"
    report = run_c3_first_results_neural_preflight(
        action,
        config_path=config_path,
        suite_dir=suite,
        case_id="validation_random_trace",
        output_dir=output,
        dimensions=DIMENSIONS,
        prediction_config_path=prediction_path,
        query_counts=(1, 2),
        query_limit=2,
    )
    assert report["status"] == "success", report
    assert not report["training_started"]
    assert not report["full_validation_prediction_completed"]
    assert not report["state_reused_by_pilot"]
    assert json.loads((output / "preflight.json").read_text()) == report
    assert load_resolved_config(config_path) == config
    if action == "gnn-preflight":
        train = report["training_measurement"]
        assert train["pool"] == "all_train_traces"
        assert train["time_samples"] == [0, 4]
        assert train["query_count"] == 2
        assert train["episode_seed"] == 20260908
        manifest = load_c3_benchmark_input_manifest(suite, dimensions=DIMENSIONS)
        entry = c3_suite_case(manifest, "validation_random_trace")
        assert (
            report["validation_measurements"][-1]["input"]["benchmark"]["case_id"]
            == entry["case_id"]
        )
        assert report["estimates"]["trainer_validation_passes"] == 1
        assert "model_metrics" not in report["validation_measurements"][-1]
    elif action == "ccnet-train":
        assert report["smoke_steps"] == 2
        assert report["halo_radius"] == 4
        assert report["measured_maximum_tile_shape"] == report["full_volume_shape"]
        assert [tile["role"] for tile in report["tile_measurements"]] == [
            "maximum_halo",
            "edge",
        ]
        for tile in report["tile_measurements"]:
            assert tile["input_shape"] == tile["output_shape"]
            assert tile["forward_seconds"] > 0
            assert tile["resources"]["process_max_rss_bytes"] > 0
    else:
        assert report["smoke_steps"] == 10
        assert report["sampled_prediction_point_count"] == 8
        if amplitude_scaling == "per_trace_rms":
            assert report["amplitude_scaling"] == amplitude_scaling
            assert report["scale_interpolation"] == config["prediction"]["scale_interpolation"]


def test_nersi_preflight_runs_ten_observed_only_steps_without_target_metrics(nersi_suite, tmp_path):
    config = _config(nersi_suite, "nersi")
    config_path = tmp_path / "nersi.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "smoke"

    report = run_c3_first_results_neural_preflight(
        "nersi",
        config_path=config_path,
        suite_dir=nersi_suite,
        case_id="validation_random_trace",
        output_dir=output,
        dimensions=NERSI_DIMENSIONS,
        smoke_steps=10,
    )

    assert report["status"] == "success", report
    assert report["action"] == "nersi"
    assert report["scope"] == "disposable_neural_preflight"
    assert report["smoke_steps"] == 10
    assert report["training_random_seed"] == 20260908
    assert report["profile_shape"] == [8, 8]
    # The synthetic validation partition has one source line, two shots, and
    # two receiver-x positions; receiver-y remains the profile output axis.
    assert report["profile_count"] == 1 * 2 * 2 == 4
    assert 0 < report["training_profile_count"] <= report["profile_count"]
    assert report["observed_trace_count"] > 0
    assert report["observed_sample_count"] == report["observed_trace_count"] * 8
    assert report["sampled_prediction_profile_count"] == 3
    assert report["model_config"]["profile_shape"] == (8, 8)
    assert report["model_config"]["fourier_components"] == 2
    assert report["parameter_count"] > 0
    assert not report["training_started"]
    assert not report["full_validation_prediction_completed"]
    assert not report["state_reused_by_pilot"]
    forbidden_metric_keys = {
        "evaluation_target",
        "snr_db",
        "rmse",
        "reference_energy",
        "error_energy",
        "target_metric",
        "target_metrics",
    }
    assert not (forbidden_metric_keys & _nested_mapping_keys(report))
    assert all(math.isfinite(value) and value >= 0.0 for value in report["timings"].values())
    assert report["estimates"]["kind"] == "linear_extrapolation_not_measured_full_run"
    for name in ("estimated_training_seconds", "estimated_full_profile_prediction_seconds"):
        assert math.isfinite(report["estimates"][name])
        assert report["estimates"][name] >= 0.0
    assert report["resources"]["process_max_rss_bytes"] > 0
    assert report["resources"]["cpu_memory_scope"] == "process_lifetime_peak"
    assert report["resources"]["cuda_max_memory_allocated_bytes"] is None
    assert report["resources"]["cuda_max_memory_reserved_bytes"] is None
    assert report["resources"]["cuda_memory_scope"] == "unmeasured"
    stored = json.loads((output / "preflight.json").read_text())
    assert stored == json.loads(json.dumps(report, allow_nan=False))
    assert load_resolved_config(config_path) == config


@pytest.mark.parametrize("problem", ["wrong_seed", "query_limit"])
def test_gnn_preflight_records_blocked_without_starting_pilot(suite, tmp_path, problem):
    config = _config(suite, "gnn-preflight")
    if problem == "wrong_seed":
        config["project"]["random_seed"] = 142
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    report = run_c3_first_results_neural_preflight(
        "gnn-preflight",
        config_path=path,
        suite_dir=suite,
        case_id="validation_random_trace",
        output_dir=tmp_path / "smoke",
        dimensions=DIMENSIONS,
        query_counts=(1, 2),
        query_limit=33 if problem == "query_limit" else 2,
    )
    assert report["status"] == "blocked"
    assert not report["training_started"]
    assert not report["full_validation_prediction_completed"]


def test_gnn_training_batch_preflight_rechecks_physical_bound(suite, tmp_path, monkeypatch):
    config = _config(suite, "gnn-preflight")
    config["training_data"]["max_abs_amplitude"] = 1e-12
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "seis_interp.pipelines.preflight_c3_first_results_neural.preflight_relational_trace_graph_run",
        lambda **kwargs: {"status": "success"},
    )
    monkeypatch.setattr(
        "seis_interp.pipelines.preflight_c3_first_results_neural."
        "measure_c3_first_results_graph_training_batch",
        lambda *args, **kwargs: pytest.fail("invalid training amplitudes must block the batch"),
    )

    report = run_c3_first_results_neural_preflight(
        "gnn-preflight",
        config_path=path,
        suite_dir=suite,
        case_id="validation_random_trace",
        output_dir=tmp_path / "smoke",
        dimensions=DIMENSIONS,
        query_counts=(1,),
        query_limit=1,
    )

    assert report["status"] == "blocked"
    assert not report["training_started"]
    assert "max_abs_amplitude=1e-12" in report["blockers"][0]["message"]
    assert "training_measurement" not in report


@pytest.mark.parametrize(
    "shear,time_weight_scale,envelope",
    [
        (0.0, 1.0, False),
        (0.0006, 1.0, False),
        (0.0, 3.0, False),
        (0.0, 1.0, True),
        (0.0, 3.0, True),
    ],
)
def test_siren_preflight_uses_cartesian_complete_trace_training(
    suite, tmp_path, shear, time_weight_scale, envelope
):
    config = _config(suite, "siren")
    config["model"].update(
        coordinate_features="cmp_cartesian_half_offset",
        input_features=5,
        time_coordinate_scale=2.0,
        relative_receiver_y_time_shear_s_per_m=shear,
    )
    config["training"].update(
        batch_mode="random_complete_traces",
        traces_per_step=None,
        learning_rate_schedule="cosine",
        minimum_learning_rate=1e-6,
    )
    if time_weight_scale != 1.0:
        config["training"]["initial_time_weight_scale"] = time_weight_scale
    if envelope:
        config["training"]["envelope_loss"] = {
            "weight": 1.0,
            "sigma_samples": [0.25, 0.5],
            "decay_steps": 2500,
        }
    path = tmp_path / "complete.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_c3_first_results_neural_preflight(
        "siren",
        config_path=path,
        suite_dir=suite,
        case_id="validation_random_trace",
        output_dir=tmp_path / "smoke",
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "success", result
    assert result["smoke_steps"] == 10
    assert result["complete_trace_training"] == {
        "traces_per_step": None,
        "learning_rate_schedule": "cosine",
        "minimum_learning_rate": 1e-6,
        **({"envelope_loss": config["training"]["envelope_loss"]} if envelope else {}),
    }
    assert not result["state_reused_by_pilot"]
    if time_weight_scale != 1.0:
        assert result["initial_time_weight_scale"] == time_weight_scale
    if envelope:
        assert result["training_history"][-1]["envelope_weight"] == pytest.approx(1 - 9 / 2499)
        assert result["estimates"]["training_scope"] == (
            "envelope_active_smoke_extrapolated_to_all_updates"
        )
    if shear:
        assert result["relative_receiver_y_time_shear_s_per_m"] == shear
    else:
        assert "relative_receiver_y_time_shear_s_per_m" not in result


def test_ccnet_preflight_measures_two_full_batches_and_propagates_benchmark_false(
    suite, tmp_path, monkeypatch
):
    config = _config(suite, "ccnet-train")
    config["patches"]["fit_count"] = 5
    config["training"].update(batch_size=3, max_epochs=2, cudnn_benchmark=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prediction_path = tmp_path / "prediction.yaml"
    prediction_path.write_text(yaml.safe_dump({"prediction": {"core_shape": [2, 1, 1, 2, 2]}}))
    seed = ccnet_pipeline.seed_global_model_initialization
    constructor = ccnet_pipeline.CCNet5D
    modes = []
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)

    def seed_with_cuda_behavior(*args, **kwargs):
        seed(*args, **kwargs)
        torch.backends.cudnn.benchmark = True

    def construct(*args, **kwargs):
        modes.append(torch.backends.cudnn.benchmark)
        return constructor(*args, **kwargs)

    monkeypatch.setattr(ccnet_pipeline, "seed_global_model_initialization", seed_with_cuda_behavior)
    monkeypatch.setattr(ccnet_pipeline, "CCNet5D", construct)
    output = tmp_path / "smoke"
    report = run_c3_first_results_neural_preflight(
        "ccnet-train",
        config_path=config_path,
        suite_dir=suite,
        case_id="validation_random_trace",
        output_dir=output,
        dimensions=DIMENSIONS,
        prediction_config_path=prediction_path,
    )
    assert report["status"] == "success", report
    assert report["smoke_steps"] == 2
    assert modes == [False]
    smoke_config = yaml.safe_load((output / "smoke_config.yaml").read_text())
    assert smoke_config["patches"]["fit_count"] == 6
    assert smoke_config["training"]["batch_size"] == 3
    assert smoke_config["training"]["cudnn_benchmark"] is False
    metrics = json.loads((output / "smoke_native/metrics.json").read_text())
    assert [row["sample_count"] for row in metrics["training_history"]] == [24, 24]
    assert report["estimates"]["training_seconds"] == pytest.approx(
        report["timings"]["smoke_training_and_selection_seconds"] * 4 / 2
    )
    assert load_resolved_config(config_path) == config
