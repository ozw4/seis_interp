from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, ConfigurationError
from seis_interp.data.c3_benchmark_suite import c3_suite_case, suite_path
from seis_interp.data.c3_first_result_inputs import (
    build_c3_first_result_native_config,
    resolve_c3_first_result_inputs,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.pipelines.c3_first_results import (
    dispatch_c3_first_results_action,
    run_c3_first_results,
)
from seis_interp.pipelines.prepare_c3_benchmark import prepare_c3_benchmark_artifacts
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_prediction_config,
    validate_relational_trace_graph_training_config,
)
from tests.fixtures.c3_benchmark import (
    make_benchmark_interim,
    synthetic_benchmark_cases,
    synthetic_benchmark_config,
)

DIMENSIONS = C3BenchmarkDimensions((0, 4), (1, 2), (4, 2, 2, 2, 4))
STUDY = REPOSITORY_ROOT / "studies/study_028_c3_first_results"
SEEDS = {
    "partition": 42,
    "training": 20260908,
    "siren_sampler": 20260908,
    "ccnet_patches": 20260908,
    "gnn_episodes": 20260908,
}


@pytest.fixture(scope="module")
def suite_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("first_results_suite")
    interim = make_benchmark_interim(root / "source")
    directory = root / "suite"
    cases = synthetic_benchmark_cases()
    for case in cases:
        case["random_seed"] = 142
    suite = prepare_c3_benchmark_artifacts(
        interim,
        directory,
        config=synthetic_benchmark_config(),
        inputs={"cases": cases},
        dimensions=DIMENSIONS,
    )
    entry = c3_suite_case(suite, "validation_random_trace")
    volume = json.loads((suite_path(directory, entry["volume_dir"]) / "volume.json").read_text())
    return {
        "frozen_suite": {
            "manifest": str(directory / "benchmark_suite.json"),
            "expected_sha256": file_sha256(directory / "benchmark_suite.json"),
        },
        "required_cases": [entry["case_id"]],
        "validation_contract": {key: volume[key] for key in ("partition", "shape", "selection")},
        "outputs": {"runs": "runs", "results": "results"},
    }


def _write_inputs(tmp_path, suite_inputs):
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(suite_inputs))
    return path


def _fragment(action):
    name = "gnn-train" if action == "gnn-preflight" else action
    result = yaml.safe_load((STUDY / "methods" / f"{name.replace('-', '_')}.yaml").read_text())
    if action in ("gnn-train", "gnn-preflight"):
        result["training_data"]["time_samples"] = [0, 4]
    if action in ("pocs", "drr"):
        result["evaluation"] = synthetic_benchmark_config()["evaluation"]
    return result


def _plan(tmp_path, action, fragment):
    fragment_path = tmp_path / "method.yaml"
    fragment_path.write_text(yaml.safe_dump(fragment))
    plan = {
        "seeds": SEEDS,
        "execution": {
            "device": "cuda:1",
            "thread_count": 1,
            "action_timeout_seconds": 60,
            "preflight_timeout_seconds": 30,
        },
        "methods": {action: {"native_fragment": fragment_path.name}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(plan))
    return path


@pytest.mark.parametrize(
    "action",
    [
        "pocs",
        "drr",
        "siren",
        "ccnet-predict",
        "gnn-train",
        "gnn-predict",
        "gnn-preflight",
    ],
)
def test_native_seed_and_section_contracts(tmp_path, suite_inputs, action):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    native = build_c3_first_result_native_config(
        action, fragment=_fragment(action), binding=binding, seeds=SEEDS
    )
    assert native["project"] == {"random_seed": 42 if action.startswith("gnn") else 142}
    assert native["data"] == {"dataset_id": "synthetic_c3"}
    if action in ("gnn-train", "gnn-preflight"):
        assert not {"benchmark_case", "benchmark_volume", "interpolation_mask"} & native.keys()
        validate_relational_trace_graph_training_config(native)
    if action == "gnn-predict":
        validate_relational_trace_graph_prediction_config(native)
    if action == "siren":
        assert native["model"]["input_features"] == 6
        assert native["training"]["random_seed"] == SEEDS["training"]
    if action == "ccnet-predict":
        assert not {"model", "training"} & native.keys()
        assert set(native["evaluation"]) == {"primary_metric", "domain"}


def test_native_evaluation_projects_only_required_metric_keys(tmp_path, suite_inputs):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    manifest = deepcopy(binding.manifest)
    manifest["evaluation"]["additional_metrics"] = ["diagnostic_metric"]
    native = build_c3_first_result_native_config(
        "ccnet-predict",
        fragment=_fragment("ccnet-predict"),
        binding=replace(binding, manifest=manifest),
        seeds=SEEDS,
    )
    assert set(native["evaluation"]) == {"primary_metric", "domain"}


@pytest.mark.parametrize("action", ["gnn-train", "gnn-preflight"])
def test_gnn_native_fragment_keeps_explicit_physical_bound(tmp_path, suite_inputs, action):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    fragment = _fragment(action)
    fragment["training_data"]["max_abs_amplitude"] = 10000.0

    native = build_c3_first_result_native_config(
        action, fragment=fragment, binding=binding, seeds=SEEDS
    )

    assert native["training_data"] == {
        "pool": "all_train_traces",
        "time_samples": [0, 4],
        "max_abs_amplitude": 10000.0,
    }
    validate_relational_trace_graph_training_config(native)


@pytest.mark.parametrize(
    "changes",
    [
        {"pool": "mask_observed"},
        {"time_samples": [1, 4]},
        {"clip": True},
        {"max_abs_amplitude": True},
        {"max_abs_amplitude": None},
        {"max_abs_amplitude": 0},
        {"max_abs_amplitude": float("inf")},
    ],
)
def test_gnn_physical_bound_preserves_strict_pool_time_and_key_contract(
    tmp_path, suite_inputs, changes
):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    fragment = _fragment("gnn-train")
    fragment["training_data"].update({"max_abs_amplitude": 10000.0, **changes})
    with pytest.raises(ConfigurationError):
        build_c3_first_result_native_config(
            "gnn-train", fragment=fragment, binding=binding, seeds=SEEDS
        )


def test_ccnet_training_separates_partition_patch_and_model_seeds(tmp_path, suite_inputs):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    fragment = _fragment("ccnet-train")
    fragment["patches"]["random_seed"] = 71
    regions = {"fit": {"time": [0, 4]}, "selection": {"time": [0, 4]}}
    native = build_c3_first_result_native_config(
        "ccnet-train",
        fragment=fragment,
        binding=binding,
        seeds={**SEEDS, "ccnet_patches": 71},
        ccnet_regions=regions,
    )
    assert native["project"]["random_seed"] == 42
    assert native["training"]["random_seed"] == 20260908
    assert native["patches"]["random_seed"] == 71
    assert "benchmark_volume" not in native


def test_prediction_rejects_even_null_model_section(tmp_path, suite_inputs):
    binding = resolve_c3_first_result_inputs(
        _write_inputs(tmp_path, suite_inputs), dimensions=DIMENSIONS
    )
    with pytest.raises(ConfigurationError, match="exactly"):
        build_c3_first_result_native_config(
            "ccnet-predict",
            fragment={**_fragment("ccnet-predict"), "model": None},
            binding=binding,
            seeds=SEEDS,
        )


@pytest.mark.parametrize("change", ["hash", "case", "time", "shape", "test", "ccnet_time"])
def test_binding_rejects_wrong_frozen_contract(tmp_path, suite_inputs, change):
    inputs = deepcopy(suite_inputs)
    if change == "hash":
        inputs["frozen_suite"]["expected_sha256"] = "0" * 64
    elif change == "case":
        inputs["required_cases"] = ["missing_validation_case"]
    elif change == "time":
        inputs["validation_contract"]["selection"]["time"] = [1, 4]
    elif change == "shape":
        inputs["validation_contract"]["shape"][0] = 3
    elif change == "ccnet_time":
        inputs["ccnet_regions"] = {"time_samples": [0, 3]}
    else:
        inputs["required_cases"] = ["test_random_trace"]
    with pytest.raises((ValueError, KeyError)):
        resolve_c3_first_result_inputs(_write_inputs(tmp_path, inputs), dimensions=DIMENSIONS)


def test_dry_run_writes_nothing_and_rejects_case_override(tmp_path, suite_inputs):
    inputs = _write_inputs(tmp_path, suite_inputs)
    plan = _plan(tmp_path, "pocs", _fragment("pocs"))
    before = set(tmp_path.rglob("*"))
    result = run_c3_first_results(
        config_path=plan, inputs_path=inputs, action="pocs", dimensions=DIMENSIONS
    )
    assert result["status"] == "dry_run"
    assert result["request"]["native_config"]["project"]["random_seed"] == 142
    assert set(tmp_path.rglob("*")) == before
    result = run_c3_first_results(
        config_path=plan,
        inputs_path=inputs,
        action="pocs",
        case_id="test_random_trace",
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "blocked"
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("action", ["pocs", "drr"])
def test_tiny_suite_reaches_real_native_pipeline_in_fresh_process(tmp_path, suite_inputs, action):
    fragment = _fragment(action)
    fragment[action]["n_iterations"] = 2 if action == "pocs" else 1
    if action == "pocs":
        fragment[action].update(window_shape=None, overlap=None)
    else:
        fragment[action].update(rank=1, spatial_window_shape=None, spatial_overlap=None)
    result = run_c3_first_results(
        config_path=_plan(tmp_path, action, fragment),
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action=action,
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "success", result
    native = Path(result["native_run_directory"])
    assert (native / "artifacts/prediction.npy").is_file()
    saved = yaml.safe_load((native / "config.resolved.yaml").read_text())
    assert saved["project"]["random_seed"] == 142
    outer = Path(result["outer_run_directory"])
    assert json.loads((outer / "result.json").read_text())["status"] == "success"
    assert (
        json.loads((outer / "request.json").read_text())["input_hashes"]["suite"]
        == suite_inputs["frozen_suite"]["expected_sha256"]
    )


def test_execute_records_input_failure_without_native_run(tmp_path, suite_inputs):
    inputs = deepcopy(suite_inputs)
    inputs["frozen_suite"]["expected_sha256"] = "0" * 64
    result = run_c3_first_results(
        config_path=_plan(tmp_path, "pocs", _fragment("pocs")),
        inputs_path=_write_inputs(tmp_path, inputs),
        action="pocs",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "blocked"
    assert (Path(result["outer_run_directory"]) / "result.json").is_file()
    assert not Path(result["native_run_directory"]).exists()


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
def test_siren_bridge_supports_cartesian_complete_trace_variant(
    tmp_path, suite_inputs, shear, time_weight_scale, envelope
):
    fragment = _fragment("siren")
    fragment["model"].update(
        coordinate_features="cmp_cartesian_half_offset",
        input_features=5,
        hidden_width=8,
        hidden_layers=1,
        time_coordinate_scale=2.0,
        relative_receiver_y_time_shear_s_per_m=shear,
    )
    fragment["training"].update(
        max_steps=2,
        report_interval=1,
        batch_size=7,
        batch_mode="random_complete_traces",
        traces_per_step=None,
        learning_rate_schedule="cosine",
        minimum_learning_rate=1e-6,
    )
    if time_weight_scale != 1.0:
        fragment["training"]["initial_time_weight_scale"] = time_weight_scale
    if envelope:
        fragment["training"]["envelope_loss"] = {
            "weight": 1.0,
            "sigma_samples": [0.25, 0.5],
            "decay_steps": 2,
        }
    result = run_c3_first_results(
        config_path=_cpu_plan(tmp_path, {"siren": fragment}),
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="siren",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "success", result
    native = Path(result["native_run_directory"])
    run = json.loads((native / "run.json").read_text())
    assert run["coordinates"]["features"] == "cmp_cartesian_half_offset"
    assert run["coordinates"]["time_coordinate_scale"] == 2.0
    if shear:
        assert run["coordinates"]["relative_receiver_y_time_shear_s_per_m"] == shear
    assert run["training"]["batch_mode"] == "random_complete_traces"
    if time_weight_scale != 1.0:
        assert run["training"]["initial_time_weight_scale"] == time_weight_scale
    if envelope:
        assert run["training"]["envelope_loss"] == fragment["training"]["envelope_loss"]
        history = json.loads((native / "metrics.json").read_text())["training"]["history"]
        assert history[0]["envelope_weight"] == 1.0
        assert history[-1]["envelope_weight"] == 0.0


def test_native_failure_is_recorded(tmp_path, suite_inputs):
    fragment = _fragment("pocs")
    fragment["pocs"]["n_iterations"] = 0
    result = run_c3_first_results(
        config_path=_plan(tmp_path, "pocs", fragment),
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="pocs",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "failed"
    assert "n_iterations" in result["reason"]


def test_dispatch_gnn_training_passes_fixed_validation_only(tmp_path, suite_inputs, monkeypatch):
    plan = _plan(tmp_path, "gnn-train", _fragment("gnn-train"))
    result = run_c3_first_results(
        config_path=plan,
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="gnn-train",
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "dry_run", result
    request = result["request"]
    request.update(
        native_config_path=str(tmp_path / "native.yaml"),
        native_run_directory=str(tmp_path / "native"),
    )
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "seis_interp.pipelines.train_relational_trace_graph.train_relational_trace_graph_run",
        spy,
    )
    dispatch_c3_first_results_action(request)
    assert captured["validation_volume_dir"] == Path(request["paths"]["volume_dir"])
    assert not {"train_case_dir", "train_mask_dir", "train_volume_dir"} & captured.keys()


def test_action_timeout_is_retained(tmp_path, suite_inputs, monkeypatch):
    original = subprocess.run

    def timeout(command, **kwargs):
        if "--worker" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_c3_first_results(
        config_path=_plan(tmp_path, "pocs", _fragment("pocs")),
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="pocs",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "timeout"
    assert (
        json.loads((Path(result["outer_run_directory"]) / "result.json").read_text())["status"]
        == "timeout"
    )


def _cpu_plan(tmp_path, fragments):
    methods = {}
    for action, fragment in fragments.items():
        for section in ("training", "prediction"):
            if "device" in fragment.get(section, {}):
                fragment[section]["device"] = "cpu"
        path = tmp_path / f"{action}.yaml"
        path.write_text(yaml.safe_dump(fragment))
        methods[action] = {"native_fragment": path.name}
    config = {
        "seeds": SEEDS,
        "execution": {
            "device": "cpu",
            "thread_count": 1,
            "action_timeout_seconds": 60,
            "preflight_timeout_seconds": 30,
        },
        "methods": methods,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("amplitude_scaling", ["train_global_rms", "per_trace_rms"])
def test_tiny_siren_bridge_trains_observed_and_scores_all_targets(
    tmp_path, suite_inputs, amplitude_scaling
):
    fragment = _fragment("siren")
    fragment["model"].update(hidden_width=8, hidden_layers=1)
    fragment["training"].update(max_steps=2, batch_size=8, report_interval=1)
    fragment["prediction"]["batch_size"] = 16
    if amplitude_scaling == "per_trace_rms":
        fragment["training"]["amplitude_scaling"] = amplitude_scaling
        fragment["prediction"]["scale_interpolation"] = {
            "neighbors": 2,
            "power": 2.0,
            "distance_scales_m": [1.0] * 4,
        }
    result = run_c3_first_results(
        config_path=_cpu_plan(tmp_path, {"siren": fragment}),
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="siren",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "success", result
    native = Path(result["native_run_directory"])
    assert (native / "artifacts/final.pt").is_file()
    assert (native / "artifacts/prediction.npy").is_file()
    run = json.loads((native / "run.json").read_text())
    assert run["random_seed"] == 142
    assert run["amplitude"]["scaling"] == amplitude_scaling
    if amplitude_scaling == "per_trace_rms":
        assert (native / "artifacts/trace_amplitude_scales.npy").is_file()
        assert not run["amplitude"]["target_amplitudes_used_for_scale"]


@pytest.mark.parametrize("method", ["ccnet", "gnn"])
def test_tiny_neural_bridge_trains_and_predicts_native_final(tmp_path, suite_inputs, method):
    training, prediction = _fragment(f"{method}-train"), _fragment(f"{method}-predict")
    inputs = deepcopy(suite_inputs)
    if method == "ccnet":
        training["model"].update(hidden_channels=2, intermediate_channels=2, kernel_size=1)
        training["patches"].update(shape=[4, 1, 1, 2, 2], fit_count=1, selection_count=1)
        training["training"].update(max_epochs=1, validate_every_steps=1, report_every_steps=1)
        prediction["prediction"]["core_shape"] = [4, 1, 1, 2, 2]
        fit = {
            "time": [0, 4],
            "source_line": [0, 1],
            "shot_in_line": [0, 1],
            "relative_receiver_x": [3, 5],
            "relative_receiver_y": [32, 36],
        }
        inputs["ccnet_regions"] = {
            "fit": fit,
            "selection": {**fit, "shot_in_line": [1, 2]},
            "time_samples": [0, 4],
        }
    else:
        training["model"].update(
            width=8,
            message_passing_rounds=1,
            temporal_dilations=[1],
            attention_width=4,
            relation_embedding_dim=2,
        )
        training["graph"]["neighbors_per_relation"] = 1
        training["training"].update(max_steps=1, query_batch_size=2, validation_interval=1)
        training["evaluation"]["query_batch_size"] = 8
        prediction["prediction"]["query_batch_size"] = 8
    plan = _cpu_plan(tmp_path, {f"{method}-train": training, f"{method}-predict": prediction})
    inputs_path = _write_inputs(tmp_path, inputs)
    trained = run_c3_first_results(
        config_path=plan,
        inputs_path=inputs_path,
        action=f"{method}-train",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert trained["status"] == "success", trained
    if method == "ccnet":
        request = json.loads((Path(trained["outer_run_directory"]) / "request.json").read_text())
        # Each teacher region has 1 line x 1 shot x 2 receivers-x x 4 receivers-y.
        assert request["training_inputs"]["fit_trace_count"] == 8
        assert request["training_inputs"]["selection_trace_count"] == 8
    checkpoint = Path(trained["native_run_directory"]) / "artifacts/final.pt"
    predicted = run_c3_first_results(
        config_path=plan,
        inputs_path=inputs_path,
        action=f"{method}-predict",
        execute=True,
        checkpoint_path=checkpoint,
        dimensions=DIMENSIONS,
    )
    assert predicted["status"] == "success", predicted
    native = Path(predicted["native_run_directory"])
    assert (native / "artifacts/prediction.npy").is_file()
    resolved = yaml.safe_load((native / "config.resolved.yaml").read_text())
    assert resolved["project"]["random_seed"] == (142 if method == "ccnet" else 42)
    assert not {"model", "training"} & resolved.keys()


def test_renaming_best_checkpoint_does_not_make_it_final(tmp_path, monkeypatch):
    from types import SimpleNamespace

    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"synthetic renamed checkpoint")
    monkeypatch.setattr(
        "seis_interp.training.ccnet5d_checkpoints.load_ccnet5d_checkpoint",
        lambda path, device: SimpleNamespace(checkpoint_role="best_selection"),
    )
    request = {
        "action": "ccnet-predict",
        "preflight": False,
        "paths": {},
        "native_config_path": str(tmp_path / "config.yaml"),
        "native_run_directory": str(tmp_path / "native"),
        "suite_manifest": str(tmp_path / "benchmark_suite.json"),
        "dimensions": {
            "time_range": DIMENSIONS.time_range,
            "sail_line_numbers": DIMENSIONS.sail_line_numbers,
            "shape": DIMENSIONS.shape,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
    }
    with pytest.raises(ValueError, match="checkpoint_role=final"):
        dispatch_c3_first_results_action(request)
    assert not (tmp_path / "native").exists()


def test_partial_explicit_ccnet_region_is_not_replaced(tmp_path, suite_inputs):
    inputs = deepcopy(suite_inputs)
    inputs["ccnet_regions"] = {"fit": {"time": [0, 4]}, "selection": None}
    result = run_c3_first_results(
        config_path=_plan(tmp_path, "ccnet-train", _fragment("ccnet-train")),
        inputs_path=_write_inputs(tmp_path, inputs),
        action="ccnet-train",
        dimensions=DIMENSIONS,
    )
    assert result["status"] == "blocked"
    assert "resolved together" in result["reason"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1])
@pytest.mark.parametrize("key", ["action_timeout_seconds", "preflight_timeout_seconds"])
def test_nonfinite_or_nonpositive_timeout_is_rejected_before_writes(
    tmp_path, suite_inputs, key, value
):
    plan_path = _plan(tmp_path, "pocs", _fragment("pocs"))
    plan = yaml.safe_load(plan_path.read_text())
    plan["execution"][key] = value
    plan_path.write_text(yaml.safe_dump(plan))
    inputs_path = _write_inputs(tmp_path, suite_inputs)
    with pytest.raises(ConfigurationError, match="positive and finite"):
        run_c3_first_results(
            config_path=plan_path,
            inputs_path=inputs_path,
            action="pocs",
            execute=True,
            dimensions=DIMENSIONS,
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("query_counts", [(1,), (1, 8, 32)])
def test_dispatch_gnn_preflight_uses_declared_query_counts(tmp_path, monkeypatch, query_counts):
    request = {
        "action": "gnn-preflight",
        "preflight": True,
        "paths": {},
        "native_config_path": str(tmp_path / "config.yaml"),
        "native_run_directory": str(tmp_path / "native"),
        "suite_manifest": str(tmp_path / "benchmark_suite.json"),
        "case_id": "validation_random_trace",
        "dimensions": {
            "time_range": DIMENSIONS.time_range,
            "sail_line_numbers": DIMENSIONS.sail_line_numbers,
            "shape": DIMENSIONS.shape,
        },
        "experiment_config": {
            "methods": {
                "gnn-train": {"preflight": {"validation_query_counts": list(query_counts)}},
            },
        },
    }
    captured = {}

    def spy(action, **kwargs):
        assert action == "gnn-preflight"
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(
        "seis_interp.pipelines.preflight_c3_first_results_neural.run_c3_first_results_neural_preflight",
        spy,
    )
    assert dispatch_c3_first_results_action(request)["status"] == "success"
    assert captured["query_counts"] == query_counts
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize(
    ("train_seconds", "predict_seconds", "status"),
    [
        (4000, 100, "blocked"),
        (100, 4000, "blocked"),
        (3500, 3500, "success"),
        (None, 100, "blocked"),
    ],
)
def test_gnn_preflight_success_requires_both_full_actions_fit_budget(
    tmp_path, suite_inputs, monkeypatch, train_seconds, predict_seconds, status
):
    plan_path = _plan(tmp_path, "gnn-train", _fragment("gnn-train"))
    plan = yaml.safe_load(plan_path.read_text())
    plan["execution"]["action_timeout_seconds"] = 3600
    plan_path.write_text(yaml.safe_dump(plan))
    estimates = {
        "estimated_train_action_seconds": train_seconds,
        "estimated_final_predict_action_seconds": predict_seconds,
    }
    original = subprocess.run

    def measured(command, **kwargs):
        if "--worker" not in command:
            return original(command, **kwargs)
        output = Path(command[-1]).parent
        (output / "worker.result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "metrics": {"status": "success", "estimates": estimates},
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", measured)
    result = run_c3_first_results(
        config_path=plan_path,
        inputs_path=_write_inputs(tmp_path, suite_inputs),
        action="gnn-preflight",
        execute=True,
        dimensions=DIMENSIONS,
    )
    assert result["status"] == status
    assert result["metrics"] == {"status": "success", "estimates": estimates}
    outer = Path(result["outer_run_directory"])
    assert json.loads((outer / "result.json").read_text())["status"] == status
    assert json.loads((outer / "worker.result.json").read_text())["status"] == "success"
    assert not Path(result["native_run_directory"]).exists()
