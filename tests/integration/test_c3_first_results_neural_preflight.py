"""Small suite checks for disposable neural measurements and authorized teachers."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.data.c3_benchmark_inputs import load_c3_benchmark_supervised_source
from seis_interp.data.c3_benchmark_suite import c3_suite_case, load_c3_benchmark_input_manifest
from seis_interp.data.c3_first_results_training_regions import (
    resolve_c3_first_results_ccnet_regions,
)
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
METHODS = REPOSITORY_ROOT / "studies/study_028_c3_first_results/methods"


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


def _regions(suite, **kwargs):
    return resolve_c3_first_results_ccnet_regions(
        suite,
        dimensions=DIMENSIONS,
        fit_source_line_range=(0, 1),
        selection_source_line_range=kwargs.get("selection_source_line_range", (1, 2)),
        spatial_lengths=(2, 2, 4),
    )


def _config(suite, action):
    filename = {"siren": "siren", "ccnet-train": "ccnet_train", "gnn-preflight": "gnn_train"}[
        action
    ]
    config = load_resolved_config(METHODS / f"{filename}.yaml")
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
    else:
        config["model"].update(hidden_width=8, hidden_layers=2)
        config["training"]["batch_size"] = 8
        config["prediction"]["batch_size"] = 8
    return config


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


@pytest.mark.parametrize("action", ["siren", "ccnet-train", "gnn-preflight"])
def test_neural_smoke_measures_tiny_fixed_suite_without_full_prediction(suite, tmp_path, action):
    config = _config(suite, action)
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
        assert [tile["role"] for tile in report["tile_measurements"]] == ["maximum_halo", "edge"]
        for tile in report["tile_measurements"]:
            assert tile["input_shape"] == tile["output_shape"]
            assert tile["forward_seconds"] > 0
            assert tile["resources"]["process_max_rss_bytes"] > 0
    else:
        assert report["smoke_steps"] == 10
        assert report["sampled_prediction_point_count"] == 8


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
