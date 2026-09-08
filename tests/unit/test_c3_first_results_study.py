"""The pilot plan is finite, data-independent, and preserves native model settings."""

from pathlib import Path

import pytest

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.pipelines.interpolate_drr import _drr_settings
from seis_interp.pipelines.interpolate_pocs import _pocs_settings
from seis_interp.pipelines.interpolate_siren import _model_constructor_config, _training_settings
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_training_config,
)

STUDY = REPOSITORY_ROOT / "studies/study_028_c3_first_results"


def test_plan_keeps_one_fixed_validation_and_separate_model_fragments():
    plan = load_resolved_config(STUDY / "config.yaml")
    inputs = load_resolved_config(STUDY / "inputs.yaml")
    assert inputs["required_cases"] == ["c3_benchmark_validation_random_trace_80_seed142"]
    assert inputs["frozen_suite"]["expected_sha256"] == (
        "6447a35cc7d4de43532ee8e2e0de3245ac0e93d855efa7ff59c98a85e1631cab"
    )
    assert inputs["validation_contract"]["shape"] == [384, 9, 32, 8, 32]
    assert inputs["validation_contract"]["selection"]["source_line"] == [41, 50]
    assert inputs["validation_contract"]["selection"]["time"] == [0, 384]
    assert (
        not {"model", "training", "project", "benchmark_volume", "interpolation_mask"} & plan.keys()
    )
    assert not {"mask_recipes", "interpolation_mask", "sampling"} & inputs.keys()
    for method in plan["methods"].values():
        path = STUDY / method["native_fragment"]
        fragment = load_resolved_config(path)
        assert path.is_file()
        assert (
            not {
                "project",
                "data",
                "sampling",
                "benchmark_case",
                "benchmark_volume",
                "interpolation_mask",
                "extends",
            }
            & fragment.keys()
        )
    assert not Path(inputs["outputs"]["runs"]).is_absolute()


def test_plan_has_finite_budgets_final_checkpoint_rule_and_explicit_seed_roles():
    plan = load_resolved_config(STUDY / "config.yaml")
    execution = plan["execution"]
    assert execution["default_action"] == "dry_run"
    assert execution["one_action_per_invocation"]
    assert 0 < execution["action_timeout_seconds"] <= 3600
    assert 0 < execution["preflight_timeout_seconds"] <= 600
    assert execution["cpu_training_fallback"] is False
    assert plan["seeds"] == {
        "partition": 42,
        "training": 20260908,
        "siren_sampler": 20260908,
        "ccnet_patches": 20260908,
        "gnn_episodes": 20260908,
    }
    assert len(plan["completion"]["required_methods"]) == 5
    assert plan["completion"]["snr_threshold"] is None
    assert plan["completion"]["test_execution"] is False
    assert plan["evaluation"]["primary_checkpoint"] == "final_after_declared_budget"
    assert plan["evaluation"]["context_free_targets"] == "include"


@pytest.mark.parametrize("method", ["pocs", "drr"])
def test_classical_pilot_fragments_pass_native_validators(method):
    config = load_resolved_config(STUDY / f"methods/{method}.yaml")
    if method == "pocs":
        settings = _pocs_settings(config)
        assert settings.n_iterations == 20
        assert settings.window_shape == (128, 4, 8, 4, 16)
    else:
        settings = _drr_settings(config)
        assert settings.n_iterations == 5
        assert settings.spatial_window_shape == (4, 8, 4, 8)
        assert settings.frequency_min_hz == 0 and settings.frequency_max_hz is None


def test_siren_retains_six_features_and_finite_observed_fit_budget():
    config = load_resolved_config(STUDY / "methods/siren.yaml")
    model = _model_constructor_config(config)
    training = _training_settings(config)
    assert model["input_features"] == 6
    assert (model["hidden_width"], model["hidden_layers"]) == (256, 4)
    assert training.random_seed == 20260908
    assert (training.max_steps, training.batch_size) == (2000, 16384)


def test_existing_benchmark_selection_and_hash_are_preserved():
    study = REPOSITORY_ROOT / "studies/study_027_c3_na_benchmark"
    config = load_resolved_config(study / "config.yaml")
    inputs = load_resolved_config(study / "inputs.yaml")
    assert config["benchmark_volume"]["selection"] == {
        "time": [0, 384],
        "source_line": [25, 41],
        "shot_in_line": [28, 60],
        "relative_receiver_x": [0, 8],
        "relative_receiver_y": [18, 50],
    }
    assert config["sampling"]["source_line_ranges"] == {
        "train": [0, 25],
        "validation": [41, 50],
        "test": [25, 41],
    }
    assert len(inputs["cases"]) == 20
    assert (
        inputs["frozen_suite"]["sha256"]
        == load_resolved_config(STUDY / "inputs.yaml")["frozen_suite"]["expected_sha256"]
    )


def test_single_gnn_resource_revision_preserves_training_and_geometry_contract():
    plan = load_resolved_config(STUDY / "config.yaml")
    original = load_resolved_config(STUDY / "methods/gnn_train.yaml")
    revised = load_resolved_config(STUDY / plan["methods"]["gnn-train"]["native_fragment"])
    assert plan["resource_revisions"]["gnn"]["revision_count"] == 1
    assert original["graph"]["neighbors_per_relation"] == 4
    assert original["evaluation"]["query_batch_size"] == 8
    original["graph"]["neighbors_per_relation"] = 2
    original["evaluation"]["query_batch_size"] = 32
    assert revised == original
    _, graph, options = validate_relational_trace_graph_training_config(
        {**revised, "project": {"random_seed": 42}, "data": {"dataset_id": "seg_c3_na"}}
    )
    assert graph.neighbors_per_relation == 2
    assert options["query_batch_size"] == 4
    assert options["max_steps"] == 200
    prediction = load_resolved_config(STUDY / plan["methods"]["gnn-predict"]["native_fragment"])
    assert prediction["prediction"]["query_batch_size"] == 32
