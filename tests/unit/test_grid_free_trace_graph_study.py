"""Executable study variants preserve the declared comparison and evaluation controls."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_prediction_config,
    validate_relational_trace_graph_training_config,
)

STUDY = REPOSITORY_ROOT / "studies/study_026_grid_free_multi_relation_gnn"
TRAINING_CONFIGS = [
    STUDY / "config.yaml",
    STUDY / "config_smoke.yaml",
    *sorted((STUDY / "variants").glob("*.yaml")),
    *sorted((STUDY / "seeds").glob("*.yaml")),
]


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=lambda path: str(path.relative_to(STUDY)))
def test_study_training_configs_construct_declared_models(path: Path) -> None:
    config = load_resolved_config(path)
    model_config, graph, _ = validate_relational_trace_graph_training_config(config)
    model = RelationalTraceGraphInterpolator(**model_config)
    graph.validate_model_config(model.constructor_config())


@pytest.mark.parametrize(
    "path", sorted((STUDY / "inference").glob("*.yaml")), ids=lambda path: path.stem
)
def test_frozen_configs_keep_training_and_model_settings_in_checkpoint(path: Path) -> None:
    config = load_resolved_config(path)
    validate_relational_trace_graph_prediction_config(config)
    assert (
        not {"training", "training_data", "training_mask", "model", "graph", "geometry_features"}
        & config.keys()
    )


def test_ablation_table_keeps_pool_time_seed_budget_and_label_policy_fixed() -> None:
    baseline = load_resolved_config(STUDY / "config.yaml")
    baseline_model, _, _ = validate_relational_trace_graph_training_config(baseline)
    expected = {
        "mean": ("relational", "mean", "multi_relation", None, True),
        "plain_gcn": ("plain_gcn_row_normalized", "mean", "multi_relation", None, True),
        "untyped": ("untyped_edge_conditioned", "mean", "multi_relation", None, True),
        "single_4d": ("untyped_edge_conditioned", "mean", "single_4d", None, True),
        "without_source": ("relational", "learned_gate", "multi_relation", "source", True),
        "without_receiver": ("relational", "learned_gate", "multi_relation", "receiver", True),
        "without_cmp": ("relational", "learned_gate", "multi_relation", "cmp", True),
        "without_offset_azimuth": (
            "relational",
            "learned_gate",
            "multi_relation",
            "offset_azimuth",
            True,
        ),
        "without_explicit_azimuth_features": (
            "relational",
            "learned_gate",
            "multi_relation",
            None,
            False,
        ),
    }
    paths = sorted((STUDY / "variants").glob("*.yaml"))
    assert {path.stem for path in paths} == expected.keys()
    for path in paths:
        config = load_resolved_config(path)
        for section in (
            "project",
            "data",
            "training_data",
            "training_mask",
            "training",
            "evaluation",
            "geometry_features",
            "diagnostics",
        ):
            assert config[section] == baseline[section]
        model, graph, _ = validate_relational_trace_graph_training_config(config)
        assert (
            model["method_variant"],
            model["relation_fusion"],
            graph.topology,
            graph.excluded_relation,
            model["explicit_azimuth_features"],
        ) == expected[path.stem]
        for key in (
            "width",
            "message_passing_rounds",
            "time_downsample_factor",
            "stem_kernel_size",
            "temporal_kernel_size",
            "temporal_dilations",
        ):
            assert model[key] == baseline_model[key]
        assert graph.neighbors_per_relation == 8
        assert graph.single_4d_neighbors == 32
        assert graph.common_distance_scales_m == (320.0, 320.0)
        # IDW is a scored baseline. There is no model skip or zero-context exclusion switch.
        assert not {"idw_skip", "idw_residual", "skip_zero_context"} & config["model"].keys()
        assert config["evaluation"]["domain"] == "evaluation_target"


def test_training_seeds_share_fixed_partition_cases_masks_and_frozen_configs() -> None:
    configs = [
        load_resolved_config(STUDY / path)
        for path in ("config.yaml", "seeds/seed43.yaml", "seeds/seed44.yaml")
    ]
    assert [config["training"]["random_seed"] for config in configs] == [42, 43, 44]
    for config in configs:
        assert config["project"]["random_seed"] == 42
        for section in config.keys() - {"training"}:
            assert config[section] == configs[0][section]
        assert {
            key: value for key, value in config["training"].items() if key != "random_seed"
        } == {key: value for key, value in configs[0]["training"].items() if key != "random_seed"}
    inputs = yaml.safe_load((STUDY / "inputs.yaml").read_text())
    cases = inputs["datasets"][0]["benchmark_cases"]
    for name, binding in cases.items():
        frozen = load_resolved_config(STUDY / "inference" / f"{name}.yaml")
        assert frozen["benchmark_case"]["id"] == binding["case_id"]
        assert frozen["project"]["random_seed"] == binding["mask_seed"] == 42
        assert frozen["interpolation_mask"] == {
            "partition": binding["partition"],
            "kind": binding["mask_kind"],
            "missing_fraction": binding["missing_fraction"],
        }
        assert frozen["diagnostics"] == configs[0]["diagnostics"]
        if binding["status"] == "not_prepared":
            assert binding["observation_mask_sha256"] is None
        else:
            assert (
                binding["observation_mask_sha256"]
                == "26926fadc0f9ddf940c9db1f62abb94096118b74442082506816e34e04ae945a"
            )
