from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.processing.c3_benchmark_contract import validate_c3_benchmark_contract
from seis_interp.processing.trace_splits import validated_c3_source_line_ranges

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_027_c3_na_benchmark"


@pytest.fixture
def config() -> dict:
    return load_resolved_config(STUDY_DIRECTORY / "config.yaml", repository_root=REPOSITORY_ROOT)


def test_materialized_contract_loads_with_resolved_spatial_starts(config: dict) -> None:
    validate_c3_benchmark_contract(config, require_resolved=True)

    assert config["data"]["dataset_id"] == "seg_c3_na"
    assert config["normalization"]["amplitude"] == "train_global_rms"
    assert config["model"] is None
    assert config["training"] is None
    assert config["study"]["status"] == "materialized_locked"
    assert config["evaluation"] == {
        "domain": "evaluation_target",
        "primary_metric": "physical_amplitude_global_snr_db",
        "metrics": ["physical_amplitude_global_snr_db"],
    }
    assert config["benchmark_volume"]["selection"] == {
        "time": [0, 384],
        "source_line": [25, 41],
        "shot_in_line": [28, 60],
        "relative_receiver_x": [0, 8],
        "relative_receiver_y": [18, 50],
    }
    assert config["sampling"]["source_line_ranges"] == {
        "train": [0, 25],
        "test": [25, 41],
        "validation": [41, 50],
    }


def test_recipes_and_training_seeds_are_distinct_config_data(config: dict) -> None:
    benchmark = config["c3_benchmark"]
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text())
    assert "mask_recipes" not in benchmark
    assert inputs["mask_contract"]["recipes"] == "inputs.yaml:cases"
    assert inputs["mask_contract"]["generate_over"] == "entire_canonical_partition"
    assert len(inputs["cases"]) == 20
    assert {case["partition"] for case in inputs["cases"]} == {"test", "validation"}
    assert config["project"]["random_seed"] == 42
    assert benchmark["training"]["time_samples"] == [0, 384]
    assert not set(benchmark["training"]["random_seeds"]) & {
        case["random_seed"] for case in inputs["cases"]
    }
    assert set(benchmark["methods"]) == {
        "pocs",
        "drr",
        "siren_5d",
        "ccnet5d",
        "relational_trace_graph",
    }
    assert benchmark["inference"] == {
        "amplitude_domain": "same_crop_observed_traces",
        "outside_crop_context": False,
        "target_coordinates": "allowed",
        "target_amplitudes": "scoring_only",
    }


def test_input_plan_defers_only_data_dependent_values(config: dict) -> None:
    inputs = yaml.safe_load((STUDY_DIRECTORY / "inputs.yaml").read_text())
    assert (STUDY_DIRECTORY / inputs["selection_config"]).is_file()
    assert (STUDY_DIRECTORY / inputs["source"]["manifest"]).is_file()
    assert not Path(inputs["outputs"]["root"]).is_absolute()
    plan = inputs["partition_plan"]
    assert plan["train"] == "all_global_source_lines_before_test"
    assert plan["test"] == "fixed_sail_line_selection"
    assert plan["validation"] == "all_global_source_lines_after_test"
    assert plan["validation_stop"] == "actual_source_line_count_from_geometry_qc"
    assert config["sampling"]["split_scope"] == "c3_source_line_blocks"
    # Any measured count greater than 41 can resolve this plan without changing test.
    # These are synthetic counts, not claims about the local survey.
    for measured_line_count in (42, 51):
        assert validated_c3_source_line_ranges(
            {"train": [0, 25], "test": [25, 41], "validation": [41, measured_line_count]}
        )["test"] == (25, 41)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("benchmark_volume.selection.time", [1, 385]),
        ("benchmark_volume.selection.time", [0, 383]),
        ("benchmark_volume.selection.time", [False, 384]),
        ("benchmark_volume.selection.source_line", [24, 40]),
        ("benchmark_volume.selection.source_line", [26, 42]),
        ("benchmark_volume.selection.shot_in_line", [0, 31]),
        ("benchmark_volume.selection.relative_receiver_x", [0, 7]),
        ("benchmark_volume.selection.relative_receiver_y", [0, 31]),
        ("benchmark_volume.selection.relative_receiver_y", [0.0, 32.0]),
        ("c3_benchmark.shape", [384, 15, 32, 8, 32]),
        ("c3_benchmark.shape", [384.0, 16, 32, 8, 32]),
        ("c3_benchmark.sail_lines.requested_inclusive", [24, 39]),
        ("c3_benchmark.sail_lines.index_range", [24, 40]),
        ("c3_benchmark.sail_lines.numbering", "one_based"),
        ("c3_benchmark.start_rule", "best_signal_energy"),
        ("c3_benchmark.training.time_samples", [384, 625]),
        ("c3_benchmark.axis_order", ["source_line", "time"]),
    ],
)
def test_rejects_changes_to_fixed_contract(config: dict, path: str, value: object) -> None:
    section = config
    *parents, name = path.split(".")
    for parent in parents:
        section = section[parent]
    section[name] = value

    with pytest.raises(ValueError, match=path):
        validate_c3_benchmark_contract(config)


def test_changing_both_declared_and_selected_lines_does_not_bypass_contract(config: dict) -> None:
    config["c3_benchmark"]["sail_lines"]["index_range"] = [24, 40]
    config["benchmark_volume"]["selection"]["source_line"] = [24, 40]
    with pytest.raises(ValueError, match="index_range"):
        validate_c3_benchmark_contract(config)


def test_requires_integer_ranges_at_preparation_boundary(config: dict) -> None:
    config["benchmark_volume"]["selection"].update(
        shot_in_line=None, relative_receiver_x=None, relative_receiver_y=None
    )
    validate_c3_benchmark_contract(config)
    with pytest.raises(ValueError, match="shot_in_line is unresolved"):
        validate_c3_benchmark_contract(config, require_resolved=True)


def test_original_numbers_can_map_to_different_internal_indices(config: dict) -> None:
    config["c3_benchmark"]["sail_lines"].update(
        numbering="original_sail_line_number",
        index_range=[24, 40],
        lines=[
            {"requested_sail_line_number": number, "source_line_index": number - 1}
            for number in range(25, 41)
        ],
    )
    config["benchmark_volume"]["selection"]["source_line"] = [24, 40]
    validate_c3_benchmark_contract(config)
    config["c3_benchmark"]["sail_lines"]["lines"][0]["requested_sail_line_number"] = 24
    with pytest.raises(ValueError, match="preserve"):
        validate_c3_benchmark_contract(config)


@pytest.mark.parametrize("shot_start,receiver_y_start", [(0, 0), (7, 18), (31, 36)])
def test_accepts_resolved_spatial_starts_without_changing_fixed_axes(
    config: dict, shot_start: int, receiver_y_start: int
) -> None:
    config["benchmark_volume"]["selection"].update(
        shot_in_line=[shot_start, shot_start + 32],
        relative_receiver_x=[0, 8],
        relative_receiver_y=[receiver_y_start, receiver_y_start + 32],
    )
    # Availability and physical grid completeness are checked by the existing index builder.
    validate_c3_benchmark_contract(config, require_resolved=True)
