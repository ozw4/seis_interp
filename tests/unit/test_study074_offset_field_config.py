"""The committed study_074 specification is executable before any run exists.

The first axis is the weight of the log-offset coordinate in the neighbour ranking,
and the parameter count stays canvas independent, so the one declared dataset
difference is the inert declared time count.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from seis_interp.c3_fair_grid_config import c3_experiment_methods, validate_c3_fair_grid_configs
from seis_interp.processing.c3_region_search import (
    c3_search_candidates,
    resolve_c3_candidate,
    validate_c3_experiment_plan,
)
from seis_interp.processing.experiment_commonality import load_experiment_grid
from seis_interp.processing.trace_field_coordinates import field_coordinates
from seis_interp.training.forge_mvp_methods import model_parameter_count

STUDY = Path("studies/study_074_forge_c3_offset_field")
PREVIOUS = Path("studies/study_071_forge_c3_context_field")
TARGET = 363269
METHOD = "cinr_real"
PARAMETERS = 350114
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0]
BASES = [1.05, 1.075, 1.1, 1.15, 1.25]
CANVAS = {"c3": (384, 384), "forge": (1001, 2172)}
ADMISSION = {"c3": 4, "forge": 11}


def _load(dataset, name, study=STUDY):
    return yaml.safe_load((study / dataset / f"{name}.yaml").read_text())


@pytest.fixture(params=sorted(CANVAS))
def dataset(request):
    return request.param


def _candidates(dataset):
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    return c3_search_candidates(_load(dataset, METHOD), grid["methods"][METHOD])


def test_both_datasets_declare_only_the_conditioned_field(dataset):
    contract = _load(dataset, "config")
    assert c3_experiment_methods(contract) == (METHOD,)
    assert contract["study_id"] == "study_074_forge_c3_offset_field"
    assert contract["training_loss"] == "masked_trace_mse"


def test_the_shared_grid_is_pinned_and_the_differences_are_declared(dataset):
    contract = _load(dataset, "config")
    digest = hashlib.sha256((STUDY / "grid.yaml").read_bytes()).hexdigest()
    assert contract["common_grid"] == {"path": "../grid.yaml", "sha256": digest}
    assert list(load_experiment_grid(STUDY / dataset, contract)["methods"]) == [METHOD]


def test_the_inert_declared_time_count_is_the_only_method_difference():
    declared = yaml.safe_load((STUDY / "dataset_differences.yaml").read_text())
    assert set(declared["methods"]) == {METHOD}
    assert set(declared["methods"][METHOD]) == {"model.time_sample_count"}


def test_the_first_axis_is_the_offset_weight_in_the_ranking():
    axes = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"][METHOD]
    assert [item["set"] for item in axes["axis_a"]] == [
        {"context.offset_weight": value} for value in WEIGHTS
    ]
    assert [item["set"] for item in axes["axis_b"]] == [
        {"model.frequency_base": value} for value in BASES
    ]
    # a weight of zero keeps study_071's ranking while the field still reads offset
    assert WEIGHTS[0] == 0.0


def test_the_grid_expands_to_25_candidates_with_one_canvas_independent_count(dataset):
    native, canvas = CANVAS[dataset]
    candidates = _candidates(dataset)
    assert [item["id"] for item in candidates] == [
        f"a{a:02d}_b{b:02d}" for a in range(1, 6) for b in range(1, 6)
    ]
    assert {model_parameter_count(item["config"], canvas) for item in candidates} == {PARAMETERS}
    assert abs(PARAMETERS / TARGET - 1) <= 0.05
    model = _load(dataset, METHOD)["model"]
    assert model["input_features"] == 5 and model["width"] == 112
    assert model["time_sample_count"] == native and canvas % model["time_downsample_factor"] == 0


def test_every_candidate_reads_offset_and_keeps_the_fixed_neighbour_count(dataset):
    seen = set()
    for candidate in _candidates(dataset):
        config = candidate["config"]
        assert config["model"]["input_features"] == 5
        assert set(config["context"]) == {"neighbor_count", "offset_weight"}
        assert config["context"]["neighbor_count"] == 16
        assert "distance" not in config["context"]
        seen.add(config["context"]["offset_weight"])
    assert sorted(seen) == WEIGHTS


def test_the_log_offset_axis_is_inside_the_unclipped_span(dataset):
    """Both acquisitions sit strictly inside the declared four-decade span."""
    root = Path("runs/study_071_forge_c3_context_field")
    reference = {
        "c3": root / "20261109T042809568788Z_766f4f10a666_traces/input/geometry.parquet",
        "forge": root / "20261109T042814452404Z_766f4f10a666_forge/input/real_geometry.parquet",
    }[dataset]
    if not reference.exists():
        pytest.skip("the study_071 production inputs are not present in this checkout")
    import pandas as pd

    shape = yaml.safe_load((STUDY / dataset / "config.yaml").read_text())["spatial_shape"]
    axis = field_coordinates(pd.read_parquet(reference), shape, 5)[:, 4]
    assert 0.0 < axis.min() < axis.max() < 1.0


def test_the_admission_request_follows_the_measured_memory(dataset):
    contract = _load(dataset, "config")
    assert contract["cuda_execution"]["minimum_free_memory_gib"] == {METHOD: ADMISSION[dataset]}
    assert contract["method_execution"] == {METHOD: {"parallel_candidates": 4}}


def test_the_c3_contract_validates_the_resolved_method():
    contract = _load("c3", "config")
    validate_c3_experiment_plan(contract, yaml.safe_load((STUDY / "grid.yaml").read_text()))
    resolved = resolve_c3_candidate(_load("c3", METHOD), contract["observed_trace_count"], 64)
    validate_c3_fair_grid_configs({METHOD: resolved}, contract)


def test_everything_but_the_method_matches_study_071(dataset):
    contract = _load(dataset, "config")
    previous = _load(dataset, "config", PREVIOUS)
    for key in (
        "candidate_id",
        "spatial_shape",
        "time_sample_count",
        "mask_id",
        "mask_seed",
        "mask_study_id",
        "validation_study_id",
        "trace_split",
        "validation_holdout",
        "preprocessing",
        "selection",
        "training_loss",
        "model_seed",
        "matched_training",
    ):
        assert contract[key] == previous[key], key


def test_the_evaluation_isolates_the_offset_axis_and_records_the_prediction():
    evaluation = yaml.safe_load((STUDY / "evaluation.yaml").read_text())
    families = evaluation["contrasts"]["holm_families"]
    assert families["offset_axis_snr"]["contrast"] == "cinr_real_074_minus_cinr_real_071"
    assert families["offset_axis_relmse"]["metric"] == "relative_mse"
    assert evaluation["left"]["label"] == "cinr_real_074"
    assert set(evaluation["right"]["sources"]) == {"study_067", "study_070", "study_071"}
    assert evaluation["right"]["sources"]["study_071"]["label"] == "cinr_real_071"
    prediction = evaluation["prediction"]
    assert prediction["declared_before"].startswith("any_study_074")
    assert "improves" in prediction["forge"] and "neutral" in prediction["c3"]
    assert "refutes" in prediction["falsifier"]
    for source in evaluation["right"]["sources"].values():
        for path, digest in (
            (source["grid"], source["grid_sha256"]),
            (source["evaluation"], source["evaluation_sha256"]),
            *(
                (item["path"] + "/experiment.json", item["sha256"])
                for item in source["experiments"].values()
            ),
        ):
            if Path(path).exists():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path


def test_decisions_record_the_sha256_of_every_study_yaml():
    recorded = dict(
        re.findall(
            r"^\| \[([^\]]+)\]\([^)]+\) \| `([0-9a-f]{64})` \|$",
            (STUDY / "decisions.md").read_text(),
            re.MULTILINE,
        )
    )
    actual = {
        path.relative_to(STUDY).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in STUDY.rglob("*.yaml")
    }
    assert recorded == actual
