"""The committed study_071 specification is executable before any run exists.

One conditioned method, one shared 5x5 grid whose first axis is the amount of
observed data a query reads, and a parameter count that no longer depends on the
time axis, so the only declared dataset difference is the inert declared count.
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
from seis_interp.training.forge_mvp_methods import model_parameter_count

STUDY = Path("studies/study_071_forge_c3_context_field")
TARGET = 363269
METHOD = "cinr_real"
PARAMETERS = 362642
NEIGHBORS = [1, 2, 4, 8, 16]
BASES = [1.05, 1.075, 1.1, 1.15, 1.25]
CANVAS = {"c3": (384, 384), "forge": (1001, 2172)}
# study_067's C-0 rule over the measured peak reserved memory.
ADMISSION = {"c3": [3, 3, 3, 3, 4], "forge": [3, 4, 5, 7, 11]}


def _load(dataset, name):
    return yaml.safe_load((STUDY / dataset / f"{name}.yaml").read_text())


@pytest.fixture(params=sorted(CANVAS))
def dataset(request):
    return request.param


def _candidates(dataset):
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    return c3_search_candidates(_load(dataset, METHOD), grid["methods"][METHOD])


def test_both_datasets_declare_only_the_conditioned_field(dataset):
    contract = _load(dataset, "config")
    assert c3_experiment_methods(contract) == (METHOD,)
    assert contract["study_id"] == "study_071_forge_c3_context_field"
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
    assert declared["methods"][METHOD]["model.time_sample_count"].strip()


def test_the_axes_are_the_context_size_and_the_shared_fourier_band():
    axes = yaml.safe_load((STUDY / "grid.yaml").read_text())["methods"][METHOD]
    assert [item["set"] for item in axes["axis_a"]] == [
        {"context.neighbor_count": value} for value in NEIGHBORS
    ]
    assert [item["set"] for item in axes["axis_b"]] == [
        {"model.frequency_base": value} for value in BASES
    ]


def test_the_grid_expands_to_25_candidates_with_one_canvas_independent_count(dataset):
    native, canvas = CANVAS[dataset]
    candidates = _candidates(dataset)
    assert [item["id"] for item in candidates] == [
        f"a{a:02d}_b{b:02d}" for a in range(1, 6) for b in range(1, 6)
    ]
    counts = {model_parameter_count(item["config"], canvas) for item in candidates}
    assert counts == {PARAMETERS}
    assert abs(PARAMETERS / TARGET - 1) <= 0.05
    model = _load(dataset, METHOD)["model"]
    assert model["time_sample_count"] == native
    assert model["width"] == 120 and model["time_downsample_factor"] == 4
    assert canvas % model["time_downsample_factor"] == 0


def test_every_candidate_keeps_the_matched_budget_and_reads_its_declared_context(dataset):
    contract = _load(dataset, "config")
    assert contract["matched_training"]["loss_traces_per_update"] == 64
    seen = set()
    for candidate in _candidates(dataset):
        config = candidate["config"]
        training, model = config["training"], config["model"]
        assert config["method_id"] == METHOD and config["geometry_mode"] == "real"
        assert config["loss"] == contract["training_loss"]
        assert config["checkpoint_rule"] == "fixed_final_ema"
        assert config["model_seed"] == contract["model_seed"] == 7301
        assert training["optimizer"] == "adamw" and training["max_steps"] == 50000
        assert training["batch_size"] == contract["matched_training"]["loss_traces_per_update"]
        assert training["ema_eval_decays"] == [0.999, 0.9995, 0.9998]
        assert model["input_features"] == 4 and model["fourier_components"] == 40
        assert set(config["context"]) == {"neighbor_count"}
        seen.add(config["context"]["neighbor_count"])
        assert "hidden_width" not in model and "hidden_layers" not in model
    assert sorted(seen) == NEIGHBORS


def test_the_admission_request_follows_the_measured_memory(dataset):
    declared = _load(dataset, "config")["cuda_execution"]["minimum_free_memory_gib"][METHOD]
    assert len(declared) == 25
    for index, expected in enumerate(ADMISSION[dataset], start=1):
        for axis_b in range(1, 6):
            assert declared[f"a{index:02d}_b{axis_b:02d}"] == expected
    assert _load(dataset, "config")["method_execution"] == {METHOD: {"parallel_candidates": 4}}


def test_the_c3_contract_validates_the_resolved_method():
    contract = _load("c3", "config")
    validate_c3_experiment_plan(contract, yaml.safe_load((STUDY / "grid.yaml").read_text()))
    resolved = resolve_c3_candidate(_load("c3", METHOD), contract["observed_trace_count"], 64)
    validate_c3_fair_grid_configs({METHOD: resolved}, contract)


def test_the_historical_namespaces_match_study_070(dataset):
    contract = _load(dataset, "config")
    previous = yaml.safe_load(
        (Path("studies/study_070_forge_c3_trace_field") / dataset / "config.yaml").read_text()
    )
    for key in (
        "mask_study_id",
        "validation_study_id",
        "trace_split",
        "validation_holdout",
        "mask_id",
        "mask_seed",
        "preprocessing",
        "selection",
        "model_seed",
    ):
        assert contract[key] == previous[key], key


def test_the_evaluation_isolates_the_conditioning_and_pins_both_references():
    evaluation = yaml.safe_load((STUDY / "evaluation.yaml").read_text())
    families = evaluation["contrasts"]["holm_families"]
    assert families["conditioning_snr"] == {
        "contrast": "cinr_real_minus_inr_real",
        "metric": "snr_db",
    }
    assert families["conditioning_relmse"]["contrast"] == "cinr_real_minus_inr_real"
    assert evaluation["left"]["method"] == METHOD
    assert evaluation["floor"]["forge"] == 0.4 and evaluation["floor"]["c3"] == 10.0
    sources = evaluation["right"]["sources"]
    assert set(sources) == {"study_067", "study_070"}
    for source in sources.values():
        for path, digest in (
            (source["grid"], source["grid_sha256"]),
            (source["evaluation"], source["evaluation_sha256"]),
            *(
                (item["path"] + "/experiment.json", item["sha256"])
                for item in source["experiments"].values()
            ),
        ):
            referenced = Path(path)
            if not referenced.exists():
                continue
            assert hashlib.sha256(referenced.read_bytes()).hexdigest() == digest, path


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
