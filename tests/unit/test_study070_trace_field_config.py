"""The committed study_070 specification is executable before any run exists.

One method, one shared 5x5 grid, two declared dataset differences, and a single
parameter count per dataset inside the matched band.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from seis_interp.c3_fair_grid_config import c3_experiment_methods, validate_c3_fair_grid_configs
from seis_interp.processing.c3_fair_grid_contract import make_c3_outer_mask
from seis_interp.processing.c3_region_search import (
    c3_search_candidates,
    resolve_c3_candidate,
    validate_c3_experiment_plan,
)
from seis_interp.processing.c3_trace_partition import partition_c3_traces
from seis_interp.processing.experiment_commonality import load_experiment_grid
from seis_interp.processing.forge_grid_partition import partition_forge_traces
from seis_interp.training.forge_mvp_methods import model_parameter_count

STUDY = Path("studies/study_070_forge_c3_trace_field")
TARGET = 363269
TOLERANCE = 0.05
METHOD = "inr_real"
# (native time count, aligned canvas, hidden width, parameter count)
DATASETS = {
    "c3": (384, 384, 249, 362679),
    "forge": (1001, 2172, 126, 364296),
}
NAMESPACES = {
    "c3": ("study_062_c3_region_grid", "study_063_c3_trace_grid"),
    "forge": ("study_052_forge_m1_mvp", "study_060_forge_fair_grid"),
}
FORBIDDEN_MODEL = ("coordinate_transform", "polar_offset", "profile_shape", "latent_channels")
FORBIDDEN_TRAINING = (
    "supervised_traces_per_update",
    "weight_decay_after",
    "nuclear_norm_weight",
    "coordinate_jitter_cells",
    "loss_support_mask",
)


def _load(dataset, name):
    return yaml.safe_load((STUDY / dataset / f"{name}.yaml").read_text())


@pytest.fixture(params=sorted(DATASETS))
def dataset(request):
    return request.param


def _candidates(dataset):
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    return c3_search_candidates(_load(dataset, METHOD), grid["methods"][METHOD])


def test_both_datasets_declare_only_the_trace_field(dataset):
    contract = _load(dataset, "config")
    assert c3_experiment_methods(contract) == (METHOD,)
    assert contract["study_id"] == "study_070_forge_c3_trace_field"
    assert contract["training_loss"] == "masked_trace_mse"


def test_the_shared_grid_is_pinned_by_its_own_hash(dataset):
    contract = _load(dataset, "config")
    digest = hashlib.sha256((STUDY / "grid.yaml").read_bytes()).hexdigest()
    assert contract["common_grid"] == {"path": "../grid.yaml", "sha256": digest}
    # Runs check_fixed_settings: dataset_differences.yaml must declare every difference.
    assert list(load_experiment_grid(STUDY / dataset, contract)["methods"]) == [METHOD]


def test_only_the_time_count_and_hidden_width_differ_between_datasets():
    declared = yaml.safe_load((STUDY / "dataset_differences.yaml").read_text())
    assert set(declared["methods"]) == {METHOD}
    assert set(declared["methods"][METHOD]) == {
        "model.time_sample_count",
        "model.hidden_width",
    }
    assert all(reason.strip() for reason in declared["methods"][METHOD].values())


def test_the_grid_expands_to_25_candidates_with_one_parameter_count(dataset):
    native, canvas, width, expected = DATASETS[dataset]
    candidates = _candidates(dataset)
    assert [item["id"] for item in candidates] == [
        f"a{a:02d}_b{b:02d}" for a in range(1, 6) for b in range(1, 6)
    ]
    counts = {model_parameter_count(item["config"], canvas) for item in candidates}
    assert counts == {expected}
    assert abs(expected / TARGET - 1) <= TOLERANCE
    assert _load(dataset, METHOD)["model"]["hidden_width"] == width
    assert _load(dataset, METHOD)["model"]["time_sample_count"] == native


def test_the_axes_are_the_declared_regularization_and_fourier_bands():
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    axes = grid["methods"][METHOD]
    assert [item["set"] for item in axes["axis_a"]] == [
        {"training.weight_decay": value} for value in (0.0, 0.001, 0.01, 0.03, 0.1)
    ]
    assert [item["set"] for item in axes["axis_b"]] == [
        {"model.frequency_base": value} for value in (1.05, 1.075, 1.1, 1.15, 1.25)
    ]


def test_every_candidate_keeps_the_matched_budget_and_the_declared_contract(dataset):
    contract = _load(dataset, "config")
    assert contract["matched_training"] == {
        "max_steps": 50000,
        "learning_rate": 0.001,
        "loss_traces_per_update": 64,
        "parameter_count_target": TARGET,
        "parameter_count_tolerance": TOLERANCE,
    }
    for candidate in _candidates(dataset):
        config = candidate["config"]
        training, model = config["training"], config["model"]
        assert config["method_id"] == METHOD and config["geometry_mode"] == "real"
        assert config["loss"] == contract["training_loss"]
        assert config["checkpoint_rule"] == "fixed_final_ema"
        assert config["model_seed"] == contract["model_seed"] == 7301
        assert training["optimizer"] == "adamw"
        assert training["max_steps"] == 50000 and training["learning_rate"] == 0.001
        assert training["batch_size"] == contract["matched_training"]["loss_traces_per_update"]
        assert training["learning_rate_schedule"] == "constant"
        assert training["mixed_precision"] == "off"
        assert training["ema_decay"] == 0.999
        assert training["ema_eval_decays"] == [0.999, 0.9995, 0.9998]
        assert model["input_features"] == 4 and model["fourier_components"] == 40
        assert model["hidden_layers"] == 4
        assert model["activation"] == "gelu" and model["output_activation"] == "linear"
        assert not [key for key in FORBIDDEN_MODEL if key in model]
        assert not [key for key in FORBIDDEN_TRAINING if key in training]
        assert "execution" not in config and "canvas_time_upsampling" not in config


def test_the_c3_contract_validates_the_resolved_method():
    contract = _load("c3", "config")
    grid = yaml.safe_load((STUDY / "grid.yaml").read_text())
    validate_c3_experiment_plan(contract, grid)
    observed = contract["observed_trace_count"]
    resolved = resolve_c3_candidate(_load("c3", METHOD), observed, 64)
    validate_c3_fair_grid_configs({METHOD: resolved}, contract)


def test_the_historical_mask_and_seed_namespaces_are_preserved(dataset):
    contract = _load(dataset, "config")
    mask_study, validation_study = NAMESPACES[dataset]
    assert contract["mask_study_id"] == mask_study
    assert contract["validation_study_id"] == validation_study
    assert contract["trace_split"]["evaluation_seeds"] == [7321, 7322, 7323]
    assert contract["validation_holdout"]["mask_seed"] == contract["trace_split"]["search_seed"]


def test_the_forge_contract_keeps_its_mandatory_reference_experiment():
    contract = _load("forge", "config")
    assert contract["reference_experiment"].startswith("runs/study_066_forge_c3_reference/")
    assert contract["preprocessing"]["alignment"] == {"velocity_m_per_s": 500.0}
    assert set(contract["boundary_handling"]) == {"natural_missing"}
    assert "import_from" not in contract


def test_the_admission_request_is_declared_once_for_the_method(dataset):
    cuda = _load(dataset, "config")["cuda_execution"]
    assert cuda["minimum_free_memory_gib"] == {METHOD: 4}
    assert "excluded_devices_while_pids_alive" not in cuda
    assert _load(dataset, "config")["method_execution"] == {METHOD: {"parallel_candidates": 4}}


def test_the_evaluation_shares_the_study067_bootstrap_identity():
    evaluation = yaml.safe_load((STUDY / "evaluation.yaml").read_text())
    reference = yaml.safe_load(
        Path("studies/study_067_forge_c3_common_grid/evaluation.yaml").read_text()
    )
    assert evaluation["bootstrap"] == reference["bootstrap"]
    assert evaluation["cluster_column"] == reference["cluster_column"] == "source_cell"
    assert evaluation["datasets"] == reference["datasets"]
    assert evaluation["metrics"] == reference["metrics"]
    assert evaluation["idw"] == reference["idw"]
    assert evaluation["left"]["method"] == METHOD
    assert evaluation["floor"] == {
        "domain": "s1_validation_best_of_25",
        "metric": "mean_trace_snr_db",
        "c3": 10.0,
        "forge": 0.4,
    }
    right = evaluation["right"]
    for path, digest in (
        (right["grid"], right["grid_sha256"]),
        (right["evaluation"], right["evaluation_sha256"]),
        (
            right["experiments"]["c3"]["path"] + "/experiment.json",
            right["experiments"]["c3"]["sha256"],
        ),
        (
            right["experiments"]["forge"]["path"] + "/experiment.json",
            right["experiments"]["forge"]["sha256"],
        ),
        (right["analysis"]["path"] + "/manifest.json", right["analysis"]["manifest_sha256"]),
        (right["analysis"]["path"] + "/c3/selection.csv", right["analysis"]["selection_c3_sha256"]),
        (
            right["analysis"]["path"] + "/forge/selection.csv",
            right["analysis"]["selection_forge_sha256"],
        ),
    ):
        referenced = Path(path)
        if not referenced.exists():
            pytest.skip(f"{path} is not present in this checkout")
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


REFERENCE_RUNS = {
    "c3": Path("runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces"),
    "forge": Path("runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge"),
}


def test_every_partition_reproduces_study067(dataset):
    """The contrast against study_067 is paired cell for cell, not merely comparable.

    Both builders read ``mask_study_id`` / ``validation_study_id``, so a new
    ``study_id`` keeps the historical assignment. ``runs/`` is not committed.
    """
    reference = REFERENCE_RUNS[dataset]
    if not reference.exists():
        pytest.skip("the study_067 production runs are not present in this checkout")
    contract = _load(dataset, "config")
    search = contract["trace_split"]["search_seed"]
    for seed in (search, *contract["trace_split"]["evaluation_seeds"]):
        sealed = pd.read_parquet(reference / f"input/seed_{seed}/partition.parquet")
        if dataset == "forge":
            derived = partition_forge_traces(
                pd.read_parquet(reference / "input/manifest.parquet"), contract, seed
            )
        else:
            outer = (
                pd.read_parquet(reference / "input/outer_mask.parquet")
                if seed == search
                else make_c3_outer_mask(
                    pd.read_parquet(reference / "input/trace_manifest.parquet"),
                    {**contract, "mask_seed": seed},
                )
            )
            derived = partition_c3_traces(outer, contract, seed)
        pd.testing.assert_frame_equal(derived, sealed)
