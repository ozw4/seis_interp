from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from seis_interp.ccnet5d_poc_config import validate_ccnet5d_poc_config
from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.data.c3_poc_inputs import (
    C3_RANDOM80_MISSING_FRACTION,
    C3_RANDOM80_POC_BENCHMARK_ID,
    C3_RANDOM80_POC_DATASET_ID,
    C3_RANDOM80_POC_SELECTION,
)
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.data.c3_volume_run_inputs import validated_c3_volume_selection
from seis_interp.evaluation.c3_volume_metrics import validate_c3_volume_evaluation_config
from seis_interp.nersi_config import validate_nersi_poc_config
from seis_interp.pipelines.interpolate_drr import _drr_settings
from seis_interp.pipelines.interpolate_pocs import _pocs_settings
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config
from seis_interp.training.ccnet5d_observed_patches import CCNet5DObservedPatchSource

STUDY_DIRECTORY = REPOSITORY_ROOT / "studies" / "study_036_c3_random80_observed_only_poc"
METHODS = ("pocs", "drr", "nersi", "ccnet5d", "gnn")
CASE_ID = "c3_benchmark_test_random_trace_80_seed42"
VOLUME_ID = f"{CASE_ID}_volume"


@pytest.mark.parametrize("directory", ["methods", "smoke", "formal"])
@pytest.mark.parametrize(
    ("method", "validate"),
    [
        ("pocs", _pocs_settings),
        ("drr", _drr_settings),
        ("nersi", validate_nersi_poc_config),
        ("ccnet5d", validate_ccnet5d_poc_config),
        ("gnn", validate_relational_trace_graph_poc_config),
    ],
)
def test_each_config_passes_native_validation_and_preserves_fixed_inputs(
    directory, method, validate
):
    config = load_resolved_config(STUDY_DIRECTORY / directory / f"{method}.yaml")

    assert validate(config) is not None
    validate_c3_volume_evaluation_config(config)
    assert config["project"] == {"random_seed": 42}
    assert config["data"] == {"dataset_id": C3_RANDOM80_POC_DATASET_ID}
    assert config["benchmark_case"] == {"id": CASE_ID}
    assert config["benchmark_volume"] == {
        "id": VOLUME_ID,
        "selection": C3_RANDOM80_POC_SELECTION,
    }
    assert (
        validated_c3_volume_selection(config["benchmark_volume"]["selection"])
        == C3_RANDOM80_POC_SELECTION
    )
    assert config["interpolation_mask"] == {
        "partition": "test",
        "kind": "random_trace",
        "missing_fraction": C3_RANDOM80_MISSING_FRACTION,
    }
    assert config["evaluation"] == {
        "primary_metric": "physical_amplitude_global_snr_db",
        "domain": "evaluation_target",
    }


@pytest.mark.parametrize("method", METHODS)
def test_smoke_changes_only_iteration_budget_and_neural_report_interval(method):
    formal = load_resolved_config(STUDY_DIRECTORY / "methods" / f"{method}.yaml")
    smoke = load_resolved_config(STUDY_DIRECTORY / "smoke" / f"{method}.yaml")
    expected = deepcopy(formal)

    if method in ("pocs", "drr"):
        assert formal[method]["n_iterations"] == {"pocs": 100, "drr": 10}[method]
        assert "model" not in formal
        assert "training" not in formal
        expected[method]["n_iterations"] = {"pocs": 2, "drr": 1}[method]
    else:
        assert formal["training"]["max_steps"] == 5000
        assert formal["training"]["report_interval"] == 100
        expected["training"].update(max_steps=2, report_interval=1)

    assert smoke == expected


@pytest.mark.parametrize("directory", ["methods", "smoke", "formal"])
@pytest.mark.parametrize("method", ["nersi", "ccnet5d", "gnn"])
def test_neural_configs_declare_distinct_initialization_and_sampling_seeds(directory, method):
    config = load_resolved_config(STUDY_DIRECTORY / directory / f"{method}.yaml")
    training = config["training"]

    assert training["model_initialization_seed"] == 101
    assert training["loss"] == "masked_trace_relative_mse"
    assert training["amplitude_scaling"] == "observed_volume_global_rms"
    assert "random_seed" not in training
    if method == "ccnet5d":
        assert config["patches"]["placement_seed"] == 301
        assert config["patches"]["inner_mask_seed"] == 401
        assert "random_seed" not in config["patches"]
    else:
        key = "sampling_seed" if method == "nersi" else "episode_seed"
        assert training[key] == 201


def test_input_contract_uses_shared_benchmark_id_and_five_portable_artifact_paths():
    inputs = load_resolved_config(STUDY_DIRECTORY / "inputs.yaml")
    suite = "data/processed/c3_na/study_029_c3_amplitude_qc"
    expected_paths = {
        "interim": "data/interim/c3_na/all_ffids",
        "processed": f"{suite}/partition",
        "mask": f"{suite}/masks/{CASE_ID}",
        "case": f"{suite}/cases/{CASE_ID}",
        "volume": f"{suite}/volumes/{CASE_ID}",
    }

    assert inputs["benchmark_id"] == C3_RANDOM80_POC_BENCHMARK_ID
    assert set(inputs["paths"]) == set(expected_paths)
    for name, expected in expected_paths.items():
        path = Path(inputs["paths"][name])
        assert not path.is_absolute()
        assert (STUDY_DIRECTORY / path).resolve() == REPOSITORY_ROOT / expected
    assert inputs["outputs"]["existing_directory_policy"] == "reject"
    output_path = Path(inputs["outputs"]["runs"])
    assert not output_path.is_absolute()
    assert (STUDY_DIRECTORY / output_path).resolve() == (
        REPOSITORY_ROOT / "runs" / STUDY_DIRECTORY.name
    )


@pytest.mark.parametrize("directory", ["methods", "smoke", "formal"])
def test_ccnet_patch_source_accepts_config_with_complete_benchmark_time_axis(directory):
    config = load_resolved_config(STUDY_DIRECTORY / directory / "ccnet5d.yaml")
    settings = validate_ccnet5d_poc_config(config)
    selection = config["benchmark_volume"]["selection"]
    time_length = selection["time"][1] - selection["time"][0]
    spatial_shape = settings.patches.shape[1:]
    observed = np.zeros(spatial_shape, dtype=bool)
    observed.reshape(-1)[::5] = True
    volume = ObservedC3Volume(
        values=np.broadcast_to(observed, (time_length, *spatial_shape)).astype(np.float32),
        time_s=np.arange(time_length, dtype=np.float32) * 0.004,
        array_rows=np.arange(observed.size).reshape(spatial_shape),
        observed_trace_mask=observed,
        evaluation_target_trace_mask=~observed,
    )
    source = CCNet5DObservedPatchSource(
        volume,
        amplitude_scale=1.0,
        patch_shape=settings.patches.shape,
        inner_mask_fraction=settings.patches.inner_mask_fraction,
        placement_seed=settings.patches.placement_seed,
        inner_mask_seed=settings.patches.inner_mask_seed,
    )
    patch = source.sample()
    assert patch.model_input.shape == settings.patches.shape
    assert patch.patch_slices[0] == slice(0, time_length)
    assert patch.visible_observed_mask.any() and patch.pseudo_target_mask.any()
    assert not (patch.visible_observed_mask & patch.pseudo_target_mask).any()


@pytest.mark.parametrize("method", METHODS)
def test_formal_config_is_standalone_and_differs_from_smoke_only_in_budget(tmp_path, method):
    source = STUDY_DIRECTORY / "formal" / f"{method}.yaml"
    isolated = tmp_path / source.name
    isolated.write_bytes(source.read_bytes())
    formal = load_resolved_config(isolated)
    assert formal == load_resolved_config(source)
    expected_smoke = deepcopy(formal)
    if method in ("pocs", "drr"):
        assert formal[method]["n_iterations"] == {"pocs": 100, "drr": 10}[method]
        expected_smoke[method]["n_iterations"] = {"pocs": 2, "drr": 1}[method]
    else:
        assert formal["training"]["max_steps"] == 5000
        assert formal["training"]["report_interval"] == 100
        assert formal["training"]["device"] == "cuda:1"
        expected_smoke["training"].update(max_steps=2, report_interval=1)
    assert expected_smoke == load_resolved_config(STUDY_DIRECTORY / "smoke" / source.name)
