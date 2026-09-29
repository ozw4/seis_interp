from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from seis_interp.forge_mvp_config import validate_mvp_configs
from seis_interp.processing.forge_mvp_contract import METHODS


def _study_configs(name="study_056_forge_2ms_random80_trace_rms"):
    study = Path(__file__).resolve().parents[2] / "studies" / name
    return (
        yaml.safe_load((study / "config.yaml").read_text()),
        {method: yaml.safe_load((study / f"{method}.yaml").read_text()) for method in METHODS},
    )


@pytest.mark.parametrize(
    "study,fraction,observed,test",
    [
        ("study_056_forge_2ms_random80_trace_rms", 0.8, 9830, 39321),
        ("study_057_forge_2ms_random70_trace_rms", 0.7, 14745, 34406),
    ],
)
def test_six_method_comparison_fixes_requested_sampling_mask_scaling_and_update_budget(
    study, fraction, observed, test
):
    contract, configs = _study_configs(study)
    validate_mvp_configs(configs, contract)
    assert contract["sample_interval_us"] == 2000
    assert contract["time_sample_count"] == 2001
    assert contract["missing_fraction"] == fraction
    assert contract["observed_trace_count"] == observed
    assert contract["test_trace_count"] == test
    assert test == round(contract["eligible_trace_count"] * fraction)
    assert observed + test == contract["eligible_trace_count"]
    assert contract["mask_study_id"] == "study_052_forge_m1_mvp"
    assert contract["preprocessing"]["amplitude_scaling"] == "trace_oracle"
    for method in ("regsi_real", "regsi_grid", "nersi_real", "ccnet5d_grid"):
        assert configs[method]["training"]["max_steps"] == 10000
        assert configs[method]["model_seed"] == 7301
    assert configs["regsi_real"]["training"]["query_batch_size"] == 32
    assert configs["nersi_real"]["model"]["time_sample_count"] == 2001
    assert configs["ccnet5d_grid"]["patches"]["shape"][0] == 2001
    assert configs["pocs_grid"]["algorithm"]["n_iterations"] == 100
    assert configs["drr_grid"]["algorithm"]["n_iterations"] == 10


def test_random70_changes_only_missing_fraction_and_preserves_all_method_settings():
    original, configs80 = _study_configs()
    changed, configs70 = _study_configs("study_057_forge_2ms_random70_trace_rms")
    assert configs70 == configs80
    for key in ("study_id", "missing_fraction", "test_trace_count", "observed_trace_count"):
        del original[key]
        del changed[key]
    assert changed == original
    root = Path(__file__).resolve().parents[2] / "studies"
    inputs80 = yaml.safe_load(
        (root / "study_056_forge_2ms_random80_trace_rms/inputs.yaml").read_text()
    )
    inputs70 = yaml.safe_load(
        (root / "study_057_forge_2ms_random70_trace_rms/inputs.yaml").read_text()
    )
    assert inputs70 == {
        key: value
        for key, value in inputs80.items()
        if key not in ("outer_mask", "outer_mask_sha256")
    }


@pytest.mark.parametrize("fault", ["interval", "trace_rms", "idw", "nersi_length", "ccnet_length"])
def test_comparison_rejects_inconsistent_sampling_or_normalization(fault):
    original, configs = _study_configs()
    contract = deepcopy(original)
    if fault == "interval":
        contract["sample_interval_us"] = 1000
    elif fault == "trace_rms":
        contract["preprocessing"]["trace_rms"] = False
    elif fault == "idw":
        contract["preprocessing"]["idw"] = {}
    elif fault == "nersi_length":
        configs["nersi_real"]["model"]["time_sample_count"] = 4001
    elif fault == "ccnet_length":
        configs["ccnet5d_grid"]["patches"]["shape"][0] = 4001
    with pytest.raises(ValueError):
        validate_mvp_configs(configs, contract)


MATCHED_STUDY = "study_058_forge_2ms_random80_align500_matched20k"


def test_matched_comparison_fixes_alignment_budget_batches_and_study_056_mask():
    contract, configs = _study_configs(MATCHED_STUDY)
    validate_mvp_configs(configs, contract)
    reference, _ = _study_configs()
    assert contract["preprocessing"]["alignment"] == {"velocity_m_per_s": 500.0}
    assert contract["matched_training"] == {
        "max_steps": 20000,
        "learning_rate": 0.001,
        "loss_traces_per_update": 64,
        "parameter_count_target": 363269,
        "parameter_count_tolerance": 0.05,
    }
    for key in ("mask_study_id", "mask_id", "mask_seed", "test_trace_count", "model_seed"):
        assert contract[key] == reference[key]
    assert contract["preprocessing"]["amplitude_scaling"] == "trace_oracle"
    for method in ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid"):
        training = configs[method]["training"]
        assert training["max_steps"] == 20000
        assert training["learning_rate"] == 0.001
    # 64 supervised traces per update, following the C3 NA studies 043/044/046.
    assert configs["nersi_real"]["training"]["batch_size"] == 64
    assert configs["ccnet5d_grid"]["training"]["supervised_traces_per_update"] == 64
    for method in ("regsi_real", "regsi_grid"):
        training = configs[method]["training"]
        assert training["query_batch_size"] == 64
        assert "query_microbatch_size" not in training
        assert round(contract["observed_trace_count"] * training["inner_mask_fraction"]) >= 64
    assert all("alignment" not in config for config in configs.values())
    root = Path(__file__).resolve().parents[2] / "studies"
    assert yaml.safe_load((root / MATCHED_STUDY / "inputs.yaml").read_text()) == yaml.safe_load(
        (root / "study_056_forge_2ms_random80_trace_rms/inputs.yaml").read_text()
    )


@pytest.mark.parametrize(
    "method,section,key,value",
    [
        ("ccnet5d_grid", "training", "max_steps", 10000),
        ("nersi_real", "training", "learning_rate", 0.0003),
        ("nersi_real", "training", "batch_size", 1),
        ("regsi_grid", "training", "query_batch_size", 32),
        ("ccnet5d_grid", "training", "supervised_traces_per_update", 16),
        ("ccnet5d_grid", "training", "supervised_traces_per_update", None),
        ("regsi_grid", "training", "inner_mask_fraction", 0.003),
        ("pocs_grid", None, "alignment", {"velocity_m_per_s": 500.0}),
    ],
)
def test_matched_comparison_rejects_budget_batch_or_method_alignment_changes(
    method, section, key, value
):
    contract, configs = _study_configs(MATCHED_STUDY)
    changed = deepcopy(configs)
    (changed[method][section] if section else changed[method])[key] = value
    if method == "regsi_grid":
        changed["regsi_real"]["training"][key] = value
    with pytest.raises(ValueError):
        validate_mvp_configs(changed, contract)


@pytest.mark.parametrize(
    "alignment",
    [
        {"velocity_m_per_s": 0.0},
        {"velocity": 500.0},
        {"velocity_m_per_s": 500.0, "max_delay_samples": 1},
    ],
)
def test_matched_comparison_rejects_invalid_study_alignment(alignment):
    contract, configs = _study_configs(MATCHED_STUDY)
    contract["preprocessing"]["alignment"] = alignment
    with pytest.raises(ValueError):
        validate_mvp_configs(configs, contract)
