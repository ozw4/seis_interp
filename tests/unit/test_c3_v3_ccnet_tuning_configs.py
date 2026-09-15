from copy import deepcopy
from pathlib import Path

from seis_interp.ccnet5d_poc_config import validate_ccnet5d_poc_config
from seis_interp.configuration import load_resolved_config
from seis_interp.models.ccnet5d import CCNet5D


def test_parameter_budget_candidate_changes_only_channels():
    study = Path("studies/study_043_c3_v3_ccnet_target_tuning")
    expected = load_resolved_config(study / "mask10_ema0999_adamw_trace64_50k.yaml")
    expected["model"].update(hidden_channels=53, intermediate_channels=63)
    candidate = load_resolved_config(study / "mask10_ema0999_adamw_trace64_param363k_50k.yaml")
    assert candidate == expected
    settings = validate_ccnet5d_poc_config(candidate)
    model = CCNet5D(**settings.model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363292
    assert settings.training.max_steps == 50000
    assert settings.training.supervised_traces_per_update == 64
    assert settings.training.ema_decay == 0.999


def test_adamw_trace64_candidate_keeps_ema_50k_input_and_model():
    study = Path("studies/study_043_c3_v3_ccnet_target_tuning")
    expected = load_resolved_config(study / "mask10_ema0999_50k.yaml")
    expected["training"].update(
        optimizer="adamw", weight_decay=0.0, supervised_traces_per_update=64
    )
    candidate = load_resolved_config(study / "mask10_ema0999_adamw_trace64_50k.yaml")
    assert candidate == expected
    settings = validate_ccnet5d_poc_config(candidate)
    assert settings.training.supervised_traces_per_update == 64
    assert settings.training.max_steps == 50000


def test_mask10_changes_only_inner_mask():
    studies = Path("studies")
    baseline = load_resolved_config(studies / "study_042_c3_v3_five_method_comparison/ccnet5d.yaml")
    actual = load_resolved_config(studies / "study_043_c3_v3_ccnet_target_tuning/mask10_5k.yaml")
    expected = deepcopy(baseline)
    expected["patches"]["inner_mask_fraction"] = 0.1
    assert actual == expected
    validate_ccnet5d_poc_config(actual)
    assert actual["training"]["amplitude_scaling"] == "observed_volume_global_rms"
    assert "time_alignment" not in actual


def test_long_budget_preserves_model_and_inputs():
    study = Path("studies/study_043_c3_v3_ccnet_target_tuning")
    expected = load_resolved_config(study / "mask10_5k.yaml")
    expected["training"].update(max_steps=20000, report_interval=500)
    actual = load_resolved_config(study / "mask10_20k.yaml")
    assert actual == expected
    validate_ccnet5d_poc_config(actual)


def test_ema_changes_only_weight_averaging():
    study = Path("studies/study_043_c3_v3_ccnet_target_tuning")
    expected = load_resolved_config(study / "mask10_20k.yaml")
    expected["training"]["ema_decay"] = 0.999
    actual = load_resolved_config(study / "mask10_ema0999_20k.yaml")
    assert actual == expected
    assert validate_ccnet5d_poc_config(actual).training.ema_decay == 0.999


def test_ema_50k_changes_only_update_budget():
    study = Path("studies/study_043_c3_v3_ccnet_target_tuning")
    expected = load_resolved_config(study / "mask10_ema0999_20k.yaml")
    expected["training"]["max_steps"] = 50000
    actual = load_resolved_config(study / "mask10_ema0999_50k.yaml")
    assert actual == expected
    assert validate_ccnet5d_poc_config(actual).training.max_steps == 50000
