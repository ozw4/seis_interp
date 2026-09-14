from copy import deepcopy
from pathlib import Path

from seis_interp.ccnet5d_poc_config import validate_ccnet5d_poc_config
from seis_interp.configuration import load_resolved_config


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
