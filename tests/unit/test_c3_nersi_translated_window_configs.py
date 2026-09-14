import json
from copy import deepcopy

import pytest

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.nersi_config import validate_nersi_poc_config


def test_v3_conditions_bind_selected_window_and_distinguish_reference_normalization():
    study = REPOSITORY_ROOT / "studies/study_041_c3_nersi_translated_window"
    frozen = json.loads((study / "v3_conditions.lock.json").read_text())
    config = load_resolved_config(study / "shot18_ry18.yaml")
    assert frozen["condition_id"] == "c3_random80_v3"
    assert frozen["status"] == "fixed"
    assert frozen["inputs_lock"]["selection"] == config["benchmark_volume"]["selection"]
    assert frozen["inputs_lock"]["shape"] == [384, 16, 32, 8, 32]
    assert frozen["inputs_lock"]["mask"]["random_seed"] == 42
    assert frozen["inputs_lock"]["observed_trace_count"] == 26322
    assert frozen["inputs_lock"]["target_trace_count"] == 104750
    assert frozen["primary_metric"] == "mean_trace_snr_db"
    assert frozen["evaluation_amplitude_domain"] == "physical"
    assert frozen["normalization_is_part_of_fixed_scope"] is False
    assert frozen["reference_run_normalization"] == {
        "type": "global_rms",
        "source": "O_only",
        "scale": 9.27911442626805,
    }
    assert config["training"]["amplitude_scaling"] == "observed_volume_global_rms"
    assert "trace_rms_idw" not in config


@pytest.mark.parametrize(
    "name,shot,ry", [("shot18_ry18", 18, 18), ("shot10_ry34", 10, 34), ("shot27_ry00", 27, 0)]
)
def test_window_configs_only_change_input_contract_and_evaluation(name, shot, ry):
    studies = REPOSITORY_ROOT / "studies"
    baseline = load_resolved_config(
        studies / "study_040_c3_nersi_target_tuning/aligned_ema0999.yaml"
    )
    expected = deepcopy(baseline)
    expected["input_protocol"] = "c3_random80_window"
    expected["benchmark_case"]["id"] = f"c3_window_{name}_random80_seed42"
    expected["benchmark_volume"]["id"] = expected["benchmark_case"]["id"] + "_volume"
    expected["benchmark_volume"]["selection"]["shot_in_line"] = [shot, shot + 32]
    expected["benchmark_volume"]["selection"]["relative_receiver_y"] = [ry, ry + 32]
    expected["evaluation"]["primary_metric"] = "physical_amplitude_mean_trace_snr_db"
    actual = load_resolved_config(studies / f"study_041_c3_nersi_translated_window/{name}.yaml")
    assert actual == expected
    validate_nersi_poc_config(actual)
