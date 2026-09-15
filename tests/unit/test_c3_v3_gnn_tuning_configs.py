from copy import deepcopy

import pytest

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.relational_trace_graph_poc_config import validate_relational_trace_graph_poc_config


def test_mask10_preserves_fixed_v3_global_rms_and_no_time_shift():
    studies = REPOSITORY_ROOT / "studies"
    baseline = load_resolved_config(studies / "study_042_c3_v3_five_method_comparison/gnn.yaml")
    candidate = load_resolved_config(studies / "study_044_c3_v3_gnn_target_tuning/mask10_5k.yaml")
    expected = deepcopy(baseline)
    expected["training"]["inner_mask_fraction"] = 0.1
    expected["prediction"]["query_batch_size"] = 16
    assert candidate == expected
    settings = validate_relational_trace_graph_poc_config(candidate)
    assert settings.model["max_edge_time_shift_samples"] == 0
    assert settings.model["amplitude_mode"] == "train_global_rms"
    assert candidate["training"]["amplitude_scaling"] == "observed_volume_global_rms"


@pytest.mark.parametrize(
    "filename,section,key,value",
    [
        ("mask10_fourier16_5k.yaml", "model", "node_fourier_components", 16),
        ("mask10_bf16_5k.yaml", "training", "mixed_precision", "bf16"),
        ("mask10_spectral_5k.yaml", "model", "spectral_input_block", True),
    ],
)
def test_candidates_change_one_factor(filename, section, key, value):
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_5k.yaml")
    expected[section][key] = value
    candidate = load_resolved_config(study / filename)
    assert candidate == expected
    validate_relational_trace_graph_poc_config(candidate)


def test_spectral_larger_prediction_batch_changes_only_batch_size():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_spectral_5k.yaml")
    expected["prediction"]["query_batch_size"] = 64
    candidate = load_resolved_config(study / "mask10_spectral_q64_5k.yaml")
    assert candidate == expected
    validate_relational_trace_graph_poc_config(candidate)


def test_fourier_fast_candidate_keeps_learning_and_input_contract():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_5k.yaml")
    expected["training"]["cudnn_benchmark"] = False
    expected["prediction"]["query_batch_size"] = 64
    candidate = load_resolved_config(study / "mask10_fourier16_fast_5k.yaml")
    assert candidate == expected
    validate_relational_trace_graph_poc_config(candidate)


def test_direct_neighbors_candidate_preserves_fixed_input_and_training():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_fast_5k.yaml")
    expected["model"]["message_passing_rounds"] = 1
    expected["model"]["temporal_dilations"] = [1]
    expected["graph"]["neighbors_per_relation"] = 8
    candidate = load_resolved_config(study / "mask10_fourier16_direct8_5k.yaml")
    assert candidate == expected
    validate_relational_trace_graph_poc_config(candidate)


def test_longer_budget_candidate_changes_only_update_count():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_fast_5k.yaml")
    expected["training"]["max_steps"] = 20000
    candidate = load_resolved_config(study / "mask10_fourier16_fast_20k.yaml")
    assert candidate == expected
    validate_relational_trace_graph_poc_config(candidate)


def test_cosine_candidate_changes_only_schedule_within_budget_cap():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_fast_20k.yaml")
    expected["training"]["learning_rate_schedule"] = dict(
        kind="constant_then_cosine", hold_steps=10000, minimum_learning_rate=0.00003
    )
    candidate = load_resolved_config(study / "mask10_fourier16_cosine_20k.yaml")
    assert candidate == expected
    validated = validate_relational_trace_graph_poc_config(candidate)
    assert (
        validated.training["learning_rate_schedule"]
        == expected["training"]["learning_rate_schedule"]
    )
    for path in study.glob("mask*.yaml"):
        assert load_resolved_config(path)["training"]["max_steps"] <= 20000


def test_width128_candidate_changes_only_model_width():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_fast_20k.yaml")
    expected["model"]["width"] = 128
    candidate = load_resolved_config(study / "mask10_fourier16_width128_20k.yaml")
    assert candidate == expected
    settings = validate_relational_trace_graph_poc_config(candidate)
    assert settings.model["width"] == 128
    assert settings.model["max_edge_time_shift_samples"] == 0
    assert settings.model["amplitude_mode"] == "train_global_rms"
    assert settings.training["max_steps"] == 20000
    assert "learning_rate_schedule" not in settings.training


def test_mask05_candidate_changes_only_inner_mask_and_keeps_final_evaluation():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_width128_20k.yaml")
    expected["training"]["inner_mask_fraction"] = 0.05
    candidate = load_resolved_config(study / "mask05_fourier16_width128_20k.yaml")
    assert candidate == expected
    settings = validate_relational_trace_graph_poc_config(candidate)
    assert settings.training["max_steps"] == 20000
    assert settings.training["inner_mask_fraction"] == 0.05
    assert settings.model["width"] == 128
    assert candidate["evaluation"]["primary_metric"] == "physical_amplitude_mean_trace_snr_db"
    assert "validation" not in candidate
    assert "validation_interval" not in settings.training


def test_ema_candidate_changes_only_averaging_of_width128_mask10():
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_width128_20k.yaml")
    expected["training"]["ema_decay"] = 0.999
    candidate = load_resolved_config(study / "mask10_fourier16_width128_ema999_20k.yaml")
    assert candidate == expected
    assert validate_relational_trace_graph_poc_config(candidate).training["ema_decay"] == 0.999
    candidate["training"]["ema_decay"] = None
    assert validate_relational_trace_graph_poc_config(candidate).training["ema_decay"] is None


@pytest.mark.parametrize(
    "filename,section,key,value",
    [
        (
            "mask10_fourier16_width128_ema999_neighbors4_20k.yaml",
            "graph",
            "neighbors_per_relation",
            4,
        ),
        (
            "mask10_fourier16_width128_ema999_dilation14_20k.yaml",
            "model",
            "temporal_dilations",
            [1, 4],
        ),
    ],
)
def test_20db_candidates_preserve_adopted_ema_contract(filename, section, key, value):
    from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator

    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    expected = load_resolved_config(study / "mask10_fourier16_width128_ema999_20k.yaml")
    expected[section][key] = value
    candidate = load_resolved_config(study / filename)
    assert candidate == expected
    settings = validate_relational_trace_graph_poc_config(candidate)
    assert settings.training["max_steps"] == 20000
    assert settings.training["ema_decay"] == 0.999
    assert settings.model["message_passing_rounds"] == 2
    model = RelationalTraceGraphInterpolator(**settings.model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 363269


@pytest.mark.parametrize("value", [True, 0, 1, -1, "0.999", float("nan"), float("inf")])
def test_ema_config_rejects_invalid_decay(value):
    study = REPOSITORY_ROOT / "studies/study_044_c3_v3_gnn_target_tuning"
    config = load_resolved_config(study / "mask10_fourier16_width128_20k.yaml")
    config["training"]["ema_decay"] = value
    with pytest.raises(ValueError, match="ema_decay"):
        validate_relational_trace_graph_poc_config(config)
