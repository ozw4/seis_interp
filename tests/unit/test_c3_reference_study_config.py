from pathlib import Path

import yaml

from seis_interp.c3_fair_grid_config import c3_experiment_methods, validate_c3_fair_grid_configs
from seis_interp.processing.c3_region_search import (
    c3_reference_configuration,
    c3_search_candidates,
    resolve_c3_candidate,
    validate_c3_experiment_plan,
)


def test_three_25_candidate_grids_share_mse_training_and_include_reference_settings():
    study = Path(__file__).resolve().parents[2] / "studies/study_064_c3_reference_grid"
    contract = yaml.safe_load((study / "config.yaml").read_text())
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    assert c3_experiment_methods(contract) == ("regsi_real", "ccnet5d_grid", "nersi_profile")
    assert contract["training_loss"] == "masked_trace_mse"
    assert contract["matched_training"]["learning_rate"] == 0.001
    validate_c3_experiment_plan(contract, grid)
    configs = {}
    for method in c3_experiment_methods(contract):
        base = yaml.safe_load((study / f"{method}.yaml").read_text())
        candidates = c3_search_candidates(base, grid["methods"][method])
        reference = c3_reference_configuration(candidates, contract["reference_settings"][method])
        assert (
            reference
            == {"regsi_real": "a01_b05", "ccnet5d_grid": "a01_b01", "nersi_profile": "a01_b04"}[
                method
            ]
        )
        assert len(candidates) == 25
        required_memory = contract["cuda_execution"]["minimum_free_memory_gib"][method]
        if isinstance(required_memory, dict):
            assert set(required_memory) == {candidate["id"] for candidate in candidates}
            assert all(value > 0 for value in required_memory.values())
        configs[method] = resolve_c3_candidate(base, 23593, 64)
        for candidate in candidates:
            config = candidate["config"]
            assert config["training"]["max_steps"] == 50000
            assert config["training"]["ema_decay"] == 0.999
            assert config["training"]["weight_decay"] == 0
            assert config["training"]["optimizer"] == "adamw"
            assert (
                config["training"]["learning_rate"] == contract["matched_training"]["learning_rate"]
            )
            assert config["loss"] == contract["training_loss"]
            assert config["checkpoint_rule"] == "fixed_final_ema"
            if method == "regsi_real":
                assert config["model"]["attention_pooling"] == "rms"
                assert config["geometry_features"]["position_scale_m"] == 500
                assert config["geometry_features"]["offset_scale_m"] == 500
                assert config["training"]["query_batch_size"] == 64
            elif method == "nersi_profile":
                assert config["training"]["supervised_traces_per_update"] == 64
                assert config["model"]["fourier_components"] == 40
                assert config["model"]["encoder_width"] == 40
                assert config["model"]["latent_channels"] == 28
                assert config["model"]["decoder_channels"] == [29, 14, 8]
            else:
                assert config["training"]["supervised_traces_per_update"] == 64
                assert config["model"]["hidden_channels"] == 53
                assert config["model"]["intermediate_channels"] == 63
                assert config["patches"]["inner_mask_fraction"] == 0.1
    validate_c3_fair_grid_configs(configs, {**contract, "observed_trace_count": 23593})
