from pathlib import Path

import pytest
import yaml

from seis_interp.c3_fair_grid_config import C3_NEURAL_METHODS, validate_c3_fair_grid_configs
from seis_interp.processing.c3_region_search import c3_search_candidates, resolve_c3_candidate


@pytest.mark.parametrize("name", ("study_062_c3_region_grid", "study_063_c3_trace_grid"))
def test_all_search_candidates_keep_the_requested_training_contract(name):
    study = Path(__file__).resolve().parents[2] / "studies" / name
    contract = yaml.safe_load((study / "config.yaml").read_text())
    grid = yaml.safe_load((study / "grid.yaml").read_text())
    assert contract["matched_training"]["max_steps"] == 50000
    assert contract["matched_training"]["learning_rate"] == 1e-4
    assert contract["matched_training"]["loss_traces_per_update"] == 64
    configs = {}
    for method, axes in grid["methods"].items():
        base = yaml.safe_load((study / f"{method}.yaml").read_text())
        configs[method] = resolve_c3_candidate(base, contract["observed_trace_count"], 64)
        candidates = c3_search_candidates(base, axes)
        assert len(candidates) == 25
        if method not in C3_NEURAL_METHODS:
            continue
        for candidate in candidates:
            config = candidate["config"]
            training = config["training"]
            assert training["optimizer"] == "adamw"
            assert training["learning_rate"] == 1e-4
            assert training["learning_rate_schedule"] == "constant"
            assert training["max_steps"] == 50000
            assert training["weight_decay"] == 0
            assert training["ema_decay"] == 0.999
            assert config["checkpoint_rule"] == "fixed_final_ema"
            trace_key = (
                "query_batch_size" if method == "regsi_real" else "supervised_traces_per_update"
            )
            assert training[trace_key] == 64
    validate_c3_fair_grid_configs(configs, contract)


def test_trace_study_matches_forge_observed_holdout_fraction_and_preserves_c3_outer_mask():
    root = Path(__file__).resolve().parents[2] / "studies"
    forge = yaml.safe_load((root / "study_060_forge_fair_grid/config.yaml").read_text())
    c3 = yaml.safe_load((root / "study_063_c3_trace_grid/config.yaml").read_text())
    previous = yaml.safe_load((root / "study_062_c3_region_grid/config.yaml").read_text())
    assert c3["validation_holdout"]["fraction"] == forge["validation_holdout"]["fraction"] == 0.1
    assert "region_split" not in c3
    assert c3["trace_split"]["evaluation_seeds"] == [7311, 7312, 7313]
    for key in (
        "mask_id",
        "mask_study_id",
        "mask_seed",
        "observed_trace_count",
        "test_trace_count",
    ):
        assert c3[key] == previous[key]
    validation = round(c3["validation_holdout"]["fraction"] * c3["observed_trace_count"])
    assert validation == 2621 and c3["observed_trace_count"] - validation == 23593
    for name in ("grid", "regsi_real", "ccnet5d_grid", "nersi_profile", "pocs_grid", "drr_grid"):
        assert (root / f"study_063_c3_trace_grid/{name}.yaml").read_bytes() == (
            root / f"study_062_c3_region_grid/{name}.yaml"
        ).read_bytes()
