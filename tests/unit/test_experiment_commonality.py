import shutil
from pathlib import Path

import pytest
import yaml

from seis_interp.processing.experiment_commonality import check_fixed_settings, load_experiment_grid


def test_only_declared_leaf_differences_with_reasons_are_permitted():
    first = {"patch": [384, 4, 8], "lr": 0.001}
    second = {"patch": [1001, 4, 8], "lr": 0.001}
    check_fixed_settings(first, second, {"patch.0": "dataset time length"}, "CCNet")
    second["patch"][1] = 8
    with pytest.raises(ValueError, match="patch.1"):
        check_fixed_settings(first, second, {"patch.0": "time"}, "CCNet")
    with pytest.raises(ValueError, match="reason"):
        check_fixed_settings(first, first, {"patch.0": ""}, "CCNet")


def test_study067_shared_grid_is_pinned_and_fixed_training_matches(tmp_path):
    source = Path(__file__).resolve().parents[2] / "studies/study_067_forge_c3_common_grid"
    shutil.copytree(source, tmp_path / "study")
    study = tmp_path / "study/c3"
    contract = yaml.safe_load((study / "config.yaml").read_text())
    assert len(load_experiment_grid(study, contract)["methods"]) == 5
    method = tmp_path / "study/forge/nersi_profile.yaml"
    config = yaml.safe_load(method.read_text())
    assert config["training"]["coordinate_jitter_cells"] == 0.5
    config["training"]["learning_rate"] = 0.01
    method.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="training.learning_rate"):
        load_experiment_grid(study, contract)
    method.write_bytes((source / "forge/nersi_profile.yaml").read_bytes())
    (tmp_path / "study/grid.yaml").write_text("methods: {}\n")
    with pytest.raises(ValueError, match="grid hash"):
        load_experiment_grid(study, contract)


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_study067_regsi_p2_v3_preserves_grid_budget_and_supersedes_old_reference(dataset):
    from seis_interp.processing.c3_region_search import (
        c3_reference_configuration,
        c3_search_candidates,
    )
    from seis_interp.training.forge_mvp_methods import model_parameter_count

    root = Path(__file__).resolve().parents[2]
    study = root / "studies/study_067_forge_c3_common_grid" / dataset
    contract = yaml.safe_load((study / "config.yaml").read_text())
    grid = load_experiment_grid(study, contract)
    base = yaml.safe_load((study / "regsi_real.yaml").read_text())
    candidates = c3_search_candidates(base, grid["methods"]["regsi_real"])
    assert len(candidates) == 25
    for candidate in candidates:
        assert candidate["config"]["training"]["inner_mask_fraction"] == 0.25
        assert candidate["config"]["model"]["node_fourier_components"] == 0
    counts = {
        model_parameter_count(candidate["config"], contract["time_sample_count"])
        for candidate in candidates
    }
    assert counts == {346_885}
    matched = yaml.safe_load((study.parent / "c3/config.yaml").read_text())["matched_training"]
    assert (
        abs(counts.pop() / matched["parameter_count_target"] - 1)
        <= matched["parameter_count_tolerance"]
    )
    previous = yaml.safe_load(
        (root / "studies/study_064_c3_reference_grid/config.yaml").read_text()
    )["reference_settings"]["regsi_real"]
    with pytest.raises(ValueError, match="historical reference settings are absent"):
        c3_reference_configuration(candidates, previous)
    for method, settings in contract.get("reference_settings", {}).items():
        options = c3_search_candidates(
            yaml.safe_load((study / f"{method}.yaml").read_text()), grid["methods"][method]
        )
        c3_reference_configuration(options, settings)
    pilot = (
        "runs/study_067_forge_c3_common_grid/"
        f"20261022T035444744870Z_05c92e79ce80_p2_{dataset}_regsi_v3"
    )
    assert pilot in contract["import_from"]


@pytest.mark.parametrize("dataset", ["c3", "forge"])
def test_both_prepares_reject_an_undeclared_fixed_change_before_input_io(tmp_path, dataset):
    from seis_interp.data.c3_region_inputs import prepare_c3_region_experiment
    from seis_interp.data.forge_grid_inputs import prepare_forge_grid_experiment

    source = Path(__file__).resolve().parents[2] / "studies/study_067_forge_c3_common_grid"
    shutil.copytree(source, tmp_path / "study")
    method = tmp_path / "study/forge/nersi_profile.yaml"
    config = yaml.safe_load(method.read_text())
    config["training"]["coordinate_jitter_cells"] = 0.25
    method.write_text(yaml.safe_dump(config))
    prepare = prepare_c3_region_experiment if dataset == "c3" else prepare_forge_grid_experiment
    with pytest.raises(ValueError, match="training.coordinate_jitter_cells"):
        prepare(tmp_path, tmp_path / "study" / dataset)
    assert not (tmp_path / "runs").exists()
