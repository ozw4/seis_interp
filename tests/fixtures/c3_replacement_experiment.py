"""Tiny replacement searches joined to the remaining C3 method workers."""

from pathlib import Path

import yaml

from seis_interp.data.forge_mvp_artifacts import write_json


def tiny_c3_reference_study(repo: Path, original: Path, methods=("regsi_real", "nersi_profile")):
    study = repo / "studies/tiny_c3_reference_grid"
    study.mkdir()
    contract = yaml.safe_load((original / "study/config.yaml").read_text())
    contract.update(
        study_id=study.name,
        validation_study_id=contract["study_id"],
        methods=list(methods),
        training_loss="masked_trace_mse",
        continuing_experiment=str(original.relative_to(repo)),
    )
    configs = {}
    for method in contract["methods"]:
        config = yaml.safe_load((original / "study" / f"{method}.yaml").read_text())
        config["loss"] = "masked_trace_mse"
        if method == "regsi_real":
            config["model"]["attention_pooling"] = "rms"
            config["geometry_features"]["position_scale_m"] = 500.0
            config["geometry_features"]["offset_scale_m"] = 500.0
        elif method == "nersi_profile":
            config["training"]["nuclear_norm_weight"] = 0.0
        configs[method] = config
        (study / f"{method}.yaml").write_text(yaml.safe_dump(config))
        write_json(
            original / "workers" / f"{method}.json",
            {"method_id": method, "status": "stopped", "s1_completed": False, "completed_s1": 2},
        )
    for method in ("ccnet5d_grid", "pocs_grid", "drr_grid"):
        if method in methods:
            continue
        write_json(
            original / "workers" / f"{method}.json",
            {"method_id": method, "status": "running", "s1_completed": False, "completed_s1": 0},
        )
    contract["reference_settings"] = {
        "regsi_real": {
            "model.temporal_dilations": [1, 2],
            "graph.neighbors_per_relation": 3,
            "model.attention_pooling": "rms",
            "geometry_features.position_scale_m": 500.0,
        },
        "nersi_profile": {
            "training.nuclear_norm_weight": 0.0,
            "model.frequency_base": 1.04,
        },
    }
    grid = {
        "methods": {
            "regsi_real": {
                "axis_a": [
                    {"id": str(n), "set": {"model.temporal_dilations": [n, 2 * n]}}
                    for n in range(1, 6)
                ],
                "axis_b": [
                    {"id": str(n), "set": {"graph.neighbors_per_relation": n}} for n in range(1, 6)
                ],
            },
            "nersi_profile": {
                "axis_a": [
                    {"id": str(weight), "set": {"training.nuclear_norm_weight": weight}}
                    for weight in (0.0, 0.001, 0.01, 0.03, 0.1)
                ],
                "axis_b": [
                    {"id": str(base), "set": {"model.frequency_base": base}}
                    for base in (1.01, 1.02, 1.03, 1.04, 1.05)
                ],
            },
        }
    }
    if "ccnet5d_grid" in methods:
        contract["reference_settings"]["ccnet5d_grid"] = {"model.time_dilation": 1}
        grid["methods"]["ccnet5d_grid"] = yaml.safe_load(
            (original / "study/grid.yaml").read_text()
        )["methods"]["ccnet5d_grid"]
    contract["cuda_execution"] = {
        "minimum_free_memory_gib": {
            method: {f"a{a:02d}_b{b:02d}": a + b for a in range(1, 6) for b in range(1, 6)}
            for method in methods
        }
    }
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    (study / "grid.yaml").write_text(yaml.safe_dump(grid))
    (study / "inputs.yaml").write_bytes((original / "study/inputs.yaml").read_bytes())
    return study, contract, configs
