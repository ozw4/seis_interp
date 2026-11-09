"""A complete tiny study of source-block partitioning and frozen method inputs."""

from pathlib import Path

import yaml

from tests.fixtures.c3_fair_grid import tiny_c3_study, tiny_c3_trace_field_config


def tiny_c3_region_study(root: Path):
    study, contract, configs = tiny_c3_study(root)
    contract.pop("validation_holdout")
    contract["region_split"] = {
        "source_block_shape": [1, 2],
        "block_counts": {"train": 4, "validation": 2, "test": 2},
        "search_seed": 21,
        "evaluation_seeds": [7311, 7312, 7313],
    }
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    for method in ("regsi_real", "ccnet5d_grid", "nersi_profile"):
        configs[method]["training"]["optimizer"] = "adamw"
        (study / f"{method}.yaml").write_text(yaml.safe_dump(configs[method]))
    methods = {}
    paths = {
        "pocs_grid": "algorithm.threshold_end",
        "drr_grid": "algorithm.damping_power",
        "ccnet5d_grid": "model.time_dilation",
        "regsi_real": "graph.neighbors_per_relation",
        "nersi_profile": "model.frequency_base",
    }
    for method in configs:
        values = [0.1, 0.2, 0.3, 0.4, 0.5] if method == "pocs_grid" else [1, 2, 3, 4, 5]
        if method == "nersi_profile":
            values = [1.01, 1.02, 1.03, 1.04, 1.05]
        methods[method] = {
            "axis_a": [{"id": str(value), "set": {paths[method]: value}} for value in values],
            "axis_b": [{"id": str(value), "set": {"test_axis": value}} for value in range(5)],
        }
    (study / "grid.yaml").write_text(yaml.safe_dump({"methods": methods}))
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "run_c3_region_experiment.py").write_text("# frozen test entrypoint\n")
    return study, contract, configs


def tiny_c3_trace_study(root: Path):
    """The same five models with an observed-trace holdout on one shared crop."""
    study, contract, configs = tiny_c3_region_study(root)
    contract.pop("region_split")
    contract["trace_split"] = {
        "search_seed": 21,
        "evaluation_seeds": [7311, 7312, 7313],
    }
    contract["validation_holdout"] = {
        "fraction": 0.1,
        "mask_id": "tiny_c3_validation_v1",
        "mask_seed": 21,
    }
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    return study, contract, configs


def tiny_c3_trace_field_study(root: Path):
    """One declared implicit trace field on the trace holdout, with the production axes."""
    study, contract, configs = tiny_c3_trace_study(root)
    config = tiny_c3_trace_field_config()
    config["loss"] = "masked_trace_mse"
    contract["methods"] = ["inr_real"]
    contract["training_loss"] = "masked_trace_mse"
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    (study / "inr_real.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (study / "grid.yaml").write_text(
        yaml.safe_dump(
            {
                "methods": {
                    "inr_real": {
                        "axis_a": [
                            {"id": str(value), "set": {"training.weight_decay": value}}
                            for value in (0.0, 0.001, 0.01, 0.03, 0.1)
                        ],
                        "axis_b": [
                            {"id": str(value), "set": {"model.frequency_base": value}}
                            for value in (1.05, 1.075, 1.1, 1.15, 1.25)
                        ],
                    }
                }
            }
        )
    )
    return study, contract, {"inr_real": config}
