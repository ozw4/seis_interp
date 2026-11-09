"""A tiny FORGE grid built from the existing synthetic single-config preparation."""

from pathlib import Path

import yaml

from seis_interp.data.forge_trace_experiment import prepare_forge_trace_experiment
from tests.fixtures.forge_mvp import tiny_forge_study


def tiny_forge_grid_study(root):
    study, contract, _, configs, *_ = tiny_forge_study(root)
    source = root / "src/seis_interp/fixture_snapshot.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic preparation source snapshot\n")
    contract.update(
        methods=["pocs_grid"],
        mask_study_id="historical_outer",
        validation_study_id="historical_holdout",
        natural_missing_cell_count=1,
        missing_fraction=0.5,
        test_trace_count=18,
        observed_trace_count=17,
        time_sample_count=3,
        sample_interval_us=4000,
        validation_holdout={"fraction": 0.3, "mask_id": "tiny_holdout", "mask_seed": 19},
        matched_training={"loss_traces_per_update": 2},
        time_sampling={
            "native_sample_count": 9,
            "native_sample_interval_us": 1000,
            "decimation_factor": 4,
        },
    )
    contract["preprocessing"].update(
        amplitude_scaling="trace_oracle", alignment={"velocity_m_per_s": 10000.0}
    )
    config = configs["pocs_grid"]
    config["algorithm"].update(n_iterations=2, window_shape=[3, 2, 3, 2, 3], overlap=[0] * 5)
    config["execution"] = {"backend": "batched", "cpu_threads": 1}
    (study / "pocs_grid.yaml").write_text(yaml.safe_dump(config))
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    reference = prepare_forge_trace_experiment(root, study)
    contract.update(
        reference_experiment=str(reference.relative_to(root)),
        trace_split={"search_seed": 19, "evaluation_seeds": [7311, 7312, 7313]},
    )
    (study / "config.yaml").write_text(yaml.safe_dump(contract))
    grid = {
        "methods": {
            "pocs_grid": {
                "axis_a": [
                    {"id": str(v), "set": {"algorithm.threshold_start": v}}
                    for v in [0.6, 0.7, 0.8, 0.9, 1.0]
                ],
                "axis_b": [
                    {"id": str(v), "set": {"algorithm.threshold_end": v}}
                    for v in [0.01, 0.02, 0.03, 0.04, 0.05]
                ],
            }
        }
    }
    (study / "grid.yaml").write_text(yaml.safe_dump(grid))
    script = root / "scripts/run_forge_grid_experiment.py"
    script.write_bytes(
        (Path(__file__).resolve().parents[2] / "scripts/run_forge_grid_experiment.py").read_bytes()
    )
    return study, contract, config, reference


def tiny_forge_grid_field_study(root):
    """The same sealed FORGE inputs with one declared implicit trace field."""
    study, contract, _, reference = tiny_forge_grid_study(root)
    config = {
        "method_id": "inr_real",
        "geometry_mode": "real",
        "device": "cpu",
        "loss": "masked_trace_mse",
        "checkpoint_rule": "fixed_final_ema",
        "model_seed": contract["model_seed"],
        "model": {
            "input_features": 4,
            "fourier_components": 2,
            "frequency_base": 1.1,
            "hidden_width": 4,
            "hidden_layers": 2,
            "activation": "gelu",
            "output_activation": "linear",
            "time_sample_count": contract["time_sample_count"],
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "max_steps": 2,
            "weight_decay": 0.0,
            "report_interval": 1,
            "learning_rate_schedule": "constant",
            "mixed_precision": "off",
            "ema_decay": 0.8,
            "batch_size": contract["matched_training"]["loss_traces_per_update"],
        },
        "prediction": {"batch_size": 2},
    }
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
    return study, contract, config, reference
