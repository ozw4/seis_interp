"""Validate the fixed MVP method matrix before reading any waveform."""

from seis_interp.processing.forge_mvp_contract import METHODS, validate_regsi_pair


def validate_mvp_configs(configs, contract):
    if set(configs) != set(METHODS):
        raise ValueError("MVP requires exactly six methods")
    for name, config in configs.items():
        if config["method_id"] != name or config["geometry_mode"] != name.rsplit("_", 1)[1]:
            raise ValueError("method geometry mismatch")
        if config["loss"] != "masked_trace_relative_mse":
            raise ValueError("MVP requires the existing relative-MSE loss")
        if name in ("pocs_grid", "drr_grid"):
            if config["checkpoint_rule"] != "none" or config["model_seed"] is not None:
                raise ValueError("deterministic method contract mismatch")
            if name == "drr_grid" and (
                config["algorithm"]["frequency_min_hz"] != 0
                or config["algorithm"]["frequency_max_hz"] is not None
            ):
                raise ValueError("method-specific frequency filtering is prohibited")
            continue
        if config["model_seed"] != contract["model_seed"]:
            raise ValueError("model seed mismatch")
        training = config["training"]
        if config["checkpoint_rule"] not in ("fixed_final", "fixed_final_ema"):
            raise ValueError("only fixed final checkpoints are allowed")
        if (config["checkpoint_rule"] == "fixed_final_ema") != bool(training.get("ema_decay")):
            raise ValueError("EMA checkpoint rule mismatch")
        if (
            training["max_steps"] < 1
            or training["learning_rate_schedule"] != "constant"
            or training["mixed_precision"] != "off"
        ):
            raise ValueError("unsupported fixed training schedule")
        optimizer = "adamw" if name.startswith("regsi") else "adam"
        if training["optimizer"] != optimizer:
            raise ValueError("optimizer contract mismatch")
        if (
            name == "ccnet5d_grid"
            and config["patches"]["shape"][0] != contract["time_sample_count"]
        ):
            raise ValueError("CCNet loss must use complete traces")
        if (
            name == "nersi_real"
            and config["model"]["time_sample_count"] != contract["time_sample_count"]
        ):
            raise ValueError("NeRSI output must use complete traces")
    validate_regsi_pair(configs)
