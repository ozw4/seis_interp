"""Validate the five-method C3 fair-grid contract before reading any waveform.

The C3 crop runs the FORGE fair-grid methods under the C3 v3 preprocessing: one
observed-only global RMS (no trace RMS, physical-amplitude predictions), no
moveout alignment (the canvas is the trace) and no time resampling. ReGSI runs
on real geometry only; profile NeRSI decodes complete ``(time, relative
receiver y)`` profiles; the implicit trace field returns one complete trace per
coordinate query.
"""

import numpy as np

from seis_interp.forge_mvp_config import (
    validate_added_method_config,
    validate_classical_execution,
    validate_field_method_config,
    validate_matched_training,
    validate_neural_training,
)
from seis_interp.processing.c3_fair_grid_contract import C3_DATASET_ID
from seis_interp.processing.forge_mvp_contract import AXES

C3_FAIR_GRID_METHODS = ("pocs_grid", "drr_grid", "ccnet5d_grid", "regsi_real", "nersi_profile")
C3_NEURAL_METHODS = ("ccnet5d_grid", "regsi_real", "nersi_profile")
# Join the same C3 contract only when a study declares them; not part of the five-method default.
C3_ADDED_METHODS = ("inr_real", "cinr_real")
C3_GEOMETRY_MODES = {
    "pocs_grid": "grid",
    "drr_grid": "grid",
    "ccnet5d_grid": "grid",
    "regsi_real": "real",
    "nersi_profile": "real",
    "inr_real": "real",
    "cinr_real": "real",
}


def c3_experiment_methods(contract: dict) -> tuple[str, ...]:
    """The explicit active methods, or the established five-method experiment."""
    methods = tuple(contract.get("methods", C3_FAIR_GRID_METHODS))
    supported = set(C3_FAIR_GRID_METHODS) | set(C3_ADDED_METHODS)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= supported:
        raise ValueError("C3 experiment methods must be distinct supported methods")
    return methods


def validate_c3_fair_grid_configs(configs: dict, contract: dict) -> None:
    """Shared C3 contract, then each method config under the matched budget."""
    _validate_c3_contract(contract)
    if any("alignment" in config for config in configs.values()):
        raise ValueError("alignment is a shared preprocessing condition, not a method setting")
    methods = c3_experiment_methods(contract)
    if set(configs) != set(methods):
        raise ValueError(f"the C3 fair grid requires exactly {methods}")
    training_loss = contract.get("training_loss", "masked_trace_relative_mse")
    if training_loss not in ("masked_trace_mse", "masked_trace_relative_mse"):
        raise ValueError("unsupported C3 training loss")
    for name, config in configs.items():
        if config["method_id"] != name or config["geometry_mode"] != C3_GEOMETRY_MODES[name]:
            raise ValueError("method geometry mismatch")
        trains = name in C3_NEURAL_METHODS or name in C3_ADDED_METHODS
        expected_loss = training_loss if trains else "masked_trace_relative_mse"
        if config["loss"] != expected_loss:
            if expected_loss == "masked_trace_relative_mse":
                raise ValueError("MVP requires the existing relative-MSE loss")
            raise ValueError("C3 method loss must match the declared training loss")
        if name in ("pocs_grid", "drr_grid"):
            if config["checkpoint_rule"] != "none" or config["model_seed"] is not None:
                raise ValueError("deterministic method contract mismatch")
            if name == "drr_grid" and (
                config["algorithm"]["frequency_min_hz"] != 0
                or config["algorithm"]["frequency_max_hz"] is not None
            ):
                raise ValueError("method-specific frequency filtering is prohibited")
            validate_classical_execution(config)
        elif name == "nersi_profile":
            validate_added_method_config(config, contract, loss_name=expected_loss)
        elif name in C3_ADDED_METHODS:
            validate_field_method_config(config, contract, loss_name=expected_loss)
        else:
            validate_neural_training(name, config, contract)
        if (
            name == "ccnet5d_grid"
            and config["patches"]["shape"][0] != contract["time_sample_count"]
        ):
            raise ValueError("CCNet loss must use complete traces")
    validate_matched_training(
        configs, contract["matched_training"], contract["observed_trace_count"]
    )


def _validate_c3_contract(contract: dict) -> None:
    if contract.get("dataset_id") != C3_DATASET_ID:
        raise ValueError(f"the C3 fair grid requires dataset_id {C3_DATASET_ID}")
    if contract["axis_order"] != AXES or len(contract["spatial_shape"]) != 4:
        raise ValueError("axis order or grid shape mismatch")
    if (
        contract["eligible_trace_count"] != int(np.prod(contract["spatial_shape"]))
        or contract.get("natural_missing_cell_count", 0) != 0
    ):
        raise ValueError("the C3 crop is dense: every cell holds one eligible trace")
    if "time_sampling" in contract:
        raise ValueError("C3 traces are cropped, not resampled: no time_sampling block")
    preprocessing = contract["preprocessing"]
    if (
        preprocessing["amplitude_scaling"] != "observed_global_rms"
        or preprocessing["trace_rms"] is not False
        or preprocessing.get("idw") is not None
    ):
        raise ValueError("the C3 fair grid uses the observed global RMS only")
    if preprocessing.get("alignment") is not None:
        raise ValueError("the C3 fair grid has no moveout alignment")
    if "matched_training" not in contract:
        raise ValueError("the C3 fair grid declares matched_training")
