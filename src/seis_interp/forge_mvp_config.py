"""Validate the fixed MVP method matrix before reading any waveform."""

from seis_interp.data.forge_aligned_inputs import validate_alignment
from seis_interp.processing.forge_mvp_contract import METHODS, validate_regsi_pair
from seis_interp.processing.forge_time_sampling import native_time_contract

NEURAL_METHODS = ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")
MATCHED_TRAINING_KEYS = {
    "max_steps",
    "learning_rate",
    "loss_traces_per_update",
    "parameter_count_target",
    "parameter_count_tolerance",
}


def validate_mvp_configs(configs, contract):
    """Study configs use the contract seed; matched-tuning candidates may replace it."""
    native_time_contract(contract)
    preprocessing = contract["preprocessing"]
    mode = preprocessing["amplitude_scaling"]
    if mode not in ("observed_global_rms", "observed_trace_rms_idw", "trace_oracle"):
        raise ValueError("unsupported amplitude scaling")
    if bool(preprocessing["trace_rms"]) != (mode != "observed_global_rms"):
        raise ValueError("trace RMS preprocessing contract mismatch")
    if mode == "trace_oracle" and preprocessing.get("idw") is not None:
        raise ValueError("oracle trace RMS must not declare IDW restoration")
    if mode == "observed_trace_rms_idw" and preprocessing.get("idw") != {
        "geometry": "projected_grid",
        "radius": 2,
        "power": 2.0,
        "distance": "physical",
        "restoration": "multiply_once",
    }:
        raise ValueError("unsupported FORGE trace RMS IDW settings")
    alignment = preprocessing.get("alignment")
    if alignment is not None:
        validate_alignment(alignment)
        if set(alignment) != {"velocity_m_per_s"}:
            raise ValueError("the study declares only the alignment velocity")
    if any("alignment" in config for config in configs.values()):
        raise ValueError("alignment is a shared preprocessing condition, not a method setting")
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
            validate_classical_execution(config)
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
    if "matched_training" in contract:
        validate_matched_training(
            configs, contract["matched_training"], contract["observed_trace_count"]
        )


def validate_classical_execution(config):
    """Optional CPU execution of POCS/DRR by the batched reference implementation.

    The batched CPU path uses NumPy FFT/SVD like the sequential path; it only
    groups windows (and, for DRR, runs them on worker threads).
    """
    execution = config.get("execution")
    if execution is None:
        return
    if config["device"] != "cpu" or execution.get("backend") not in ("sequential", "batched"):
        raise ValueError("classical execution must be sequential or batched on the CPU")
    allowed = {"backend", "workers"} if config["method_id"] == "drr_grid" else {"backend"}
    workers = execution.get("workers", 1)
    if (
        set(execution) - allowed
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or ("workers" in execution and execution["backend"] != "batched")
    ):
        raise ValueError("unsupported classical execution settings")


def validate_matched_training(configs, budget, observed_trace_count):
    """Equal updates, learning rate and loss traces per optimizer update.

    Follows the C3 NA convention (studies 043/044/046): one forward, one backward
    and one optimizer step per update, without gradient accumulation. NeRSI draws
    ``loss_traces_per_update`` observed traces; CCNet5D stacks sampled patches and
    selects exactly that many hidden traces; ReGSI queries that many hidden traces,
    and an episode must hide at least one full batch (its last batch may be shorter).
    """
    if set(budget) != MATCHED_TRAINING_KEYS:
        raise ValueError("matched training budget keys mismatch")
    if not 0 < budget["parameter_count_tolerance"] < 1 or budget["parameter_count_target"] < 1:
        raise ValueError("invalid matched parameter count")
    traces = budget["loss_traces_per_update"]
    for name in NEURAL_METHODS:
        if name not in configs:
            continue
        training = configs[name]["training"]
        if (
            training["max_steps"] != budget["max_steps"]
            or training["learning_rate"] != budget["learning_rate"]
        ):
            raise ValueError(f"{name}: updates and learning rate must match the budget")
        if name == "ccnet5d_grid" and training.get("supervised_traces_per_update") != traces:
            raise ValueError("CCNet5D must select exactly the budgeted traces per update")
        if name == "nersi_real" and training["batch_size"] != traces:
            raise ValueError("NeRSI batch size must equal the budgeted loss traces")
        if name.startswith("regsi"):
            hidden = round(observed_trace_count * training["inner_mask_fraction"])
            if training["query_batch_size"] != traces or hidden < traces:
                raise ValueError("ReGSI must query the budgeted traces per update")


def validate_parameter_counts(counts, budget):
    """Every neural parameter count lies within the relative tolerance of the target."""
    target, tolerance = budget["parameter_count_target"], budget["parameter_count_tolerance"]
    for name, count in counts.items():
        if abs(count / target - 1) > tolerance:
            raise ValueError(f"{name}: {count} parameters outside the matched model size")
