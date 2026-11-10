"""Validate the fixed MVP method matrix before reading any waveform."""

from seis_interp.data.forge_aligned_inputs import validate_alignment
from seis_interp.processing.forge_mvp_contract import METHODS, validate_regsi_pair
from seis_interp.processing.forge_time_sampling import native_time_contract

NEURAL_METHODS = ("ccnet5d_grid", "nersi_real", "regsi_real", "regsi_grid")
# Methods added to a sealed six-method preparation (``derive_forge_preparation``).
ADDED_METHODS = ("nersi_profile",)
# Implicit trace fields: four continuous coordinates to one complete trace, no output canvas.
# ``cinr_real`` additionally reads a declared number of observed neighbours.
FIELD_METHODS = ("inr_real", "cinr_real")
TUNABLE_NEURAL_METHODS = (*NEURAL_METHODS, *ADDED_METHODS, *FIELD_METHODS)
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
        validate_neural_training(name, config, contract)
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


def validate_added_method_config(config, contract, *, loss_name="masked_trace_relative_mse"):
    """An added method keeps the shared contract of the sealed preparation it joins.

    ``nersi_profile`` decodes one complete ``(time, receiver point)`` profile per
    source cell and receiver line; both profile sides are padded to a multiple of
    their decoder upsampling (``profile_time_multiple`` in time, eight in receiver
    points) and cropped back after decoding.
    """
    name = config["method_id"]
    if name not in ADDED_METHODS or config["geometry_mode"] != "real":
        raise ValueError("unsupported added method or geometry")
    if loss_name not in ("masked_trace_mse", "masked_trace_relative_mse"):
        raise ValueError("unsupported profile NeRSI training loss")
    if config["loss"] != loss_name:
        if loss_name == "masked_trace_relative_mse":
            raise ValueError("MVP requires the existing relative-MSE loss")
        raise ValueError("profile NeRSI loss must match the declared training loss")
    if "alignment" in config:
        raise ValueError("alignment is a shared preprocessing condition, not a method setting")
    validate_neural_training(name, config, contract)
    if "weight_decay_after" in config["training"]:
        raise ValueError("profile NeRSI keeps one constant weight decay")
    batch = config["prediction"].get("profile_batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("prediction.profile_batch_size must be a positive integer")
    expected = [
        profile_side(contract["time_sample_count"], profile_time_multiple(config["model"])),
        profile_side(contract["spatial_shape"][3]),
    ]
    if config["model"]["profile_shape"] != expected:
        raise ValueError("NeRSI profiles must cover complete traces and receiver lines")
    if "matched_training" in contract:
        validate_matched_training(
            {name: config}, contract["matched_training"], contract["observed_trace_count"]
        )


def validate_field_method_config(config, contract, *, loss_name="masked_trace_relative_mse"):
    """An implicit trace field keeps the shared contract of the preparation it joins.

    ``inr_real`` maps four unrounded local coordinates to one complete trace
    through a single linear output layer, so the output layer spans the declared
    time axis exactly and no sample of it is discarded. ``cinr_real`` decodes the
    same time axis from a declared number of observed neighbours.
    """
    name = config["method_id"]
    if name not in FIELD_METHODS or config["geometry_mode"] != "real":
        raise ValueError("unsupported field method or geometry")
    if loss_name not in ("masked_trace_mse", "masked_trace_relative_mse"):
        raise ValueError("unsupported trace field training loss")
    if config["loss"] != loss_name:
        if loss_name == "masked_trace_relative_mse":
            raise ValueError("MVP requires the existing relative-MSE loss")
        raise ValueError("trace field loss must match the declared training loss")
    if "alignment" in config:
        raise ValueError("alignment is a shared preprocessing condition, not a method setting")
    validate_neural_training(name, config, contract)
    if "weight_decay_after" in config["training"]:
        raise ValueError("the trace field keeps one constant weight decay")
    if config["model"]["time_sample_count"] != contract["time_sample_count"]:
        raise ValueError("the trace field output must use complete traces")
    batch = config["prediction"].get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("prediction.batch_size must be a positive integer")
    context = config.get("context")
    if (name == "cinr_real") != (context is not None):
        raise ValueError("only the conditioned field declares a context")
    if context is not None:
        neighbors = context.get("neighbor_count")
        if (
            not set(context) <= {"neighbor_count", "distance", "offset_weight"}
            or "neighbor_count" not in context
            or isinstance(neighbors, bool)
            or not isinstance(neighbors, int)
            or neighbors < 1
        ):
            raise ValueError("context.neighbor_count must be a positive integer")
        if context.get("distance", "index") not in ("index", "physical"):
            raise ValueError("context.distance must be index or physical")
        weight = context.get("offset_weight")
        if (config["model"]["input_features"] == 5) != (weight is not None):
            raise ValueError("the log-offset axis needs exactly one declared context.offset_weight")
        if weight is not None and (
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0
        ):
            raise ValueError("context.offset_weight must be a nonnegative number")
    if "matched_training" in contract:
        validate_matched_training(
            {name: config}, contract["matched_training"], contract["observed_trace_count"]
        )


def profile_side(length: int, multiple: int = 8) -> int:
    """Smallest multiple of ``multiple`` covering ``length`` (eight: three 2x decoder stages)."""
    return -(-int(length) // multiple) * multiple


def profile_time_multiple(model: dict) -> int:
    """Time upsampling of a profile decoder: three 2x stages and one per time-only stage."""
    return 8 * 2 ** len(model.get("time_upsample_channels") or ())


def canvas_time_stages(profile_length: int) -> int:
    """Smallest s for which ceil(T / 2**(3+s)) is at most the C3 latent length 48."""
    stages = 0
    while -(-profile_length // (8 * 2**stages)) > 48:
        stages += 1
    return stages


def validate_neural_training(name, config, contract):
    """Contract seed, fixed final checkpoint, constant schedule and optimizer of one method."""
    if config["model_seed"] != contract["model_seed"]:
        raise ValueError("model seed mismatch")
    training = config["training"]
    if config["checkpoint_rule"] not in ("fixed_final", "fixed_final_ema"):
        raise ValueError("only fixed final checkpoints are allowed")
    if (config["checkpoint_rule"] == "fixed_final_ema") != bool(training.get("ema_decay")):
        raise ValueError("EMA checkpoint rule mismatch")
    precisions = ("off", "bf16") if name == "ccnet5d_grid" else ("off",)
    if (
        training["max_steps"] < 1
        or training["learning_rate_schedule"] != "constant"
        or training["mixed_precision"] not in precisions
    ):
        raise ValueError("unsupported fixed training schedule")
    optimizers = ("adamw",) if name.startswith("regsi") else ("adam",)
    if name in ("ccnet5d_grid", "nersi_profile", "inr_real", "cinr_real"):
        optimizers = ("adam", "adamw")
    if training["optimizer"] not in optimizers:
        raise ValueError("optimizer contract mismatch")


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
    allowed = (
        {"backend", "workers", "batch_matrix_limit"}
        if config["method_id"] == "drr_grid"
        else {"backend"}
    )
    allowed.add("cpu_threads")
    workers = execution.get("workers", 1)
    threads = execution.get("cpu_threads", 1)
    matrix_limit = execution.get("batch_matrix_limit", 2_000_000)
    if (
        set(execution) - allowed
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads < 1
        or isinstance(matrix_limit, bool)
        or not isinstance(matrix_limit, int)
        or matrix_limit < 1
        or ("workers" in execution and execution["backend"] != "batched")
    ):
        raise ValueError("unsupported classical execution settings")


def validate_matched_training(configs, budget, observed_trace_count):
    """Equal updates, learning rate and loss traces per optimizer update.

    Follows the C3 NA convention (studies 043/044/046): one forward, one backward
    and one optimizer step per update, without gradient accumulation. NeRSI draws
    ``loss_traces_per_update`` observed traces; CCNet5D stacks sampled patches and
    selects exactly that many hidden traces; profile NeRSI stacks sampled profiles
    and selects exactly that many observed traces; ReGSI queries that many hidden
    traces, and an episode must hide at least one full batch (its last batch may be
    shorter).
    """
    if set(budget) != MATCHED_TRAINING_KEYS:
        raise ValueError("matched training budget keys mismatch")
    if not 0 < budget["parameter_count_tolerance"] < 1 or budget["parameter_count_target"] < 1:
        raise ValueError("invalid matched parameter count")
    traces = budget["loss_traces_per_update"]
    for name in TUNABLE_NEURAL_METHODS:
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
        if name == "nersi_profile" and training.get("supervised_traces_per_update") != traces:
            raise ValueError("profile NeRSI must select exactly the budgeted traces per update")
        if name in FIELD_METHODS and training["batch_size"] != traces:
            raise ValueError("trace field batch size must equal the budgeted loss traces")
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
