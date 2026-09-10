"""Connect fixed-suite readers to disposable neural pilot resource measurements."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_artifacts import (
    write_benchmark_json,
    write_benchmark_yaml,
)
from seis_interp.data.c3_benchmark_inputs import (
    load_c3_benchmark_supervised_source,
    load_c3_benchmark_training_graph_domain,
    load_c3_benchmark_volume_inputs,
)
from seis_interp.data.c3_benchmark_suite import (
    VerifiedC3BenchmarkSuite,
    c3_suite_case,
    load_c3_benchmark_input_manifest,
    suite_path,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.models.siren import Siren
from seis_interp.pipelines.preflight_relational_trace_graph import (
    preflight_relational_trace_graph_run,
)
from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run
from seis_interp.processing.c3_benchmark_contract import (
    MAIN_C3_DIMENSIONS,
    C3BenchmarkDimensions,
)
from seis_interp.processing.ccnet5d_tiles import iter_ccnet5d_tiles
from seis_interp.processing.trace_graph_diagnostics import (
    trace_graph_resource_measurements,
)
from seis_interp.processing.trace_graph_preprocessing import (
    fit_trace_graph_preprocessing,
)
from seis_interp.relational_trace_graph_config import (
    validate_relational_trace_graph_training_config,
)
from seis_interp.training.c3_first_results_neural_preflight import (
    estimate_c3_first_results_graph_budget,
    measure_c3_first_results_graph_training_batch,
    synchronize_preflight_device,
)
from seis_interp.training.c3_volume_siren_data import (
    build_c3_volume_siren_data,
    build_c3_volume_siren_sampler,
    validate_c3_volume_siren_scaling,
)
from seis_interp.training.c3_volume_siren_options import (
    complete_trace_training_options,
    initial_time_weight_scale,
)
from seis_interp.training.ccnet5d_checkpoints import load_ccnet5d_checkpoint
from seis_interp.training.ccnet5d_patches import (
    make_ccnet_training_dataset_patch_plan,
)
from seis_interp.training.ccnet5d_source_binding import (
    QC_TRAINING_DATASET,
    load_ccnet5d_supervised_source,
)
from seis_interp.training.devices import resolve_device
from seis_interp.training.fixed_step_siren import (
    train_siren_complete_trace_steps,
    train_siren_fixed_steps,
)
from seis_interp.training.point_sampler import build_trace_coordinate_points
from seis_interp.training.prediction import predict_points
from seis_interp.training.siren_initialization import apply_siren_time_weight_initialization


def run_c3_first_results_neural_preflight(
    action: str,
    *,
    config_path: Path,
    suite_dir: Path,
    case_id: str,
    output_dir: Path,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
    verified_suite: VerifiedC3BenchmarkSuite | None = None,
    prediction_config_path: Path | None = None,
    normalization_checkpoint_path: Path | None = None,
    query_counts: tuple[int, ...] = (1, 8),
    query_limit: int = 32,
) -> dict[str, object]:
    """Measure only the requested smoke; save successful and blocked strict JSON.

    Call in a fresh process with the execution environment's explicit timeout.
    This function never launches the main pilot or a full validation prediction.
    """
    if action not in ("siren", "ccnet-train", "gnn-preflight"):
        raise ValueError("unsupported neural preflight action")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = {
        "status": "running",
        "action": action,
        "scope": "disposable_neural_preflight",
        "training_started": False,
        "full_validation_prediction_completed": False,
        "configuration_sha256": file_sha256(config_path),
        "case_id": case_id,
    }
    try:
        config = load_resolved_config(config_path)
        write_benchmark_yaml(output / "pilot_config.yaml", config)
        device = resolve_device(config["training"]["device"])
        suite_dir = Path(suite_dir)
        verified = verified_suite
        suite = load_c3_benchmark_input_manifest(
            suite_dir, case_id=case_id, dimensions=dimensions, verified_suite=verified
        )
        entry = c3_suite_case(suite, case_id)
        arguments = {
            "suite_dir": suite_dir,
            "dimensions": dimensions,
            "verified_suite": verified,
        }
        if action == "gnn-preflight":
            details = _graph_preflight(
                config_path,
                config,
                entry,
                suite,
                device,
                arguments,
                query_counts,
                query_limit,
                output,
            )
        else:
            inputs = load_c3_benchmark_volume_inputs(case_id=case_id, **arguments)
            if action == "siren":
                if config["project"]["random_seed"] != inputs.case["mask"]["random_seed"]:
                    raise ValueError("SIREN project seed must equal the fixed case mask seed")
                details = _siren_preflight(config, inputs, device)
            else:
                if config["supervision"].get("sampling_domain") == QC_TRAINING_DATASET:
                    source = load_ccnet5d_supervised_source(
                        config,
                        interim_dir=suite_path(suite_dir, suite["interim"]),
                        processed_dir=suite_path(suite_dir, suite["processed"]),
                        suite_dir=suite_dir,
                        normalization_checkpoint_path=normalization_checkpoint_path,
                        dimensions=dimensions,
                    )
                else:
                    source = load_c3_benchmark_supervised_source(
                        fit_region=config["supervision"]["fit_region"],
                        selection_region=config["supervision"]["selection_region"],
                        **arguments,
                    )
                details = _ccnet_preflight(
                    config,
                    source,
                    inputs,
                    suite,
                    suite_dir,
                    output,
                    device,
                    prediction_config_path,
                    normalization_checkpoint_path,
                    dimensions,
                )
        report.update(status="success", device=str(device), **details)
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        report.update(
            status="blocked",
            blockers=[{"type": type(error).__name__, "message": str(error)}],
        )
    report["elapsed_seconds"] = perf_counter() - started
    write_benchmark_json(output / "preflight.json", report)
    return report


def _siren_preflight(config, inputs, device):
    started = perf_counter()
    training = config["training"]
    time_weight_scale = initial_time_weight_scale(training)
    scaling, interpolation = validate_c3_volume_siren_scaling(
        config["training"].get("amplitude_scaling", "train_global_rms"),
        config["prediction"].get("scale_interpolation"),
    )
    data = build_c3_volume_siren_data(
        inputs.observed_volume,
        inputs.index_table,
        amplitude_scaling=scaling,
        scale_interpolation=interpolation,
        coordinate_features=config["model"]["coordinate_features"],
        time_coordinate_scale=config["model"].get("time_coordinate_scale", 1.0),
        relative_receiver_y_time_shear_s_per_m=config["model"].get(
            "relative_receiver_y_time_shear_s_per_m", 0.0
        ),
    )
    data_seconds = perf_counter() - started
    model_config = {
        name: value
        for name, value in config["model"].items()
        if name
        not in (
            "name",
            "coordinate_features",
            "time_coordinate_scale",
            "relative_receiver_y_time_shear_s_per_m",
        )
    }
    complete_options = complete_trace_training_options(
        training, time_count=len(data.normalized_time)
    )
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(training["random_seed"])
        model = Siren(**model_config)
        apply_siren_time_weight_initialization(model, time_weight_scale)
        sampler = build_c3_volume_siren_sampler(data, random_seed=training["random_seed"])
        synchronize_preflight_device(device)
        started = perf_counter()
        if complete_options is None:
            result = train_siren_fixed_steps(
                model,
                sampler,
                device=device,
                learning_rate=training["learning_rate"],
                batch_size=training["batch_size"],
                max_steps=10,
                report_interval=10,
            )
        else:
            result = train_siren_complete_trace_steps(
                model,
                data.normalized_time,
                data.normalized_spatial[data.observed_flat_indices],
                data.normalized_observed_amplitudes,
                device=device,
                learning_rate=training["learning_rate"],
                microbatch_size=training["batch_size"],
                max_steps=10,
                report_interval=10,
                random_seed=training["random_seed"],
                normalized_time_offsets=(
                    None
                    if data.normalized_time_offsets is None
                    else data.normalized_time_offsets[data.observed_flat_indices]
                ),
                **complete_options,
            )
        synchronize_preflight_device(device)
        train_seconds = perf_counter() - started
        point_batch = config["prediction"]["batch_size"]
        trace_count = min(
            len(data.normalized_spatial),
            max(1, point_batch // len(data.normalized_time)),
        )
        points = build_trace_coordinate_points(
            data.normalized_time,
            data.normalized_spatial,
            np.arange(trace_count, dtype=np.int64),
            normalized_time_offsets=data.normalized_time_offsets,
        )
        started = perf_counter()
        values = predict_points(model, points, batch_size=point_batch, device=device)
        synchronize_preflight_device(device)
        prediction_seconds = perf_counter() - started
        if not np.isfinite(values).all():
            raise ValueError("SIREN smoke predictions must be finite")
        resources = trace_graph_resource_measurements(device)
    return {
        "smoke_steps": result.steps_completed,
        **(
            {"initial_time_weight_scale": time_weight_scale}
            if "initial_time_weight_scale" in training
            else {}
        ),
        **(
            {"training_history": [dict(record) for record in result.history]}
            if complete_options is not None and "envelope_loss" in complete_options
            else {}
        ),
        **(
            {
                "relative_receiver_y_time_shear_s_per_m": (
                    data.model_coordinates.relative_receiver_y_time_shear_s_per_m
                )
            }
            if data.model_coordinates.relative_receiver_y_time_shear_s_per_m != 0.0
            else {}
        ),
        **({"complete_trace_training": complete_options} if complete_options is not None else {}),
        **(
            {"amplitude_scaling": scaling, "scale_interpolation": interpolation}
            if scaling == "per_trace_rms"
            else {}
        ),
        "training_random_seed": training["random_seed"],
        "sampler_seed": training["random_seed"],
        "observed_trace_count": len(data.observed_flat_indices),
        "sampled_prediction_point_count": len(points),
        "full_volume_shape": list(inputs.observed_volume.values.shape),
        "timings": {
            "observed_data_preparation_seconds": data_seconds,
            "training_seconds": train_seconds,
            "sample_prediction_seconds": prediction_seconds,
        },
        "estimates": {
            "kind": "linear_extrapolation_not_measured_full_run",
            **(
                {"training_scope": "envelope_active_smoke_extrapolated_to_all_updates"}
                if complete_options is not None and "envelope_loss" in complete_options
                else {}
            ),
            "training_seconds": train_seconds * training["max_steps"] / 10,
            "prediction_seconds": prediction_seconds
            * inputs.observed_volume.values.size
            / len(points),
        },
        "resources": resources,
        "state_reused_by_pilot": False,
    }


def _ccnet_preflight(
    config,
    source,
    inputs,
    suite,
    suite_dir,
    output,
    device,
    prediction_path,
    normalization_checkpoint_path,
    dimensions,
):
    if prediction_path is None:
        raise ValueError("CCNet preflight requires its explicit prediction config")
    batch_size = config["training"]["batch_size"]
    smoke = {
        **config,
        "patches": {**config["patches"], "fit_count": 2 * batch_size},
        "training": {
            **config["training"],
            "max_epochs": 1,
            "validate_every_steps": 2,
            "report_every_steps": 1,
        },
    }
    smoke_path = output / "smoke_config.yaml"
    write_benchmark_yaml(smoke_path, smoke)
    full_plan_diagnostics = None
    if config["supervision"].get("sampling_domain") == QC_TRAINING_DATASET:
        patches = config["patches"]
        full_plan, full_plan_diagnostics = make_ccnet_training_dataset_patch_plan(
            source,
            patch_shape=patches["shape"],
            fit_count=patches["fit_count"],
            selection_count=patches["selection_count"],
            missing_fraction=patches["missing_fraction"],
            random_seed=patches["random_seed"],
        )
        write_benchmark_json(output / "full_patch_plan.json", full_plan.to_dict())
        write_benchmark_json(output / "full_patch_plan_diagnostics.json", full_plan_diagnostics)
    synchronize_preflight_device(device)
    started = perf_counter()
    metrics = train_ccnet5d_run(
        config_path=smoke_path,
        interim_dir=suite_path(suite_dir, suite["interim"]),
        processed_dir=suite_path(suite_dir, suite["processed"]),
        output_dir=output / "smoke_native",
        suite_dir=suite_dir,
        normalization_checkpoint_path=normalization_checkpoint_path,
        dimensions=dimensions,
    )
    synchronize_preflight_device(device)
    train_seconds = perf_counter() - started
    loaded = load_ccnet5d_checkpoint(output / "smoke_native/artifacts/final.pt", device=device)
    if loaded.training_provenance["source_inputs_lock"] != source.inputs_lock:
        raise ValueError("CCNet smoke native source differs from the verified suite source")
    model = loaded.model.eval()
    core = tuple(load_resolved_config(prediction_path)["prediction"]["core_shape"])
    tiles = list(
        iter_ccnet5d_tiles(inputs.observed_volume.values.shape, core, halo_radius=model.halo_radius)
    )
    tile = max(tiles, key=lambda value: math.prod(s.stop - s.start for s in value.input_slices))
    measurements = [
        {
            "role": role,
            **_measure_ccnet_tile(loaded, inputs.observed_volume.values, selected, device),
        }
        for role, selected in (("maximum_halo", tile), ("edge", tiles[0]))
    ]
    tile_seconds = max(measured["forward_seconds"] for measured in measurements)
    total_steps = config["training"]["max_epochs"] * math.ceil(
        config["patches"]["fit_count"] / batch_size
    )
    return {
        "smoke_steps": metrics["steps_completed"],
        "smoke_native_run": str(output / "smoke_native"),
        "source_inputs_lock": source.inputs_lock,
        "training_random_seed": config["training"]["random_seed"],
        "patch_random_seed": config["patches"]["random_seed"],
        "patches": config["patches"],
        "full_volume_shape": list(inputs.observed_volume.values.shape),
        "prediction_core_shape": list(core),
        "halo_radius": model.halo_radius,
        "measured_maximum_tile_shape": measurements[0]["input_shape"],
        "measured_tile_slices": [[s.start, s.stop] for s in tile.input_slices],
        "tile_measurements": measurements,
        "tile_count": len(tiles),
        "timings": {
            "smoke_training_and_selection_seconds": train_seconds,
            "maximum_tile_forward_seconds": measurements[0]["forward_seconds"],
            "edge_tile_forward_seconds": measurements[1]["forward_seconds"],
        },
        "estimates": {
            "kind": "linear_extrapolation_not_measured_full_run",
            "training_seconds": train_seconds * total_steps / metrics["steps_completed"],
            "prediction_seconds": tile_seconds * len(tiles),
            "training_scope": "conservative_two_step_native_run_including_selection_and_io",
            "prediction_scope": (
                "slowest_measured_tile_forward_times_total_tiles_excludes_reads_writes_scoring"
            ),
        },
        "resources": trace_graph_resource_measurements(device),
        "state_reused_by_pilot": False,
        **(
            {"full_patch_plan_diagnostics": full_plan_diagnostics}
            if full_plan_diagnostics is not None
            else {}
        ),
    }


def _measure_ccnet_tile(loaded, volume_values, tile, device):
    values = np.array(volume_values[tile.input_slices], dtype=np.float32, copy=True)
    values /= loaded.amplitude_rms
    tensor = torch.from_numpy(values[None, None]).to(device)
    synchronize_preflight_device(device)
    started = perf_counter()
    with torch.inference_mode():
        prediction = loaded.model(tensor)
    synchronize_preflight_device(device)
    seconds = perf_counter() - started
    if prediction.shape != tensor.shape or not torch.isfinite(prediction).all():
        raise ValueError("CCNet halo tile prediction has invalid shape or nonfinite amplitudes")
    return {
        "input_shape": list(values.shape),
        "output_shape": list(prediction.shape[2:]),
        "input_slices": [[s.start, s.stop] for s in tile.input_slices],
        "core_slices": [[s.start, s.stop] for s in tile.core_slices],
        "forward_seconds": seconds,
        "resources": trace_graph_resource_measurements(device),
    }


def _graph_preflight(config_path, config, entry, suite, device, arguments, counts, limit, output):
    if (
        type(limit) is not int
        or not 1 <= limit <= 32
        or not counts
        or any(type(count) is not int or not 1 <= count <= limit for count in counts)
        or list(counts) != sorted(set(counts))
    ):
        raise ValueError("GNN query counts must increase uniquely within explicit limit <=32")
    model_config, graph, options = validate_relational_trace_graph_training_config(config)
    if config["training_data"]["pool"] != "all_train_traces" or config["training_data"][
        "time_samples"
    ] != list(arguments["dimensions"].time_range):
        raise ValueError("GNN preflight requires the full suite train pool and authorized time")
    suite_dir = arguments["suite_dir"]
    paths = {
        "interim_dir": suite_path(suite_dir, suite["interim"]),
        "processed_dir": suite_path(suite_dir, suite["processed"]),
        **{
            key: suite_path(suite_dir, entry[key]) for key in ("mask_dir", "case_dir", "volume_dir")
        },
    }
    validations = []
    for count in counts:
        measured = preflight_relational_trace_graph_run(
            config_path=config_path,
            **paths,
            query_count=count,
            query_limit=limit,
            evaluate_baselines=False,
        )
        validations.append(measured)
        write_benchmark_json(output / f"validation_{count}_queries.json", measured)
        if measured["status"] != "success":
            raise ValueError(f"GNN validation preflight blocked: {measured['blockers']}")
    started = perf_counter()
    domain = load_c3_benchmark_training_graph_domain(**arguments)
    preprocessing = fit_trace_graph_preprocessing(
        domain,
        **config["geometry_features"],
        max_abs_amplitude=config["training_data"].get("max_abs_amplitude"),
    )
    preparation_seconds = perf_counter() - started
    training = measure_c3_first_results_graph_training_batch(
        domain,
        preprocessing,
        model_config=model_config,
        graph_settings=graph,
        training_options=options,
        device=device,
    )
    write_benchmark_json(output / "training_batch.json", training)
    estimates = estimate_c3_first_results_graph_budget(
        training,
        validations[-1],
        max_steps=options["max_steps"],
        validation_interval=options["validation_interval"],
        validation_query_batch_size=options["validation_query_batch_size"],
        diagnostic_baselines="diagnostics" in config,
    )
    return {
        "validation_query_selection": "leading_target_trace_ids_in_sorted_order",
        "sample_coverage_scope": "sampled_queries_only",
        "validation_measurements": validations,
        "training_measurement": training,
        "training_pool_preparation_seconds": preparation_seconds,
        "estimates": estimates,
        "state_reused_by_pilot": False,
    }
