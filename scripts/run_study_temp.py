"""Run one throwaway experiment from studies/study_temp/config.yaml, overwriting the output.

Unlike the numbered study scripts, this driver enforces no study contract: edit the config
and re-run as often as needed. Metrics land in runs/study_temp/metrics.json (overwritten).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from seis_interp.configuration import get_required_config_value, load_resolved_config
from seis_interp.pipelines import batching_ablation as pipeline
from seis_interp.training.point_sampler import build_trace_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("studies/study_temp/config.yaml"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim/c3_na/ffid_2348"))
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/c3_na/ffid_2348_random_split"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/study_temp/metrics.json"))
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_resolved_config(args.config)
    experiment = config.get("experiment", {})
    training = config["training"]

    data = pipeline._load_experiment_data(args.interim, args.processed, config)
    rows = np.sort(data["training_array_rows"]).astype(np.int64, copy=False)
    trace_count = int(experiment.get("trace_count", len(rows)))
    rows = rows[:trace_count]

    amplitudes = data["normalized_amplitudes"]
    scaling = experiment.get("amplitude_scaling", "global_rms")
    if scaling == "per_trace_rms":
        amplitudes, _ = pipeline.per_trace_rms_scaled_amplitudes(amplitudes, rows)
    elif scaling != "global_rms":
        raise ValueError(f"unknown amplitude_scaling: {scaling!r}")
    amplitude_gain = float(experiment.get("amplitude_gain", 1.0))
    if amplitude_gain != 1.0:
        amplitudes = amplitudes * np.asarray(amplitude_gain, dtype=amplitudes.dtype)

    coordinates, targets = build_trace_points(
        data["normalized_time"],
        data["normalized_spatial_by_array_row"],
        amplitudes,
        rows,
    )

    device = str(training["device"])
    batch_mode = experiment.get("batch_mode", "random_replacement")
    traces_per_update = None
    if batch_mode == "exact_full_batch":
        full_batch = True
        replacement = False
        batch_size = len(targets)
        all_coordinate_tensor, all_target_tensor = pipeline.to_model_tensors(
            coordinates, targets, device=device
        )
    elif batch_mode == "random_replacement":
        full_batch = False
        replacement = True
        batch_size = int(training["batch_size"])
        all_coordinate_tensor = all_target_tensor = None
    elif batch_mode == "random_complete_traces":
        full_batch = False
        replacement = False
        traces_per_update = int(experiment["traces_per_update"])
        batch_size = traces_per_update * data["sample_count"]
        all_coordinate_tensor = all_target_tensor = None
    else:
        raise ValueError(f"unsupported batch_mode for study_temp: {batch_mode!r}")

    metrics = pipeline.run_training_fit_condition(
        config=config,
        label="study_temp",
        batch_mode=batch_mode,
        full_batch=full_batch,
        replacement=replacement,
        total_updates=int(training["total_updates"]),
        report_interval=int(training["report_interval"]),
        batch_size=batch_size,
        normalized_time=data["normalized_time"],
        normalized_spatial_by_array_row=data["normalized_spatial_by_array_row"],
        normalized_amplitudes=amplitudes,
        selected_array_rows=rows,
        all_coordinate_tensor=all_coordinate_tensor,
        all_target_tensor=all_target_tensor,
        training_coordinates=coordinates,
        training_targets=targets,
        sample_count=data["sample_count"],
        prediction_batch_size=int(training["prediction_batch_size"]),
        device=device,
        random_seed=int(get_required_config_value(config, "project.random_seed")),
        traces_per_update=traces_per_update,
        correlation_weight=float(training.get("correlation_weight", 0.0)),
        correlation_eps=float(training.get("correlation_eps", pipeline._STUDY_005_CORRELATION_EPS)),
        loss_name=str(training.get("loss", "l2")),
        huber_delta=training.get("huber_delta"),
    )
    metrics["amplitude_scaling"] = scaling
    metrics["amplitude_gain"] = amplitude_gain

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"study_temp run failed: {error}", file=sys.stderr)
        return 1
    print(
        f"classification={metrics['classification']} best_step={metrics['best_step']} "
        f"best_median_snr_db={metrics['best_training_median_trace_snr_db']:.4f} "
        f"final_median_snr_db={metrics['final_training_median_trace_snr_db']:.4f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
