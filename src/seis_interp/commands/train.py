"""The ``train`` commands: train coordinate-based interpolation models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


def _train_siren(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_siren import train_siren_run
    from seis_interp.run_records import CHECKPOINT_RELATIVE_PATH

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = train_siren_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=progress_reporter,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train siren failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        batch_mode = summary.get("batch_mode", "random_points")
        print(f"Output directory: {args.output}")
        print(f"Batch mode: {batch_mode}")
        if "amplitude_scaling" in summary:
            print(f"Amplitude scaling: {summary['amplitude_scaling']}")
        if summary.get("loss_semantics") == "mse_plus_trace_correlation":
            print(
                "Correlation loss: "
                f"weight={summary['correlation_weight']}, eps={summary['correlation_eps']}"
            )
        trace_quality = summary.get("trace_quality")
        if isinstance(trace_quality, Mapping):
            print(f"Excluded traces: {trace_quality['excluded_trace_count']}")
        oracle_validation = summary.get("validation_metric_domain") == ("oracle_per_trace_unit_rms")
        if oracle_validation:
            print("Validation metric domain: oracle per-trace unit RMS")
        print(f"Best epoch: {summary['best_epoch']}")
        if batch_mode in ("full_ffid_epoch", "random_complete_traces"):
            label = (
                "Best oracle-normalized validation global S/N"
                if oracle_validation
                else "Best validation global S/N"
            )
            print(f"{label}: {summary['best_validation_global_snr_db']} dB")
            print(f"Optimizer steps: {summary['global_steps']}")
        else:
            median_label = (
                "Best oracle-normalized validation median trace S/N"
                if oracle_validation
                else "Best validation median trace S/N"
            )
            print(f"{median_label}: {summary['best_validation_median_trace_snr_db']} dB")
            global_label = (
                "Oracle-normalized global validation S/N at best epoch"
                if oracle_validation
                else "Global validation S/N at best epoch"
            )
            print(f"{global_label}: {summary['best_validation_global_snr_db']} dB")
        print(f"Epochs completed: {summary['epochs_completed']}")
        print(f"Stopped early: {summary['stopped_early']}")
        print(f"Checkpoint: {args.output / CHECKPOINT_RELATIVE_PATH}")
    return 0


def _train_neighbor_inpainter(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_neighbor_inpainter import train_neighbor_inpainter_run
    from seis_interp.run_records import CHECKPOINT_RELATIVE_PATH

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = train_neighbor_inpainter_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=progress_reporter,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train neighbor-inpainter failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Output directory: {args.output}")
        print("Amplitude scaling: per_trace_rms")
        print("Validation metric domain: oracle per-trace unit RMS")
        print(f"Best step: {summary['best_step']}")
        print(
            "Best oracle per-trace unit-RMS global S/N: "
            f"{summary['oracle_per_trace_unit_rms_global_snr_db']} dB"
        )
        print(f"Success threshold: > {summary['success_threshold_db']} dB")
        print(f"Metric success: {summary['metric_success']}")
        print(f"Formal scope success: {summary['scope_success']}")
        print(f"Success: {summary['success']}")
        print(f"Checkpoint: {args.output / CHECKPOINT_RELATIVE_PATH}")
    return 0


def _train_shot_gather_inpainter(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_shot_gather_inpainter import (
        train_shot_gather_inpainter_run,
    )
    from seis_interp.run_records import CHECKPOINT_RELATIVE_PATH

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = train_shot_gather_inpainter_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=progress_reporter,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train shot-gather-inpainter failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Output directory: {args.output}")
        print("Amplitude scaling: per_trace_rms")
        print("Validation metric domain: oracle per-trace unit RMS")
        print(f"Best step: {summary['best_step']}")
        print(
            "Best oracle per-trace unit-RMS global S/N: "
            f"{summary['oracle_per_trace_unit_rms_global_snr_db']} dB"
        )
        print(f"Success threshold: > {summary['success_threshold_db']} dB")
        print(f"Metric success: {summary['metric_success']}")
        print(f"Formal scope success: {summary['scope_success']}")
        print(f"Success: {summary['success']}")
        print(f"Checkpoint: {args.output / CHECKPOINT_RELATIVE_PATH}")
    return 0


def _train_trace_graph(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_trace_graph import train_trace_graph_run
    from seis_interp.run_records import CHECKPOINT_RELATIVE_PATH

    progress_reporter = _print_progress_to_stderr if args.json else None
    try:
        summary = train_trace_graph_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=progress_reporter,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train trace-graph failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Output directory: {args.output}")
        print("Amplitude scaling: per_trace_rms")
        print("Validation metric domain: oracle per-trace unit RMS")
        print(f"Best step: {summary['best_step']}")
        print(
            "Best oracle per-trace unit-RMS global S/N: "
            f"{summary['oracle_per_trace_unit_rms_global_snr_db']} dB"
        )
        print(f"Success threshold: > {summary['success_threshold_db']} dB")
        print(f"Metric success: {summary['metric_success']}")
        print(f"Formal scope success: {summary['scope_success']}")
        print(f"Success: {summary['success']}")
        print(f"Checkpoint: {args.output / CHECKPOINT_RELATIVE_PATH}")
    return 0


def _train_ccnet5d(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_ccnet5d import train_ccnet5d_run

    try:
        summary = train_ccnet5d_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            device_override=args.device,
            progress_reporter=_print_progress_to_stderr,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"train ccnet5d failed: {error}", file=sys.stderr)
        return 1

    for warning in summary.get("warnings", []):
        print(f"Warning: {warning}", file=sys.stderr)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        selection = summary["best_selection_metrics"]
        score = (
            f"{selection['snr_db']} dB"
            if selection["snr_status"] == "finite"
            else selection["snr_status"].replace("_", " ")
        )
        print(f"Output directory: {args.output}")
        print(f"Completed steps: {summary['steps_completed']}")
        print(f"Best step: {summary['best_step']}")
        print(f"Best internal selection global S/N: {score}")
    return 0


def _print_progress_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _train_relational_trace_graph(args: argparse.Namespace) -> int:
    from seis_interp.pipelines.train_relational_trace_graph import train_relational_trace_graph_run

    try:
        summary = train_relational_trace_graph_run(
            config_path=args.config,
            interim_dir=args.interim,
            processed_dir=args.processed,
            output_dir=args.output,
            validation_mask_dir=args.validation_mask,
            validation_case_dir=args.validation_case,
            validation_volume_dir=args.validation_volume,
            train_mask_dir=args.train_mask,
            train_case_dir=args.train_case,
            train_volume_dir=args.train_volume,
            device_override=args.device,
            progress_reporter=_print_progress_to_stderr,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"train relational-trace-graph failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Output directory: {args.output}")
        print(f"Completed steps: {summary['steps_completed']}")
        print(f"Best step: {summary['best_step']}")
        selection = summary["best_validation_metrics"]["evaluation_target"]
        print(f"Best validation SSE: {selection['error_energy']:.4f}")
    return 0


def add_train_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train = subparsers.add_parser("train", help="Train coordinate-based interpolation models.")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    siren = train_commands.add_parser("siren", help="Train a SIREN on prepared trace splits.")
    siren.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    siren.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    siren.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    siren.add_argument("--output", type=Path, required=True, help="Run output directory.")
    siren.add_argument("--device", help="Override training.device for this environment.")
    siren.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    siren.set_defaults(handler=_train_siren)
    neighbor = train_commands.add_parser(
        "neighbor-inpainter",
        help="Train the physical-neighbor temporal trace inpainter.",
    )
    neighbor.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    neighbor.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    neighbor.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    neighbor.add_argument("--output", type=Path, required=True, help="Run output directory.")
    neighbor.add_argument("--device", help="Override training.device for this environment.")
    neighbor.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    neighbor.set_defaults(handler=_train_neighbor_inpainter)
    shot_gather = train_commands.add_parser(
        "shot-gather-inpainter",
        help="Train the joint whole-shot gather inpainter.",
    )
    shot_gather.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    shot_gather.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    shot_gather.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Prepared split dataset.",
    )
    shot_gather.add_argument("--output", type=Path, required=True, help="Run output directory.")
    shot_gather.add_argument("--device", help="Override training.device for this environment.")
    shot_gather.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    shot_gather.set_defaults(handler=_train_shot_gather_inpainter)
    trace_graph = train_commands.add_parser(
        "trace-graph",
        help="Train the trace-node graph gather interpolator.",
    )
    trace_graph.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    trace_graph.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    trace_graph.add_argument(
        "--processed",
        type=Path,
        required=True,
        help="Prepared split dataset.",
    )
    trace_graph.add_argument("--output", type=Path, required=True, help="Run output directory.")
    trace_graph.add_argument("--device", help="Override training.device for this environment.")
    trace_graph.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    trace_graph.set_defaults(handler=_train_trace_graph)
    ccnet5d = train_commands.add_parser(
        "ccnet5d", help="Pretrain CCNet5D on complete labels from the train partition."
    )
    ccnet5d.add_argument("--config", type=Path, required=True, help="Study configuration YAML.")
    ccnet5d.add_argument("--interim", type=Path, required=True, help="Interim trace dataset.")
    ccnet5d.add_argument("--processed", type=Path, required=True, help="Prepared split dataset.")
    ccnet5d.add_argument("--output", type=Path, required=True, help="Run output directory.")
    ccnet5d.add_argument("--device", help="Override training.device for this environment.")
    ccnet5d.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    ccnet5d.set_defaults(handler=_train_ccnet5d)
    relational = train_commands.add_parser(
        "relational-trace-graph", help="Train a grid-free trace graph with fixed validation."
    )
    for option, help_text in (
        ("config", "Study configuration YAML."),
        ("interim", "Interim trace dataset."),
        ("processed", "Prepared split dataset."),
        ("output", "New run output directory."),
        ("validation-mask", "Fixed validation interpolation mask."),
        ("validation-case", "Fixed validation benchmark case."),
    ):
        relational.add_argument(f"--{option}", type=Path, required=True, help=help_text)
    for option in ("validation-volume", "train-mask", "train-case", "train-volume"):
        relational.add_argument(f"--{option}", type=Path, help="Optional input selection artifact.")
    relational.add_argument("--device", help="Override training.device for this environment.")
    relational.add_argument("--json", action="store_true", help="Print metrics as strict JSON.")
    relational.set_defaults(handler=_train_relational_trace_graph)
