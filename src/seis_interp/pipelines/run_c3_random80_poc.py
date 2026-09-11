"""Run the five fixed PoC methods in isolated processes and compare their records."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from seis_interp.evaluation.c3_poc_comparison import (
    empty_poc_summary_row,
    matching_poc_input_lock,
    read_poc_json,
    validate_poc_run_artifacts,
    write_poc_comparison,
)


def run_poc_process(arguments: list[str], *, stdout_path: Path, stderr_path: Path) -> None:
    """Invoke one CLI process, retaining its output without buffering training logs."""
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        subprocess.run(
            [sys.executable, "-m", "seis_interp.cli", *arguments],
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


def run_c3_random80_poc(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    pocs_config: Path,
    drr_config: Path,
    nersi_config: Path,
    ccnet5d_config: Path,
    gnn_config: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run each method once, preserve failures, and require verified comparable outputs."""
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    data_arguments = [
        argument
        for name, directory in (
            ("interim", interim_dir),
            ("processed", processed_dir),
            ("mask", mask_dir),
            ("case", case_dir),
            ("volume", volume_dir),
        )
        for argument in (f"--{name}", str(Path(directory).absolute()))
    ]
    methods = (
        ("pocs", "pocs", pocs_config),
        ("drr", "drr", drr_config),
        ("nersi", "nersi", nersi_config),
        ("ccnet5d", "ccnet5d", ccnet5d_config),
        ("relational_trace_graph", "relational-trace-graph", gnn_config),
    )
    rows = [empty_poc_summary_row(method, output / method) for method, _, _ in methods]
    summary = {
        "status": "failed",
        "comparison_valid": False,
        "preflight": {"status": "failed"},
        "inputs_lock": None,
        "runs": rows,
        "error_type": None,
        "error_message": None,
    }
    try:
        run_poc_process(
            ["poc", "check", *data_arguments, "--json"],
            stdout_path=logs / "preflight.stdout.log",
            stderr_path=logs / "preflight.stderr.log",
        )
        checked_inputs = read_poc_json(logs / "preflight.stdout.log")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        summary.update(_process_error(error, logs / "preflight.stderr.log"))
        summary["preflight"].update(
            error_type=summary["error_type"], error_message=summary["error_message"]
        )
        write_poc_comparison(output, summary)
        return summary
    summary["preflight"] = {"status": "success", "inputs": checked_inputs}

    for row, (method, command, config) in zip(rows, methods, strict=True):
        try:
            run_poc_process(
                [
                    "interpolate",
                    command,
                    *data_arguments,
                    "--config",
                    str(Path(config).absolute()),
                    "--output",
                    str(output / method),
                    "--json",
                ],
                stdout_path=logs / f"{method}.stdout.log",
                stderr_path=logs / f"{method}.stderr.log",
            )
        except (OSError, subprocess.SubprocessError) as error:
            row.update(status="failed", **_process_error(error, logs / f"{method}.stderr.log"))
            # A CLI may fail before its pipeline creates a run directory.
            (output / method).mkdir(exist_ok=True)
        else:
            row["status"] = "success"

    locks = []
    for index, row in enumerate(rows):
        if row["status"] != "success":
            continue
        try:
            verified_row, inputs_lock = validate_poc_run_artifacts(
                row["method"], Path(row["output_directory"])
            )
            _check_preflight_identity(checked_inputs, inputs_lock)
        except (OSError, ValueError) as error:
            row.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        else:
            rows[index] = verified_row
            locks.append(inputs_lock)

    try:
        summary["inputs_lock"] = matching_poc_input_lock(locks)
    except ValueError as error:
        summary.update(error_type=type(error).__name__, error_message=str(error))
    else:
        if all(row["status"] == "success" for row in rows):
            summary.update(status="success", comparison_valid=True)
    write_poc_comparison(output, summary)
    return summary


def _check_preflight_identity(checked_inputs: dict, inputs_lock: dict) -> None:
    for key in (
        "benchmark_id",
        "dataset_id",
        "case_id",
        "volume_id",
        "selection",
        "shape",
        "observed_trace_count",
        "target_trace_count",
    ):
        if key not in checked_inputs or checked_inputs[key] != inputs_lock.get(key):
            raise ValueError(f"run input lock differs from PoC preflight: {key}")


def _process_error(error: Exception, stderr_path: Path) -> dict[str, str]:
    message = str(error)
    if isinstance(error, subprocess.CalledProcessError) and stderr_path.is_file():
        with stderr_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4096))
            tail = stream.read().decode("utf-8", errors="replace").strip()
        if tail:
            message = f"{message}\n{tail}"
    return {"error_type": type(error).__name__, "error_message": message}
