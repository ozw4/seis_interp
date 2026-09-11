"""Run exactly one frozen-suite pilot action with immutable outer execution records."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from seis_interp import run_records
from seis_interp.configuration import REPOSITORY_ROOT, ConfigurationError
from seis_interp.data.c3_benchmark_artifacts import (
    read_benchmark_json,
    write_benchmark_json,
    write_benchmark_yaml,
)
from seis_interp.data.c3_first_result_inputs import (
    build_c3_first_result_native_config,
    read_first_results_yaml,
    resolve_c3_first_result_inputs,
    validate_c3_first_result_training_inputs,
)
from seis_interp.data.file_checksums import file_sha256
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions

ACTIONS = (
    "check",
    "zero-fill",
    "siren",
    "summarize",
)


def run_c3_first_results(
    *,
    config_path: Path,
    inputs_path: Path,
    action: str,
    execute: bool = False,
    checkpoint_path: Path | None = None,
    case_id: str | None = None,
    preflight: bool = False,
    method_runs: dict[str, Path] | None = None,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict:
    """Resolve read-only by default; explicit execution isolates a single timed action."""
    if action not in ACTIONS:
        raise ConfigurationError(f"unsupported action: {action}")
    if preflight and action not in ("siren",):
        raise ConfigurationError(f"--preflight is not supported for {action}")
    config_path, inputs_path = Path(config_path).resolve(), Path(inputs_path).resolve()
    plan = read_first_results_yaml(config_path)
    _validate_execution(plan["execution"])
    inputs = read_first_results_yaml(inputs_path)
    started = time.perf_counter()
    request = {
        "action": action,
        "preflight": preflight,
        "started_at": run_records.utc_timestamp(),
        "config_path": str(config_path),
        "inputs_path": str(inputs_path),
        "experiment_config": plan,
        "experiment_inputs": inputs,
        "seeds": plan["seeds"],
        "case_id": case_id or inputs["required_cases"][0],
        "suite_manifest": str((inputs_path.parent / inputs["frozen_suite"]["manifest"]).resolve()),
        "expected_suite_sha256": inputs["frozen_suite"]["expected_sha256"],
        "requested_checkpoint_path": str(Path(checkpoint_path).resolve())
        if checkpoint_path
        else None,
        "dimensions": {
            "time_range": list(dimensions.time_range),
            "sail_line_numbers": list(dimensions.sail_line_numbers),
            "shape": list(dimensions.shape),
        },
        "method_runs": {
            key: str(Path(value).resolve()) for key, value in (method_runs or {}).items()
        },
        **run_records.current_git_metadata(),
    }
    output = None
    if execute:
        root = (inputs_path.parent / inputs["outputs"]["runs"]).resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = root / f"{stamp}_{request['git_commit'][:12]}_{action}"
        output.mkdir(parents=True, exist_ok=False)
        request["native_run_directory"] = str(output / "native")
    try:
        if not execute:
            _resolve_request(
                request,
                config_path=config_path,
                inputs_path=inputs_path,
                checkpoint_path=checkpoint_path,
                dimensions=dimensions,
            )
            return {"status": "dry_run", "request": request, "writes": False}
        write_benchmark_json(output / "request.planned.json", request)
        result = _execute_worker(request, output)
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, TypeError) as error:
        result = {
            "status": "blocked",
            "reason": str(error),
            "error_type": type(error).__name__,
            "stage": "input_resolution",
        }
    except KeyboardInterrupt:
        result = {"status": "interrupted", "reason": "interrupted during input resolution"}
    if output is not None:
        if not (output / "request.json").exists():
            write_benchmark_json(output / "request.json", request)
        result.update(
            action=action,
            preflight=request["preflight"],
            outer_run_directory=str(output),
            native_run_directory=request["native_run_directory"],
            elapsed_seconds=time.perf_counter() - started,
            finished_at=run_records.utc_timestamp(),
        )
        write_benchmark_json(output / "result.json", result)
    else:
        result.update(action=action, writes=False, request=request)
    return result


def _resolve_request(request, *, config_path, inputs_path, checkpoint_path, dimensions):
    binding = resolve_c3_first_result_inputs(
        inputs_path, case_id=request["case_id"], dimensions=dimensions
    )
    action = request["action"]
    plan = request["experiment_config"]
    request["input_hashes"] = binding.input_hashes
    request["paths"] = {key: str(path) for key, path in binding.paths.items()}
    request["case"] = binding.entry
    request["volume"] = binding.volume
    request["train_pool"] = binding.manifest["train_pool"]
    if checkpoint_path is not None:
        raise ConfigurationError("--checkpoint is not supported for these actions")
    if action in ("check", "zero-fill", "summarize"):
        request["native_config"] = plan
        return
    method = action
    fragment_path = config_path.parent / plan["methods"][method]["native_fragment"]
    fragment = read_first_results_yaml(fragment_path)
    native = build_c3_first_result_native_config(
        action, fragment=fragment, binding=binding, seeds=plan["seeds"]
    )
    device_section = native.get("training", native.get("prediction", {}))
    if "device" in device_section and device_section["device"] != plan["execution"]["device"]:
        raise ConfigurationError("method device differs from execution.device")
    request["native_config"] = native
    request["native_fragment_sha256"] = file_sha256(fragment_path)
    request["training_inputs"] = validate_c3_first_result_training_inputs(
        action,
        binding=binding,
        config=native,
        dimensions=dimensions,
    )


def _validate_execution(execution: dict) -> None:
    for key in ("action_timeout_seconds", "preflight_timeout_seconds"):
        value = execution[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ConfigurationError(f"execution.{key} must be positive and finite")
    threads = execution.get("thread_count", 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ConfigurationError("execution.thread_count must be a positive integer")


def _execute_worker(request: dict, output: Path) -> dict:
    execution = request["experiment_config"]["execution"]
    key = "preflight_timeout_seconds" if request["preflight"] else "action_timeout_seconds"
    timeout = execution[key]
    threads = execution.get("thread_count", 1)
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = str(threads)
    declared_environment = execution.get("environment", {})
    if not isinstance(declared_environment, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in declared_environment.items()
    ):
        raise ConfigurationError("execution.environment must map strings to strings")
    environment.update(declared_environment)
    print(f"Executing {request['action']} in a fresh process; timeout={timeout}s", file=sys.stderr)
    try:
        with (
            (output / "worker.stdout.log").open("w") as stdout,
            (output / "worker.stderr.log").open("w") as stderr,
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "seis_interp.pipelines.c3_first_results",
                    "--worker",
                    str(output / "request.planned.json"),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": f"action exceeded {timeout} seconds"}
    except KeyboardInterrupt:
        return {"status": "interrupted", "reason": "action interrupted"}
    worker_result = output / "worker.result.json"
    if worker_result.exists():
        result = read_benchmark_json(worker_result)
        if completed.returncode and result.get("status") == "success":
            result.update(status="failed", reason=f"worker exited {completed.returncode}")
        return result
    return {"status": "failed", "reason": f"worker exited {completed.returncode} without a result"}


def dispatch_c3_first_results_action(request: dict) -> dict:
    """Call a public native run API using the exact paths verified by the outer reader."""
    action = request["action"]
    if action not in ACTIONS:
        raise ConfigurationError(f"unsupported action: {action}")
    paths = {key: Path(value) for key, value in request["paths"].items()}
    config_path = Path(request["native_config_path"])
    output = Path(request["native_run_directory"])
    suite_dir = Path(request["suite_manifest"]).parent
    dimensions = C3BenchmarkDimensions(
        **{key: tuple(value) for key, value in request["dimensions"].items()}
    )
    if action in ("check", "zero-fill"):
        from seis_interp.pipelines.check_c3_first_results import (
            check_c3_first_results,
            zero_fill_c3_first_results,
        )

        function = check_c3_first_results if action == "check" else zero_fill_c3_first_results
        return function(
            suite_dir=suite_dir,
            case_id=request["case_id"],
            output_dir=output,
            config=request["native_config"],
            dimensions=dimensions,
            expected_sha256=request["input_hashes"]["suite"],
        )
    if action == "summarize":
        from seis_interp.evaluation.c3_first_results import summarize_c3_first_results

        methods = {
            name: None for name in ("pocs", "drr", "siren5d", "ccnet5d", "relational_trace_graph")
        }
        methods.update({key: Path(value) for key, value in request["method_runs"].items()})
        return summarize_c3_first_results(
            suite_path=Path(request["suite_manifest"]),
            case_id=request["case_id"],
            method_runs=methods,
            output_dir=output,
            dimensions=dimensions,
        )
    if request["preflight"]:
        from seis_interp.pipelines.preflight_c3_first_results_neural import (
            run_c3_first_results_neural_preflight,
        )

        return run_c3_first_results_neural_preflight(
            action,
            config_path=config_path,
            suite_dir=suite_dir,
            case_id=request["case_id"],
            output_dir=output,
            dimensions=dimensions,
        )
    arguments = {"config_path": config_path, "output_dir": output, "progress_reporter": _progress}
    if action == "siren":
        from seis_interp.pipelines.interpolate_siren import interpolate_siren_run

        return interpolate_siren_run(**arguments, **paths)
    raise ConfigurationError(f"unsupported action: {action}")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _worker(request_path: Path) -> int:
    request = read_benchmark_json(request_path)
    started = time.perf_counter()
    stage = "input_resolution"
    try:
        checkpoint = request["requested_checkpoint_path"]
        dimensions = C3BenchmarkDimensions(
            **{key: tuple(value) for key, value in request["dimensions"].items()}
        )
        _resolve_request(
            request,
            config_path=Path(request["config_path"]),
            inputs_path=Path(request["inputs_path"]),
            checkpoint_path=Path(checkpoint) if checkpoint else None,
            dimensions=dimensions,
        )
        request["native_config_path"] = str(request_path.parent / "config.generated.yaml")
        write_benchmark_yaml(Path(request["native_config_path"]), request["native_config"])
        write_benchmark_json(request_path.parent / "request.json", request)
        stage = "native_action"
        metrics = dispatch_c3_first_results_action(request)
        status = metrics.get("status", "success")
        status = (
            status
            if status in ("blocked", "failed", "not_run", "timeout", "interrupted")
            else "success"
        )
        result = {"status": status, "metrics": metrics}
        if status != "success":
            result["reason"] = metrics.get("reason", metrics.get("blockers", status))
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, TypeError) as error:
        result = {
            "status": "blocked" if stage == "input_resolution" else "failed",
            "reason": str(error),
            "error_type": type(error).__name__,
            "stage": stage,
        }
    except KeyboardInterrupt:
        result = {"status": "interrupted", "reason": "worker interrupted", "stage": stage}
    if not (request_path.parent / "request.json").exists():
        write_benchmark_json(request_path.parent / "request.json", request)
    result["worker_elapsed_seconds"] = time.perf_counter() - started
    write_benchmark_json(request_path.parent / "worker.result.json", result)
    return 0 if result["status"] == "success" else 1


def main(argv: list[str] | None = None) -> int:
    """Keep JSON results on stdout and progress on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "studies/study_028_c3_first_results/config.yaml",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=REPOSITORY_ROOT / "studies/study_028_c3_first_results/inputs.yaml",
    )
    parser.add_argument("--action", choices=ACTIONS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--method-run", action="append", default=[], metavar="METHOD=PATH")
    args = parser.parse_args(argv)
    if args.worker is not None:
        return _worker(args.worker)
    if args.action is None:
        parser.error("--action is required")
    method_runs = {}
    for value in args.method_run:
        name, separator, path = value.partition("=")
        if not separator or not path or name in method_runs:
            parser.error("--method-run requires unique METHOD=PATH entries")
        method_runs[name] = Path(path)
    try:
        result = run_c3_first_results(
            config_path=args.config,
            inputs_path=args.inputs,
            action=args.action,
            execute=args.execute,
            checkpoint_path=args.checkpoint,
            case_id=args.case_id,
            preflight=args.preflight,
            method_runs=method_runs,
        )
    except (OSError, ValueError, RuntimeError) as error:
        result = {"status": "blocked", "reason": str(error), "error_type": type(error).__name__}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in ("success", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
