"""Run the five methods in separate processes against the pinned v3 inputs."""

import subprocess
import tarfile
from pathlib import Path

from seis_interp.configuration import load_resolved_config
from seis_interp.data.c3_benchmark_artifacts import write_benchmark_yaml
from seis_interp.data.c3_poc_inputs import load_c3_random80_v3_inputs
from seis_interp.data.file_checksums import file_sha256
from seis_interp.evaluation.c3_v3_comparison import validate_v3_run_artifacts, write_v3_comparison
from seis_interp.pipelines.run_c3_random80_poc import run_poc_process

METHODS = ("pocs", "drr", "nersi", "ccnet5d", "relational_trace_graph")


def run_c3_v3_comparison(
    *, input_paths: dict[str, Path], configs: dict[str, Path], output_dir: Path
) -> dict:
    if set(configs) != set(METHODS):
        raise ValueError("v3 comparison requires exactly five method configs")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "configs").mkdir()
    rows = [
        dict(method=method, status="not_run", output_directory=str(output / method))
        for method in METHODS
    ]
    summary = dict(
        condition_id="c3_random80_v3",
        primary_metric="mean_trace_snr_db",
        status="failed",
        comparison_valid=False,
        independent_evaluation=False,
        equal_compute_budget=False,
        inputs_lock=None,
        runs=rows,
    )
    try:
        resolved = {method: load_resolved_config(path) for method, path in configs.items()}
        first = resolved["pocs"]
        for config in resolved.values():
            for key in (
                "input_protocol",
                "input_conditions_lock",
                "input_conditions_sha256",
                "benchmark_case",
                "benchmark_volume",
                "evaluation",
            ):
                if config.get(key) != first.get(key):
                    raise ValueError(f"method configs disagree on {key}")
        if first.get("input_protocol") != "c3_random80_v3":
            raise ValueError("v3 input protocol is required")
        verified = load_c3_random80_v3_inputs(config=first, **input_paths)
        summary["inputs_lock"] = verified.inputs_lock
        del verified
        for method, config in resolved.items():
            write_benchmark_yaml(output / "configs" / f"{method}.yaml", config)
        archive_path = output / "source.tar"
        with tarfile.open(archive_path, "x") as archive:
            for path in sorted(Path("src/seis_interp").rglob("*.py")):
                archive.add(path)
            for path in sorted((output / "configs").glob("*.yaml")):
                archive.add(path, arcname="configs/" + path.name)
            archive.add(Path(first["input_conditions_lock"]), arcname="v3_conditions.lock.json")
        summary["source_archive_sha256"] = file_sha256(archive_path)
    except (OSError, ValueError, KeyError) as error:
        summary.update(error_type=type(error).__name__, error_message=str(error))
        write_v3_comparison(output, summary)
        return summary
    arguments = [
        item
        for key, path in input_paths.items()
        for item in ("--" + key.removesuffix("_dir"), str(Path(path).absolute()))
    ]
    for row in rows:
        method = row["method"]
        print(f"Starting {method}: {output / method}", flush=True)
        try:
            run_poc_process(
                [
                    "interpolate",
                    method.replace("_", "-"),
                    *arguments,
                    "--config",
                    str(output / "configs" / f"{method}.yaml"),
                    "--output",
                    str(output / method),
                    "--json",
                ],
                stdout_path=output / "logs" / f"{method}.stdout.log",
                stderr_path=output / "logs" / f"{method}.stderr.log",
            )
            actual_config = load_resolved_config(output / method / "config.resolved.yaml")
            if actual_config != resolved[method]:
                raise ValueError("run config differs from the pinned config snapshot")
            row.update(validate_v3_run_artifacts(method, output / method, summary["inputs_lock"]))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            row.update(status="failed", error_type=type(error).__name__, error_message=str(error))
            (output / method).mkdir(exist_ok=True)
        write_v3_comparison(output, summary)
        print(f"Finished {method}: {row['status']}", flush=True)
    if all(row["status"] == "success" for row in rows):
        summary.update(status="success", comparison_valid=True)
    write_v3_comparison(output, summary)
    return summary
