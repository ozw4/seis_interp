"""Run each bounded memory/shape check in its own process, without evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_model_inputs,
    verify_hashes,
    verify_mvp_implementation,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity, peak_memory, utc_now
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.training.forge_mvp_preflight import check_method_shapes


def preflight_forge_method(repo: Path, preparation: Path, output: Path, method_id: str):
    started = perf_counter()
    record = {
        "method_id": method_id,
        "start_time": utc_now(),
        "status": "checking",
        **execution_identity(repo),
    }
    device = "cpu"
    try:
        seal = json.loads((preparation / "preparation_manifest.json").read_text())
        verify_mvp_implementation(repo, seal["implementation_hashes"])
        inputs, config = load_model_inputs(preparation, method_id)
        device = config["device"]
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        record.update(check_method_shapes(inputs, config))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        record.update(
            status="passed",
            config_hash=sha256_file(preparation / "configs" / f"{method_id}.yaml"),
            mask_hash=seal["mask_hash"],
            test_amplitudes_read=0,
        )
    except Exception as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(
            end_time=utc_now(), runtime_seconds=perf_counter() - started, **peak_memory(device)
        )
        write_json(output, record)


def preflight_forge_mvp(repo: Path, preparation: Path, output: Path):
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    verify_mvp_implementation(repo, seal["implementation_hashes"])
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    records = {}
    for method in METHODS:
        print(f"Checking full-time input and kernel shape: {method}", flush=True)
        with (output / f"{method}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts/forge_m1_mvp.py"),
                    "check-method",
                    "--preparation",
                    str(preparation),
                    "--output",
                    str(output / f"{method}.json"),
                    "--method",
                    method,
                ],
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = output / f"{method}.json"
        records[method] = (
            json.loads(path.read_text())
            if path.is_file()
            else {"status": "failed", "failure_reason": "worker exited without report"}
        )
    complete = all(r["status"] == "passed" for r in records.values())
    if (
        complete
        and records["regsi_real"]["initialization_hash"]
        != records["regsi_grid"]["initialization_hash"]
    ):
        raise ValueError("ReGSI initial states differ")
    report = {
        "status": "passed" if complete else "failed",
        "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
        "methods": records,
        "test_amplitudes_read": 0,
        "production_runs_started": 0,
        "quality_metrics_computed": False,
        "scope": "bounded checks; complete-run resource peaks remain unmeasured",
        "artifacts": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "preflight_manifest.json", report)
    return report


def validate_preflight(preparation: Path, preflight: Path):
    report = json.loads((preflight / "preflight_manifest.json").read_text())
    verify_hashes(preflight, report["artifacts"])
    if (
        report["preparation_hash"] != sha256_file(preparation / "preparation_manifest.json")
        or report["status"] != "passed"
        or set(report["methods"]) != set(METHODS)
    ):
        raise ValueError("production requires a passing preflight for this preparation")
    for method, record in report["methods"].items():
        if record["status"] != "passed" or record["config_hash"] != sha256_file(
            preparation / "configs" / f"{method}.yaml"
        ):
            raise ValueError("preflight method/config mismatch")
    return sha256_file(preflight / "preflight_manifest.json")
