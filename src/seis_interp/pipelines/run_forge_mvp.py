"""Execute frozen methods in isolated processes, then open evaluation targets."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import (
    load_model_inputs,
    verify_hashes,
    verify_mvp_implementation,
    write_json,
)
from seis_interp.data.forge_mvp_run_records import execution_identity, peak_memory, utc_now
from seis_interp.evaluation.forge_mvp_metrics import validate_predictions
from seis_interp.pipelines.preflight_forge_mvp import validate_preflight
from seis_interp.processing.forge_mvp_contract import METHODS
from seis_interp.training.forge_mvp_methods import run_method

POLL_SECONDS = 5


def run_forge_method(repo: Path, preparation: Path, output: Path, method_id: str):
    output.mkdir(parents=True, exist_ok=False)
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    config_path = preparation / "configs" / f"{method_id}.yaml"
    config = yaml.safe_load(config_path.read_text())
    mask = pd.read_parquet(preparation / "masks/m1_random80_mvp_v1.parquet")
    manifest = {
        key: seal[key]
        for key in (
            "study_id",
            "candidate_id",
            "mask_id",
            "mask_seed",
            "input_hashes",
            "mapping_hash",
            "mask_hash",
            "global_rms",
            "observed_trace_count",
            "test_trace_count",
            "clean_test_trace_count",
            "time_sample_count",
        )
    }
    if "trace_scale_hash" in seal:
        manifest.update(
            {
                key: seal[key]
                for key in ("amplitude_scaling", "trace_scale_hash", "idw")
                if key in seal
            }
        )
    manifest.update(execution_identity(repo))
    manifest.update(
        method_id=method_id,
        geometry_mode=config["geometry_mode"],
        model_seed=config["model_seed"],
        config_hash=sha256_file(config_path),
        checkpoint_hash=None,
        prediction_hash=None,
        start_time=utc_now(),
        end_time=None,
        status="running",
        failure_reason=None,
        preparation_hash=sha256_file(preparation / "preparation_manifest.json"),
    )
    write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_bytes(config_path.read_bytes())
    started = perf_counter()
    try:
        verify_mvp_implementation(repo, seal["implementation_hashes"])
        inputs, config = load_model_inputs(preparation, method_id)
        if config["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(config["device"])
        result = run_method(inputs, config)
        if config["device"].startswith("cuda"):
            torch.cuda.synchronize(config["device"])
        order = validate_predictions(
            result.prediction, result.cell_ids, mask, seal["time_sample_count"]
        )
        np.save(output / "prediction.npy", result.prediction[order], allow_pickle=False)
        index = mask.loc[mask.split.eq("test")].copy().reset_index(drop=True)
        index["prediction_row"] = np.arange(len(index))
        index.to_parquet(output / "prediction_index.parquet", index=False)
        pd.DataFrame(
            result.history, columns=None if result.history else ["step", "loss"]
        ).to_parquet(output / "training_log.parquet", index=False)
        if result.checkpoint is not None:
            (output / "checkpoint").mkdir()
            torch.save(result.checkpoint, output / "checkpoint/final.pt")
            manifest["checkpoint_hash"] = sha256_file(output / "checkpoint/final.pt")
        manifest.update(result.diagnostics)
        manifest.update(
            status="predicted",
            prediction_hash=sha256_file(output / "prediction.npy"),
            prediction_index_hash=sha256_file(output / "prediction_index.parquet"),
            amplitude_units="original",
        )
    except Exception as error:
        manifest.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(
            end_time=utc_now(),
            runtime_seconds=perf_counter() - started,
            **peak_memory(config["device"]),
        )
        write_json(output / "run_manifest.json", manifest)


def run_forge_mvp(
    repo: Path, preparation: Path, output: Path, preflight: Path, *, concurrent: bool = False
):
    """Run every frozen method in its own process, then evaluate once.

    ``concurrent`` runs CPU methods and different CUDA devices at the same time,
    while methods configured on the same CUDA device still run one after another
    in the fixed method order. Predictions and metrics are unchanged; per-method
    runtimes then share the machine and are recorded as not comparable.
    """
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    verify_mvp_implementation(repo, seal["implementation_hashes"])
    preflight_hash = validate_preflight(preparation, preflight)
    queues = _execution_queues(preparation, concurrent)
    output.mkdir(parents=True, exist_ok=False)
    identity = execution_identity(repo)
    write_json(
        output / "matrix_manifest.json",
        {
            "preparation": str(preparation.relative_to(repo)),
            "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
            "methods": list(METHODS),
            "start_time": utc_now(),
            "preflight_hash": preflight_hash,
            **identity,
            **(
                {
                    "execution": "concurrent_by_device",
                    "execution_queues": queues,
                    "runtime_comparable": False,
                }
                if concurrent
                else {}
            ),
        },
    )
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )

    def start(method):
        print(f"Starting fixed MVP method: {method}", flush=True)
        log = (output / f"{method}.log").open("w")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(repo / "scripts/forge_m1_mvp.py"),
                    "method",
                    "--preparation",
                    str(preparation),
                    "--output",
                    str(output / "runs" / method),
                    "--method",
                    method,
                ],
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            log.close()
            raise
        return method, process, log

    pending = {key: list(methods) for key, methods in queues.items()}
    running = {}
    try:
        for key, methods in pending.items():
            running[key] = start(methods.pop(0))
        while running:
            for key, (method, process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                _finish_method(method, process, log)
                del running[key]
                if pending[key]:
                    running[key] = start(pending[key].pop(0))
            if running:
                sleep(POLL_SECONDS)
    except BaseException:
        # Never leave workers training for a matrix that will not be evaluated.
        _stop_methods(running.values())
        raise
    from seis_interp.pipelines.evaluate_forge_mvp import evaluate_forge_mvp

    return evaluate_forge_mvp(repo, preparation, output)


def _execution_queues(preparation, concurrent):
    """One queue in fixed order, or one per CUDA device plus one per CPU method."""
    if not concurrent:
        return {"sequential": list(METHODS)}
    queues = {}
    for method in METHODS:
        config = yaml.safe_load((preparation / "configs" / f"{method}.yaml").read_text())
        device = config["device"]
        queues.setdefault(device if device.startswith("cuda") else method, []).append(method)
    return queues


def _finish_method(method, process, log):
    returncode = process.wait()
    log.close()
    if returncode:
        print(f"{method} exited with status {returncode}; preserving failure", flush=True)


def _stop_methods(running):
    running = list(running)
    for _, process, _ in running:
        if process.poll() is None:
            process.terminate()
    for _, process, log in running:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
