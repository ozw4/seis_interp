"""Execute frozen methods in isolated processes, then open evaluation targets."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

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


def run_forge_mvp(repo: Path, preparation: Path, output: Path, preflight: Path):
    seal = json.loads((preparation / "preparation_manifest.json").read_text())
    verify_hashes(preparation, seal["artifacts"])
    verify_mvp_implementation(repo, seal["implementation_hashes"])
    preflight_hash = validate_preflight(preparation, preflight)
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
        },
    )
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    for method in METHODS:
        print(f"Starting fixed MVP method: {method}", flush=True)
        with (output / f"{method}.log").open("w") as log:
            result = subprocess.run(
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
        if result.returncode:
            print(
                f"{method} exited with status {result.returncode}; preserving failure", flush=True
            )
    from seis_interp.pipelines.evaluate_forge_mvp import evaluate_forge_mvp

    return evaluate_forge_mvp(repo, preparation, output)
