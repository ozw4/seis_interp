"""Run a separate observed-only fixed-batch diagnostic for FORGE neural methods."""

import gc
import json
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from seis_interp.data.forge_headers import sha256_file
from seis_interp.data.forge_mvp_artifacts import load_model_inputs, write_json
from seis_interp.data.forge_mvp_run_records import execution_identity, utc_now
from seis_interp.training.forge_overfit import NEURAL_METHODS, overfit_forge_batch


def run_forge_observed_overfit(
    repo, preparation, *, output=None, steps=2000, device="cuda:0", methods=NEURAL_METHODS
):
    """Save a separate diagnostic run without changing sealed study artifacts."""
    if steps < 1 or len(set(methods)) != len(methods) or not set(methods).issubset(NEURAL_METHODS):
        raise ValueError("positive steps and unique neural methods required")
    preparation = preparation.resolve()
    identity = execution_identity(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output or preparation.parent / (
        f"{stamp}_{identity['git_commit'][:12]}_observed_overfit"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    manifest = {
        **identity,
        "start_time": utc_now(),
        "status": "running",
        "preparation": str(preparation.relative_to(repo)),
        "preparation_hash": sha256_file(preparation / "preparation_manifest.json"),
        "steps": steps,
        "methods": list(methods),
        "scope": "first native observed-only batch repeated; training fit, not held-out quality",
        "amplitude_units": "model space; no IDW restoration",
        "checkpoint_rule": "fixed final raw; EMA scored separately",
        "device": device,
    }
    sources = [
        *sorted((repo / "src/seis_interp").rglob("*.py")),
        repo / "scripts/forge_observed_overfit.py",
    ]
    manifest["implementation_hashes"] = {}
    for source in sources:
        relative = source.relative_to(repo)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest["implementation_hashes"][str(relative)] = sha256_file(source)
    write_json(output / "manifest.json", manifest)
    summaries = []
    try:
        for method in methods:
            inputs, config = load_model_inputs(preparation, method)
            if config.get("alignment") is not None:
                raise ValueError("observed overfit does not support the aligned canvas")
            config["device"] = device
            directory = output / method
            directory.mkdir()
            (directory / "config.yaml").write_text(yaml.safe_dump(config))
            started = perf_counter()
            result = overfit_forge_batch(
                inputs, config, steps=steps, reporter=lambda message: print(message, flush=True)
            )
            elapsed = perf_counter() - started
            for name in ("target", "target_cells", "context_cells"):
                np.save(directory / f"{name}.npy", result[name], allow_pickle=False)
            for name, prediction in result["predictions"].items():
                np.save(directory / f"prediction_{name}.npy", prediction, allow_pickle=False)
            torch.save(result["checkpoint"], directory / "raw_state_dict.pt")
            pd.DataFrame(result["history"]).to_csv(directory / "training_history.csv", index=False)
            summary = {
                "method": method,
                "target_count": len(result["target_cells"]),
                "context_count": len(result["context_cells"]),
                "runtime_seconds": elapsed,
                "initialization_hash": result["initialization_hash"],
                **result["metrics"],
            }
            write_json(directory / "metrics.json", summary)
            summaries.append(summary)
            write_json(output / "summary.json", summaries)
            print(json.dumps(summary), flush=True)
            del result, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["end_time"] = utc_now()
        manifest["artifacts"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        }
        write_json(output / "manifest.json", manifest)

    return output
