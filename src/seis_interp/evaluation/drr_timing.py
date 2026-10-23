"""Measure production CPU DRR blocks without reading scoring targets."""

import json
import resource
from time import perf_counter
from unittest.mock import patch

import numpy as np

from seis_interp.processing import drr_batched
from seis_interp.processing.drr_grid import resolve_drr_grid_config
from seis_interp.training.c3_region_methods import run_c3_region_method
from seis_interp.training.forge_c3_methods import run_forge_c3_method


def measure_drr_candidate(inputs, config, block_log):
    """Time every block in a complete candidate, keeping the production math.

    Each measurement must have a fresh process: Linux ru_maxrss includes input
    loading and all allocations since process startup. The hook only observes
    the production block call; workers=1 avoids concurrent hook/log access.
    """
    config = resolve_drr_grid_config(config)
    algorithm = config["algorithm"]
    if (
        config["device"] != "cpu"
        or config["execution"]["workers"] != 1
        or algorithm["backend"] != "drr_batched"
        or algorithm["n_iterations"] not in (1, 2)
    ):
        raise ValueError("timing requires overlapping CPU DRR, workers=1, and 1 or 2 iterations")
    reconstruct = drr_batched._reconstruct_batch
    records = []

    def timed_block(batch, **kwargs):
        if len(batch) != 1:
            raise ValueError("per-block timing requires one spatial block per batch")
        started = perf_counter()
        result = reconstruct(batch, **kwargs)
        record = {
            "block_index": len(records),
            "location": list(batch[0]),
            "empty": not bool(result[0]),
            "wall_seconds": perf_counter() - started,
            "process_peak_rss_bytes_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        records.append(record)
        block_log.write(json.dumps(record) + "\n")
        block_log.flush()
        return result

    run = run_forge_c3_method if config.get("alignment") else run_c3_region_method
    started = perf_counter()
    with patch.object(drr_batched, "_reconstruct_batch", timed_block):
        result = run(inputs, config, reporter=lambda message: None)
    elapsed = perf_counter() - started
    block_seconds = sum(record["wall_seconds"] for record in records)
    overhead = max(0.0, elapsed - block_seconds)
    iterations = algorithm["n_iterations"]
    estimate = overhead + block_seconds * 60 / iterations
    nonempty = [record["wall_seconds"] for record in records if not record["empty"]]
    return {
        "status": "completed",
        "config": config,
        "wall_seconds": elapsed,
        "block_count": len(records),
        "nonempty_block_count": len(nonempty),
        "block_wall_seconds_total": block_seconds,
        "mean_nonempty_block_wall_seconds": float(np.mean(nonempty)) if nonempty else None,
        "non_block_overhead_seconds": overhead,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rss_measurement": "Linux ru_maxrss; process lifetime, including input loading",
        "prediction_shape": list(result.prediction.shape),
        "extrapolated_60_iteration_full_candidate_seconds": estimate,
        "extrapolated_60_iteration_mean_block_seconds": (
            float(np.mean(nonempty)) * 60 / iterations if nonempty else None
        ),
        "extrapolation": "linear iteration scaling; FFT/setup also scaled, conservative estimate",
        "exceeds_72_hours": estimate > 72 * 3600,
        "validation_only": True,
        "validation_scoring_calls": 0,
        "test_scoring_calls": 0,
    }
