"""Read-only verification of the fixed random-80 PoC inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.processing.c3_benchmark_contract import MAIN_C3_DIMENSIONS, C3BenchmarkDimensions
from seis_interp.training.amplitude_scaling import compute_observed_global_rms

POC_SELECTION = {
    "time": [0, 384],
    "source_line": [25, 41],
    "shot_in_line": [28, 60],
    "relative_receiver_x": [0, 8],
    "relative_receiver_y": [18, 50],
}


def check_c3_poc_inputs(
    *,
    interim_dir: Path,
    processed_dir: Path,
    mask_dir: Path,
    case_dir: Path,
    volume_dir: Path,
    selection: Mapping[str, object] = POC_SELECTION,
    dimensions: C3BenchmarkDimensions = MAIN_C3_DIMENSIONS,
) -> dict[str, object]:
    """Verify the intended full selection without reading evaluation-target truth."""
    inputs = load_c3_random80_poc_inputs(
        config={"benchmark_volume": {"selection": selection}},
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        mask_dir=mask_dir,
        case_dir=case_dir,
        volume_dir=volume_dir,
        dimensions=dimensions,
    )
    volume = inputs.observed_volume
    return {
        key: inputs.inputs_lock[key]
        for key in (
            "benchmark_id",
            "dataset_id",
            "case_id",
            "volume_id",
            "selection",
            "shape",
            "observed_trace_count",
            "target_trace_count",
        )
    } | {
        "observed_global_rms": compute_observed_global_rms(
            volume.values, volume.observed_trace_mask
        )
    }
