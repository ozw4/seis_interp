"""Small C3-compatible surveys and explicit contracts for suite integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.processing.geometry import compute_trace_geometry
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def make_benchmark_interim(
    directory: Path, *, hole: bool = False, time_origin: float = 0.125
) -> Path:
    """Build five lines, three shots/line, and the existing full receiver lattice."""
    table = make_c3_trace_table(physical_source_line_indices=(0, 1, 2, 3, 4))
    if hole:
        # The central receiver window [32,36) contains this cell; [33,37) is dense.
        remove = (
            table.source_x_m.eq(1160)
            & table.source_y_m.eq(120)
            & (table.receiver_x_m - table.source_x_m).eq(-20)
            & (table.receiver_y_m - table.source_y_m).eq(-1400)
        )
        table = table.loc[~remove].reset_index(drop=True)
    geometry = compute_trace_geometry(
        *(
            table[name].to_numpy()
            for name in ("source_x_m", "source_y_m", "receiver_x_m", "receiver_y_m")
        )
    )
    for name, values in zip(
        ("cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"), geometry, strict=True
    ):
        table[name] = values
    table["trace_index"] = np.arange(len(table), dtype=np.int64)
    table["sample_interval_s"] = 0.008
    table = table.drop(columns="array_row")
    time_s = time_origin + np.arange(8, dtype=np.float64) * 0.008
    rng = np.random.default_rng(23)
    amplitudes = rng.normal(size=(len(table), 8)).astype(np.float32)
    # A legitimate zero-valued receiver column remains observed when its mask says so.
    amplitudes[(table.receiver_x_m - table.source_x_m).eq(-20).to_numpy()] = 0.0
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic.sgy"
    source.write_bytes(b"synthetic source for C3 benchmark tests")
    interim = directory / "interim"
    write_interim_trace_dataset(
        interim, table, amplitudes, time_s, source, "synthetic_c3", {"ffid_scope": "all"}
    )
    return interim


def synthetic_benchmark_config() -> dict:
    """Return explicit small expectations; production CLI never selects this profile."""
    return {
        "project": {"random_seed": 42},
        "data": {"dataset_id": "synthetic_c3"},
        "normalization": {
            "coordinates": "train_minmax_linear_plus_azimuth_sin_cos",
            "amplitude": "train_global_rms",
        },
        "sampling": {"split_scope": "c3_source_line_blocks"},
        "c3_benchmark": {
            "extraction_reference": "5-D Seismic Data Interpolation by Continuous Representation",
            "axis_order": [
                "time",
                "source_line",
                "shot_in_line",
                "relative_receiver_x",
                "relative_receiver_y",
            ],
            "shape": [4, 2, 2, 2, 4],
            "sail_lines": {
                "requested_inclusive": [1, 2],
                "numbering": "global_source_line_index",
                "index_range": [1, 3],
            },
            "start_rule": "centered_contiguous",
            "training": {"time_samples": [0, 4], "random_seeds": [71]},
            "inference": {
                "amplitude_domain": "same_crop_observed_traces",
                "outside_crop_context": False,
                "target_coordinates": "allowed",
                "target_amplitudes": "scoring_only",
            },
        },
        "benchmark_volume": {
            "selection": {
                "time": [0, 4],
                "source_line": [1, 3],
                "shot_in_line": None,
                "relative_receiver_x": None,
                "relative_receiver_y": None,
            }
        },
        "evaluation": {
            "domain": "evaluation_target",
            "primary_metric": "physical_amplitude_global_snr_db",
        },
    }


def synthetic_benchmark_cases() -> list[dict]:
    return [
        {
            "case_id": f"{partition}_{kind}",
            "partition": partition,
            "kind": kind,
            "missing_fraction": 0.5,
            "random_seed": 42,
        }
        for partition in ("test", "validation")
        for kind in ("random_trace", "random_whole_ffid")
    ]
