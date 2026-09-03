"""Synthetic fixed-grid whole-shot survey shared by gather pipeline tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.pipelines.prepare_baseline import prepare_baseline_dataset
from seis_interp.processing.trace_amplitude_filter import TraceAmplitudeFilterConfig

FFID_RANGE = (10, 17)
RANDOM_SEED = 5
TIME_SAMPLE_COUNT = 5


def prepare_whole_shot_survey(
    tmp_path: Path,
) -> tuple[Path, Path, TraceAmplitudeFilterConfig]:
    """Write an 8-FFID fixed 8 x 68 interim + whole-FFID processed dataset."""
    source = tmp_path / "source.sgy"
    source.write_bytes(b"synthetic whole-shot source")
    receiver_x_offsets = np.arange(-140.0, 180.0, 40.0)
    receiver_y_offsets = np.arange(-2680.0, 40.0, 40.0)
    time_s = np.arange(TIME_SAMPLE_COUNT, dtype=np.float64) * 0.008
    rows: list[dict[str, object]] = []
    amplitudes: list[np.ndarray] = []
    trace_index = 0
    for ffid_index, ffid in enumerate(range(FFID_RANGE[0], FFID_RANGE[1] + 1)):
        source_x = 4000.0 + (ffid_index // 4) * 160.0
        source_y = 1000.0 + (ffid_index % 4) * 80.0
        for receiver_x_offset in receiver_x_offsets:
            for receiver_y_offset in receiver_y_offsets:
                receiver_x = source_x + receiver_x_offset
                receiver_y = source_y + receiver_y_offset
                offset = math.hypot(receiver_x_offset, receiver_y_offset)
                rows.append(
                    {
                        "trace_index": trace_index,
                        "ffid": ffid,
                        "source_x_m": source_x,
                        "source_y_m": source_y,
                        "receiver_x_m": receiver_x,
                        "receiver_y_m": receiver_y,
                        "cmp_x_m": (source_x + receiver_x) / 2.0,
                        "cmp_y_m": (source_y + receiver_y) / 2.0,
                        "offset_m": offset,
                        "azimuth_deg": math.degrees(
                            math.atan2(-receiver_x_offset, -receiver_y_offset)
                        )
                        % 360.0,
                        "sample_interval_s": 0.008,
                        "trace_count_in_ffid": 544,
                        "is_complete_ffid": True,
                    }
                )
                phase = ffid_index * 0.1 + receiver_x_offset * 0.001
                trace = np.sin(np.arange(TIME_SAMPLE_COUNT) * 0.4 + phase)
                trace += 0.2 * np.cos(
                    np.arange(TIME_SAMPLE_COUNT) * 0.2 + receiver_y_offset * 0.001
                )
                amplitudes.append(trace.astype(np.float32))
                trace_index += 1
    interim = tmp_path / "interim"
    write_interim_trace_dataset(
        interim,
        pd.DataFrame(rows),
        np.stack(amplitudes),
        time_s,
        source,
        "synthetic_shot_gather_grid",
        selection={"ffid_min": FFID_RANGE[0], "ffid_max": FFID_RANGE[1]},
    )
    trace_filter = TraceAmplitudeFilterConfig(
        exclude_all_zero=True,
        max_abs_amplitude=10000.0,
    )
    processed = tmp_path / "processed"
    prepare_baseline_dataset(
        interim,
        processed,
        holdout_fraction=0.5,
        validation_fraction_of_holdout=0.5,
        random_seed=RANDOM_SEED,
        split_scope="whole_ffid",
        trace_amplitude_filter=trace_filter,
        config_source="studies/synthetic_shot_gather/config.yaml",
    )
    return interim, processed, trace_filter
