"""Prepare one complete SEG C3 NA shot as an interim trace dataset."""

from __future__ import annotations

from pathlib import Path

from seis_interp.data.segy_index import scan_segy_headers
from seis_interp.data.segy_reader import build_time_axis, read_trace_amplitudes
from seis_interp.data.trace_store import write_interim_trace_dataset
from seis_interp.processing.ffid_selection import select_ffid

# Traces in a complete SEG C3 Narrow-Azimuth shot. Shots at the sail-line ends
# have fewer. Other surveys have their own count, so this stays out of
# seis_interp.processing.ffid_selection.
C3_COMPLETE_SHOT_TRACE_COUNT = 544


def prepare_c3_complete_shot(
    input_path: Path,
    output_dir: Path,
    ffid: int | None = None,
    expected_trace_count: int = C3_COMPLETE_SHOT_TRACE_COUNT,
    dataset_id: str = "seg_c3_na",
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan one SEG-Y file, select one FFID and write the interim dataset.

    With ``ffid=None`` the numerically smallest complete FFID is selected.
    Returns exactly the metadata that was written to ``dataset.json``.
    """
    input_path = Path(input_path)

    trace_table = scan_segy_headers(input_path)
    selected_traces = select_ffid(
        trace_table,
        ffid=ffid,
        expected_trace_count=expected_trace_count,
    )

    amplitudes = read_trace_amplitudes(input_path, selected_traces["trace_index"].tolist())
    time_s = build_time_axis(
        int(selected_traces["sample_count"].iloc[0]),
        float(selected_traces["sample_interval_s"].iloc[0]),
    )

    return write_interim_trace_dataset(
        output_dir=Path(output_dir),
        trace_table=selected_traces,
        amplitudes=amplitudes,
        time_s=time_s,
        source_path=input_path,
        dataset_id=dataset_id,
        selection={
            "ffid": int(selected_traces["ffid"].iloc[0]),
            "expected_trace_count": int(expected_trace_count),
        },
        overwrite=overwrite,
    )
