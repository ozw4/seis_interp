from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seis_interp.pipelines.qc_c3 import qc_c3_crop, qc_c3_geometry
from seis_interp.processing.c3_crop_selection import resolve_c3_crop
from seis_interp.processing.c3_crop_signal_qc import summarize_c3_crop_signal
from seis_interp.processing.c3_geometry_qc import summarize_c3_geometry, summarize_time_axis
from tests.fixtures.c3_benchmark import make_benchmark_interim
from tests.fixtures.c3_volume_artifacts import make_c3_trace_table


def test_geometry_original_numbers_and_global_indices_preserve_mapping():
    table = make_c3_trace_table(physical_source_line_indices=(0, 1, 2, 3))
    table["sail_line_number"] = ((table.source_x_m - 1000) / 160).astype(int) + 25
    _, summary = summarize_c3_geometry(
        table, np.arange(5) * 0.008 + 0.7, requested_inclusive=(26, 27), time_range=(0, 4)
    )
    assert summary["sail_lines"]["numbering"] == "original_sail_line_number"
    assert summary["sail_lines"]["index_range"] == [1, 3]
    assert summary["time"]["time_start_s"] == 0.7
    assert summary["duplicate_physical_row_count"] == 0
    assert summary["incomplete_ffid_count"] == 0
    assert summary["stagger"][0]["absolute_differences_m"] == [40.0]
    _, again = summarize_c3_geometry(
        table.sample(frac=1, random_state=7),
        np.arange(5) * 0.008 + 0.7,
        requested_inclusive=(26, 27),
        time_range=(0, 4),
    )
    assert again == summary


def test_geometry_reports_missing_shot_and_duplicate_cell():
    table = make_c3_trace_table(omitted_shots=frozenset({(0, 1)}))
    duplicate = table.iloc[[0]].assign(array_row=len(table))
    table = pd.concat([table, duplicate], ignore_index=True)
    _, summary = summarize_c3_geometry(
        table, np.arange(4), requested_inclusive=(0, 1), time_range=(0, 4)
    )
    assert summary["lines"][0]["missing_intermediate_shots"] == 1
    assert not summary["lines"][0]["regular_shot_spacing"]
    assert summary["duplicate_physical_row_count"] == 1
    with pytest.raises(ValueError, match="requires 384"):
        summarize_time_axis(np.arange(383), (0, 384))


def test_central_crop_canonicalization_and_row_order():
    table = make_c3_trace_table()
    duplicate = table.iloc[[500]].assign(array_row=len(table))
    duplicated = pd.concat([table, duplicate], ignore_index=True)

    def resolve(t):
        return resolve_c3_crop(
            t, np.arange(5), time_range=(0, 4), source_line_range=(0, 2), spatial_lengths=(2, 2, 4)
        )

    index, summary = resolve(duplicated)
    assert summary["selection"] == {
        "time": [0, 4],
        "source_line": [0, 2],
        "shot_in_line": [0, 2],
        "relative_receiver_x": [3, 5],
        "relative_receiver_y": [32, 36],
    }
    other, other_summary = resolve(duplicated.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(index, other)
    assert summary == other_summary
    assert summary["adjustment_reason"] is None


def test_crop_moves_only_unspecified_starts_when_center_has_hole():
    table = make_c3_trace_table()
    table = table.loc[
        ~(
            table.source_x_m.eq(1000)
            & table.source_y_m.eq(0)
            & table.receiver_x_m.eq(980)
            & table.receiver_y_m.eq(-1400)
        )
    ]
    index, summary = resolve_c3_crop(
        table,
        np.arange(5),
        time_range=(0, 4),
        source_line_range=(0, 2),
        spatial_lengths=(2, 2, 4),
        explicit_ranges={"shot_in_line": [0, 2], "relative_receiver_x": [3, 5]},
    )
    assert summary["selection"]["relative_receiver_y"] == [33, 37]
    assert summary["selection"]["time"] == [0, 4]
    assert summary["selection"]["source_line"] == [0, 2]
    assert "not dense" in summary["adjustment_reason"]
    assert len(index) == 2 * 2 * 2 * 4
    with pytest.raises(ValueError, match="no valid crop"):
        resolve_c3_crop(
            table,
            np.arange(5),
            time_range=(0, 4),
            source_line_range=(0, 2),
            spatial_lengths=(2, 2, 4),
            explicit_ranges={
                "shot_in_line": [0, 2],
                "relative_receiver_x": [3, 5],
                "relative_receiver_y": [32, 36],
            },
        )


def test_signal_chunks_exclude_outside_samples_and_preserve_zeros():
    values = np.array([[1, 2, 0, 4, 1e9], [0, 0, 0, 0, 1e9], [1e9] * 5], dtype=np.float32)
    times = 0.1 + np.arange(5) * 0.008
    summary = summarize_c3_crop_signal(
        values, np.array([0, 1]), times, time_range=(0, 4), chunk_rows=1
    )
    assert summary["energy"] == 21
    assert summary["rms"] == np.sqrt(21 / 8)
    assert summary["zero_count"] == 5
    assert summary["max_abs_amplitude"] == 4
    assert summary["time"]["time_start_s"] == 0.1
    assert sum(b["energy"] for b in summary["time_band_energy"]) == 21
    values[0, 1] = np.nan
    summary = summarize_c3_crop_signal(values, np.array([0, 1]), times, time_range=(0, 4))
    assert not summary["ok"] and summary["nonfinite_count"] == 1
    assert summary["rms"] is None


def test_center_uses_common_selected_line_bounds_in_global_receiver_indices():
    table = make_c3_trace_table(physical_source_line_indices=(0, 1, 2, 3))
    in_lines = table.source_x_m.between(1160, 1320)
    low_receiver = (table.receiver_y_m - table.source_y_m).lt(-2680 + 20 * 40)
    table = table.loc[~(in_lines & low_receiver)]
    _, summary = resolve_c3_crop(
        table, np.arange(4), time_range=(0, 4), source_line_range=(1, 3), spatial_lengths=(2, 2, 4)
    )
    assert summary["selection"]["relative_receiver_y"] == [42, 46]
    assert summary["common_available_ranges"]["relative_receiver_y"] == [20, 68]


def test_geometry_io_does_not_open_amplitudes_and_refuses_overwrite(tmp_path, monkeypatch):
    interim = make_benchmark_interim(tmp_path)
    actual = np.load

    def guarded(path, *args, **kwargs):
        assert "amplitudes" not in str(path)
        return actual(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    summary = qc_c3_geometry(
        interim, tmp_path / "geometry", requested_inclusive=(1, 2), time_range=(0, 4)
    )
    assert summary["sail_lines"]["index_range"] == [1, 3]
    with pytest.raises(FileExistsError):
        qc_c3_geometry(
            interim, tmp_path / "geometry", requested_inclusive=(1, 2), time_range=(0, 4)
        )


def test_crop_qc_writes_premask_mapping_and_measured_times(tmp_path):
    interim = make_benchmark_interim(tmp_path)
    result = qc_c3_crop(
        interim,
        tmp_path / "crop",
        source_line_range=(1, 3),
        time_range=(0, 4),
        spatial_lengths=(2, 2, 4),
    )
    assert result["shape"] == [4, 2, 2, 2, 4]
    assert result["signal"]["ok"]
    assert result["time"]["time_start_s"] == 0.125
    assert set(p.name for p in (tmp_path / "crop").iterdir()) == {"crop.json", "crop_index.parquet"}
