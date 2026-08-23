from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_interp.data.segy_index import TRACE_TABLE_COLUMNS, scan_segy_headers
from tests.fixtures.tiny_segy import DEFAULT_HEADERS, write_tiny_segy


@pytest.fixture
def tiny_segy_path(tmp_path: Path) -> Path:
    return write_tiny_segy(tmp_path / "tiny.sgy").path


def test_scan_returns_expected_columns_in_order(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    assert tuple(trace_table.columns) == TRACE_TABLE_COLUMNS


def test_scan_returns_one_row_per_trace(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    assert len(trace_table) == len(DEFAULT_HEADERS)


def test_trace_index_is_unique_and_zero_based(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    assert trace_table["trace_index"].is_unique
    np.testing.assert_array_equal(
        trace_table["trace_index"].to_numpy(), np.arange(len(DEFAULT_HEADERS))
    )


def test_trace_index_in_ffid_restarts_per_ffid(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    np.testing.assert_array_equal(
        trace_table["trace_index_in_ffid"].to_numpy(), [0, 1, 2, 0, 1, 2, 3]
    )
    np.testing.assert_array_equal(trace_table["ffid"].to_numpy(), [10, 10, 10, 20, 20, 20, 20])


def test_coordinate_scalars_are_applied_per_trace(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    # Positive scalar 10, zero scalar and negative scalar -100 all map to the
    # same source position in metres.
    np.testing.assert_allclose(trace_table["source_x_m"].to_numpy(), 1000.0)
    np.testing.assert_allclose(trace_table["source_y_m"].to_numpy(), 2000.0)
    np.testing.assert_allclose(
        trace_table["receiver_x_m"].to_numpy(),
        [1400.0, 1000.0, 1300.0, 1000.0, 1400.0, 600.0, 1000.0],
    )


def test_geometry_matches_known_values(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    np.testing.assert_allclose(
        trace_table["cmp_x_m"].to_numpy(),
        [1200.0, 1000.0, 1150.0, 1000.0, 1200.0, 800.0, 1000.0],
    )
    np.testing.assert_allclose(
        trace_table["cmp_y_m"].to_numpy(),
        [2000.0, 2300.0, 2200.0, 2000.0, 2150.0, 2000.0, 1700.0],
    )
    np.testing.assert_allclose(
        trace_table["offset_m"].to_numpy(),
        [400.0, 600.0, 500.0, 0.0, 500.0, 400.0, 600.0],
    )
    np.testing.assert_allclose(
        trace_table["azimuth_deg"].to_numpy(),
        [270.0, 180.0, 216.86989765, 0.0, 233.13010235, 90.0, 0.0],
    )


def test_sample_metadata_comes_from_the_file(tmp_path: Path) -> None:
    tiny = write_tiny_segy(tmp_path / "tiny.sgy", sample_count=12, sample_interval_s=0.002)

    trace_table = scan_segy_headers(tiny.path)

    assert trace_table["sample_count"].unique().tolist() == [12]
    np.testing.assert_allclose(trace_table["sample_interval_s"].unique(), [0.002])


def test_source_file_stores_basename_only(tiny_segy_path: Path) -> None:
    trace_table = scan_segy_headers(tiny_segy_path)

    assert trace_table["source_file"].unique().tolist() == ["tiny.sgy"]


def test_unsupported_coordinate_units_are_rejected(tmp_path: Path) -> None:
    tiny = write_tiny_segy(tmp_path / "feet.sgy", coordinate_units=2)

    with pytest.raises(ValueError, match="unsupported coordinate units"):
        scan_segy_headers(tiny.path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_segy_headers(tmp_path / "absent.sgy")
