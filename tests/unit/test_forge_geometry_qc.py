import numpy as np
import pandas as pd
import pytest

from seis_interp.processing.forge_geometry_qc import (
    availability_matrix,
    fit_station_lattice,
    prepare_geometry,
    sparse_grid_occupancy,
    spatial_coordinates,
    station_id_occupancy,
    station_spacing,
)


def test_availability_excludes_dead_and_zero_but_retains_review_and_unknown_station():
    table = pd.DataFrame(
        {
            "source_line": [1, 1, 1, 2, 2, 2],
            "source_point": 1,
            "receiver_line": 1,
            "receiver_point": [1, 2, 3, 1, 2, 3],
            "source_x_m": [0.0, 0.0, 0.0, 20.0, 20.0, 20.0],
            "source_y_m": 0.0,
            "receiver_x_m": [10.0, 0.0, 0.0, 10.0, 30.0, 0.0],
            "receiver_y_m": 0.0,
            "trace_identification_code": [1, 2, 2, 1, 1, 2],
            "header_eligible": [True, False, False, True, True, False],
            "numerically_usable": [True, False, False, False, True, False],
            "waveform_status": [
                "review",
                "header_excluded",
                "header_excluded",
                "all_zero",
                "passed_numeric_checks",
                "header_excluded",
            ],
        }
    )
    original = table.copy(deep=True)
    domain, sources, receivers, observed = prepare_geometry(table)
    np.testing.assert_array_equal(availability_matrix(domain, 2, 3), [[3, 1, 1], [2, 4, 1]])
    assert receivers.receiver_x_m.iloc[1] == 30  # Coordinate known from the other shot.
    assert np.isnan(receivers.receiver_x_m.iloc[2])  # Never fill an unknown coordinate.
    assert domain.receiver_x_m.iloc[1] == 0
    assert len(observed) == 2
    np.testing.assert_allclose(observed.azimuth_deg, [90, 90])
    assert len(sources) == 2
    pd.testing.assert_frame_equal(table, original)
    with pytest.raises(ValueError, match="duplicate recorded"):
        prepare_geometry(pd.concat([table, table.iloc[:1]]))
    table.loc[3, "receiver_x_m"] = 11
    with pytest.raises(ValueError, match="varying receiver"):
        prepare_geometry(table)


def test_full_offset_transform_is_invertible_and_rotates_both_roles():
    table = pd.DataFrame(
        {"source_x_m": [10.0], "source_y_m": [20.0], "receiver_x_m": [16.0], "receiver_y_m": [28.0]}
    )
    sr = spatial_coordinates(table, "source_receiver", [10, 20], [0, 1])
    mo = spatial_coordinates(table, "midpoint_offset", [10, 20], [0, 1])
    np.testing.assert_allclose(sr, [[0, 0, 8, -6]])
    np.testing.assert_allclose(mo, [[4, -3, 8, -6]])
    np.testing.assert_allclose(mo[:, :2] - mo[:, 2:] / 2, sr[:, :2])
    np.testing.assert_allclose(mo[:, :2] + mo[:, 2:] / 2, sr[:, 2:])


def test_grid_separates_structural_voids_qc_holes_and_collisions():
    # A bounding rectangle has three cells, but no reference pair in the middle.
    reference = np.array([[0.0, 0, 0, 0], [0.1, 0, 0, 0], [2.0, 0, 0, 0]])
    observed = reference[:2]
    summary, cells = sparse_grid_occupancy(reference, observed, [1] * 4)
    assert summary["bbox_cells"] == 3
    assert summary["reference_cells"] == 2
    assert summary["observed_cells"] == 1
    assert summary["outside_reference_cells"] == 1
    assert summary["unobserved_reference_cells"] == 1
    assert summary["traces_in_collision_cells"] == 2
    assert summary["extra_traces_beyond_one_per_cell"] == 1
    assert summary["bbox_fill_fraction"] == 1 / 3
    assert summary["reference_fill_fraction"] == 0.5
    assert cells.observed_traces.tolist() == [2, 0]
    with pytest.raises(ValueError, match="subset"):
        sparse_grid_occupancy(reference, np.array([[1.0, 0, 0, 0]]), [1] * 4)


def test_grid_negative_coordinates_half_open_boundaries_and_phase():
    coordinates = np.array([[-1.0, 0, 0, 0], [-0.01, 0, 0, 0], [0.0, 0, 0, 0], [1.0, 0, 0, 0]])
    summary, cells = sparse_grid_occupancy(coordinates, coordinates, [1] * 4)
    assert cells.i0.tolist() == [-1, 0, 1]
    assert cells.observed_traces.tolist() == [2, 1, 1]
    _, shifted = sparse_grid_occupancy(coordinates, coordinates, [1] * 4, 0.5)
    assert shifted.i0.tolist() == [-2, -1, 0]
    assert shifted.observed_traces.tolist() == [1, 2, 1]
    assert summary["reference_fill_fraction"] == 1
    with pytest.raises(ValueError, match="positive"):
        sparse_grid_occupancy(coordinates, coordinates, [0] * 4)


def test_affine_lattice_and_spacing_keep_id_gaps_explicit():
    stations = pd.DataFrame(
        {
            "receiver_line": [101, 101, 105, 105],
            "receiver_point": [501, 503, 501, 503],
            "receiver_x_m": [0.0, 0.0, 200.0, 200.0],
            "receiver_y_m": [0.0, 100.0, 0.0, 100.0],
        }
    )
    fit, residual = fit_station_lattice(stations, "receiver")
    np.testing.assert_allclose(fit["coefficients_xy_m"][1:], [[50, 0], [0, 50]], atol=1e-12)
    assert residual.fit_residual_m.max() < 1e-12
    along, across = station_spacing(stations, "receiver")
    assert along.point_id_step.tolist() == [2, 2]
    assert along.distance_m.tolist() == [100, 100]
    assert across.line_id_step.tolist() == [4, 4]
    assert across.distance_m.tolist() == [200, 200]
    sources = stations.rename(columns=lambda c: c.replace("receiver", "source"))
    result = station_id_occupancy(sources, stations, pd.DataFrame(index=range(16)), [4, 1, 4, 1])
    assert result["shape"] == [2, 3, 2, 3]
    assert result["fill_fraction"] == 16 / 36
