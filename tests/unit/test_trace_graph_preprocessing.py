from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from tests.fixtures.relational_trace_graph import make_relational_trace_domains


class RecordingAmplitudes:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.reads: list[tuple[np.ndarray, slice]] = []

    def __getitem__(self, selection: tuple[np.ndarray, slice]) -> np.ndarray:
        rows, times = selection
        self.reads.append((rows.copy(), times))
        return self.values[selection]


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_fit_uses_global_training_energy_selected_times_and_fixed_mean_cmp(chunk_size: int) -> None:
    _, training, values = make_relational_trace_domains()
    reader = RecordingAmplitudes(values)

    preprocessing = fit_trace_graph_preprocessing(
        training,
        reader,
        position_scale_m=10.0,
        offset_scale_m=2.0,
        azimuth_min_offset_m=0.1,
        row_chunk_size=chunk_size,
    )

    # Training rows have factors 5, 1, 6; the selected sample factors are 2..6.
    expected_rms = np.sqrt((25 + 1 + 36) * (4 + 9 + 16 + 25 + 36) / 15)
    assert preprocessing.amplitude_scale == pytest.approx(expected_rms, rel=1e-14)
    assert preprocessing.midpoint_origin_m == (0.5, -1.0)
    assert preprocessing.time_s == tuple(training.time_s)
    assert preprocessing.position_scale_m == 10.0
    assert preprocessing.offset_scale_m == 2.0
    assert preprocessing.fit_domain["pool"] == "mask_observed"
    assert preprocessing.fit_domain["trace_count"] == 3
    assert preprocessing.fit_domain["time_samples"] == [1, 6]
    assert preprocessing.fit_domain["inputs_lock"] == training.inputs_lock
    assert sorted(np.concatenate([rows for rows, _ in reader.reads]).tolist()) == [0, 4, 5]
    assert all(times == slice(1, 6) for _, times in reader.reads)


def test_held_out_values_never_affect_fit_and_input_arrays_are_preserved() -> None:
    _, training, values = make_relational_trace_domains()
    original = values.copy()
    settings = {"position_scale_m": 10.0, "offset_scale_m": 2.0, "azimuth_min_offset_m": 0.1}
    before = fit_trace_graph_preprocessing(training, values, **settings)
    np.testing.assert_array_equal(values, original)
    values[[1, 2, 3, 6]] = np.nan
    values[:, [0, 6]] = np.inf

    after = fit_trace_graph_preprocessing(training, values, **settings)

    assert before == after
    training.inputs_lock["fixture"] = "changed later"
    assert before.fit_domain["inputs_lock"]["fixture"] == "training_traces"


def test_pool_order_and_chunks_preserve_the_fitted_domain_identity() -> None:
    _, training, values = make_relational_trace_domains()
    order = np.array([2, 0, 1])
    permuted = replace(
        training,
        trace_ids=training.trace_ids[order],
        array_rows=training.array_rows[order],
        source_xy_m=training.source_xy_m[order],
        receiver_xy_m=training.receiver_xy_m[order],
    )
    settings = {"position_scale_m": 10.0, "offset_scale_m": 2.0, "azimuth_min_offset_m": 0.1}
    one = fit_trace_graph_preprocessing(training, values, row_chunk_size=1, **settings)
    two = fit_trace_graph_preprocessing(permuted, values, row_chunk_size=2, **settings)

    assert one == two


def test_training_pool_statistics_are_fixed_before_artificial_hiding() -> None:
    _, training, values = make_relational_trace_domains()
    settings = {"position_scale_m": 10.0, "offset_scale_m": 2.0, "azimuth_min_offset_m": 0.1}
    fixed = fit_trace_graph_preprocessing(training, values, **settings)
    values[training.array_rows[0], 1:6] *= 10
    refitted = fit_trace_graph_preprocessing(training, values, **settings)

    assert fixed.amplitude_scale != refitted.amplitude_scale
    # An episode cannot quietly fit a new scale after changing visibility.
    episode = replace(training, observed_mask=np.array([False, True, True]))
    with pytest.raises(ValueError, match="complete unmasked training pool"):
        fit_trace_graph_preprocessing(episode, values, **settings)


def test_evaluation_domain_is_rejected_before_reading_any_amplitudes() -> None:
    domain, _, values = make_relational_trace_domains()
    reader = RecordingAmplitudes(values)

    with pytest.raises(ValueError, match="training pool"):
        fit_trace_graph_preprocessing(
            domain,
            reader,
            position_scale_m=10.0,
            offset_scale_m=2.0,
            azimuth_min_offset_m=0.1,
        )
    assert reader.reads == []


@pytest.mark.parametrize("bad_values", [0.0, np.nan])
def test_undefined_training_scale_is_rejected(bad_values: float) -> None:
    _, training, values = make_relational_trace_domains()
    values[:] = bad_values
    with pytest.raises(ValueError, match="finite|positive"):
        fit_trace_graph_preprocessing(
            training,
            values,
            position_scale_m=10.0,
            offset_scale_m=2.0,
            azimuth_min_offset_m=0.1,
        )


@pytest.mark.parametrize("name", ["amplitude_scale", "position_scale_m", "offset_scale_m"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf])
def test_fixed_preprocessing_rejects_invalid_scales(name: str, value: float) -> None:
    _, training, values = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training, values, position_scale_m=10.0, offset_scale_m=2.0, azimuth_min_offset_m=0.1
    )
    with pytest.raises(ValueError, match=name):
        replace(preprocessing, **{name: value})


@pytest.mark.parametrize("name", ["position_scale_m", "offset_scale_m"])
@pytest.mark.parametrize("value", [True, "2.0"])
def test_fit_does_not_coerce_invalid_configured_scale_types(name: str, value: object) -> None:
    _, training, values = make_relational_trace_domains()
    settings = {"position_scale_m": 10.0, "offset_scale_m": 2.0, "azimuth_min_offset_m": 0.1}
    settings[name] = value
    with pytest.raises(ValueError, match=name):
        fit_trace_graph_preprocessing(training, values, **settings)
