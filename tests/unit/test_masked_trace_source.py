from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_interp.data.masked_trace_inputs import MaskedTraceGraphInputs
from seis_interp.data.masked_trace_source import (
    MaskedTraceSource,
    assemble_masked_trace_graph_inputs,
)
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from tests.fixtures.relational_trace_graph import (
    make_relational_trace_domains,
    make_relational_trace_plan,
)


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


def _source():
    domain, training, values = make_relational_trace_domains()
    preprocessing = fit_trace_graph_preprocessing(
        training, values, position_scale_m=10.0, offset_scale_m=2.0, azimuth_min_offset_m=0.1
    )
    reader = RecordingAmplitudes(values)
    source = MaskedTraceSource(domain, preprocessing, reader)
    return source, domain, reader


def test_only_support_rows_are_read_and_query_waveforms_start_at_exact_zero() -> None:
    source, domain, reader = _source()
    assert reader.reads == []
    plan = make_relational_trace_plan(domain)
    original = reader.values.copy()

    inputs = source.inputs(plan)

    rows_by_id = dict(zip(domain.trace_ids, domain.array_rows, strict=True))
    expected_rows = np.array([rows_by_id[i] for i in plan.trace_ids[plan.observed_mask]])
    assert len(reader.reads) == 1
    np.testing.assert_array_equal(reader.reads[0][0], expected_rows)
    assert reader.reads[0][1] == slice(1, 6)
    assert set(expected_rows).isdisjoint(domain.array_rows[domain.query_mask])
    assert set(expected_rows) < set(domain.array_rows[domain.observed_mask])
    torch.testing.assert_close(
        inputs.waveforms[inputs.observed_mask],
        torch.tensor(
            original[expected_rows, 1:6].astype(np.float64) / source.preprocessing.amplitude_scale,
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(inputs.waveforms[inputs.query_indices]) == 0
    assert inputs.waveforms.shape == (len(plan.trace_ids), 5)
    assert all(
        "label" not in field.name and "array_row" not in field.name for field in fields(inputs)
    )
    np.testing.assert_array_equal(reader.values, original)


def test_fixed_input_tensors_do_not_depend_on_hidden_labels_or_unused_observations() -> None:
    source, domain, reader = _source()
    plan = make_relational_trace_plan(domain)
    before = source.inputs(plan)
    reader.values[domain.array_rows[domain.query_mask]] = np.nan
    used_rows = set(reader.reads[0][0])
    unused = [row for row in domain.array_rows if row not in used_rows]
    reader.values[unused] = np.inf

    after = source.inputs(plan)

    for field in fields(MaskedTraceGraphInputs):
        assert torch.equal(getattr(before, field.name), getattr(after, field.name))


def test_query_splitting_keeps_fixed_features_and_normalized_observations() -> None:
    source, domain, _ = _source()
    joint_plan = make_relational_trace_plan(domain)
    joint = source.inputs(joint_plan)
    joint_positions = {int(trace_id): i for i, trace_id in enumerate(joint_plan.trace_ids)}
    for trace_id in domain.trace_ids[domain.query_mask]:
        plan = make_relational_trace_plan(domain, np.array([trace_id]))
        inputs = source.inputs(plan)
        positions = [joint_positions[int(i)] for i in plan.trace_ids]
        assert torch.equal(inputs.waveforms, joint.waveforms[positions])
        assert torch.equal(inputs.node_features, joint.node_features[positions])
        joint_edges = {
            (int(joint_plan.trace_ids[s]), int(joint_plan.trace_ids[d]), int(relation)): feature
            for (s, d), relation, feature in zip(
                joint_plan.edge_index.T, joint_plan.edge_type, joint.edge_features, strict=True
            )
        }
        for (s, d), relation, feature in zip(
            plan.edge_index.T, plan.edge_type, inputs.edge_features, strict=True
        ):
            key = (int(plan.trace_ids[s]), int(plan.trace_ids[d]), int(relation))
            assert torch.equal(feature, joint_edges[key])


def test_source_memory_maps_file_without_reading_target_values(tmp_path: Path) -> None:
    source, domain, reader = _source()
    values = reader.values.copy()
    values[domain.array_rows[domain.query_mask]] = np.nan
    path = tmp_path / "amplitudes.npy"
    np.save(path, values)

    lazy = MaskedTraceSource(replace(domain, amplitudes_path=path), source.preprocessing)
    inputs = lazy.inputs(make_relational_trace_plan(domain))

    assert isinstance(lazy._amplitudes, np.memmap)
    assert torch.isfinite(inputs.waveforms).all()


@pytest.mark.parametrize("coordinate_shift", [0.0, 100.0])
def test_arbitrary_query_assembly_requires_no_array_row(coordinate_shift: float) -> None:
    source, domain, _ = _source()
    observed_rows = np.flatnonzero(domain.observed_mask)
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m[observed_rows],
        domain.receiver_xy_m[observed_rows],
        azimuth_min_offset_m=0.1,
    )
    query = compute_trace_graph_geometry(
        np.array([[1.25 + coordinate_shift, 0.2]]),
        np.array([[1.25 + coordinate_shift, -1.8]]),
        azimuth_min_offset_m=0.1,
    )
    plan = build_trace_graph_subgraph(
        query,
        np.array([-17]),
        geometry,
        domain.trace_ids[observed_rows],
        np.ones(len(observed_rows), dtype=bool),
        rounds=2,
        relation_scales_m=np.ones((4, 2)),
        neighbors_per_relation=2,
    )
    ids = domain.trace_ids[observed_rows][::-1]
    values = source.read_observed_rows(ids)
    inputs = assemble_masked_trace_graph_inputs(
        plan,
        observed_trace_ids=ids,
        observed_waveforms=values,
        time_s=domain.time_s,
        preprocessing=source.preprocessing,
    )

    assert plan.trace_ids[plan.query_indices].tolist() == [-17]
    assert torch.count_nonzero(inputs.waveforms[inputs.query_indices]) == 0
    assert torch.equal(source.inputs(plan).waveforms, inputs.waveforms)
    if coordinate_shift:
        assert inputs.edge_index.shape == (2, 0)
        assert torch.count_nonzero(inputs.coverage) == 0


def test_duplicate_physical_query_alias_is_rejected_by_both_assembly_paths() -> None:
    source, domain, reader = _source()
    query = compute_trace_graph_geometry(
        domain.source_xy_m[:1], domain.receiver_xy_m[:1], azimuth_min_offset_m=0.1
    )
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=0.1
    )
    plan = build_trace_graph_subgraph(
        query,
        np.array([-1]),
        geometry,
        domain.trace_ids,
        domain.observed_mask,
        rounds=2,
        relation_scales_m=np.ones((4, 2)),
    )
    with pytest.raises(ValueError, match="duplicates an observed physical"):
        source.inputs(plan)
    assert reader.reads == []
    ids = domain.trace_ids[domain.observed_mask]
    with pytest.raises(ValueError, match="duplicates an observed physical"):
        assemble_masked_trace_graph_inputs(
            plan,
            observed_trace_ids=ids,
            observed_waveforms=source.read_observed_rows(ids),
            time_s=domain.time_s,
            preprocessing=source.preprocessing,
        )


def test_query_ids_cannot_be_declared_observed_to_low_level_assembly() -> None:
    source, domain, _ = _source()
    with pytest.raises(ValueError, match="query IDs"):
        assemble_masked_trace_graph_inputs(
            make_relational_trace_plan(domain),
            observed_trace_ids=domain.trace_ids,
            observed_waveforms=np.zeros((len(domain.trace_ids), 5), dtype=np.float32),
            time_s=domain.time_s,
            preprocessing=source.preprocessing,
        )


def test_forged_support_and_geometry_are_rejected_before_row_reads() -> None:
    source, domain, reader = _source()
    plan = make_relational_trace_plan(domain)
    bad_mask = plan.observed_mask.copy()
    bad_mask[plan.query_indices[0]] = True
    with pytest.raises(ValueError, match="outside the observed domain"):
        source.inputs(replace(plan, observed_mask=bad_mask))
    shifted = compute_trace_graph_geometry(
        plan.geometry.source_xy_m + 1, plan.geometry.receiver_xy_m, azimuth_min_offset_m=0.1
    )
    with pytest.raises(ValueError, match="geometry does not match"):
        source.inputs(replace(plan, geometry=shifted))
    assert reader.reads == []


def test_fixed_time_grid_is_checked_without_resampling() -> None:
    source, domain, reader = _source()
    with pytest.raises(ValueError, match="fixed preprocessing time grid"):
        MaskedTraceSource(
            replace(domain, time_s=domain.time_s + 0.001), source.preprocessing, reader
        )
    with pytest.raises(ValueError, match="fixed preprocessing time grid"):
        assemble_masked_trace_graph_inputs(
            make_relational_trace_plan(domain),
            observed_trace_ids=np.array([], dtype=np.int64),
            observed_waveforms=np.empty((0, 5), dtype=np.float32),
            time_s=domain.time_s[:-1],
            preprocessing=source.preprocessing,
        )
    assert reader.reads == []
