"""Observed-only PoC graph domains, fixed coordinates, and episode boundaries."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from seis_interp.data.c3_poc_inputs import load_c3_random80_poc_inputs
from seis_interp.data.c3_poc_trace_graph import build_c3_poc_trace_graph_training_data
from seis_interp.processing.c3_benchmark_contract import C3BenchmarkDimensions
from seis_interp.processing.trace_graph_geometry import (
    build_trace_graph_node_features,
    compute_trace_graph_geometry,
)
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph
from seis_interp.training.amplitude_scaling import compute_observed_global_rms
from seis_interp.training.c3_poc_trace_graph_episodes import (
    PocTraceGraphEpisodeGenerator,
    PocTraceGraphEpisodeLabelReader,
    build_poc_trace_graph_context_source,
)
from tests.fixtures.c3_volume_run_artifacts import prepare_c3_volume_run_artifacts


def _inputs(path, *, target_offset=0.0):
    artifacts = prepare_c3_volume_run_artifacts(
        path, dataset_id="seg_c3_na", missing_fraction=0.8, target_offset=target_offset
    )
    metadata = artifacts.volume_metadata
    selection = metadata["selection"]
    start, stop = selection["source_line"]
    return load_c3_random80_poc_inputs(
        interim_dir=artifacts.interim,
        processed_dir=artifacts.processed,
        mask_dir=artifacts.mask,
        case_dir=artifacts.case,
        volume_dir=artifacts.volume,
        config={"benchmark_volume": {"selection": selection}},
        dimensions=C3BenchmarkDimensions(
            time_range=tuple(selection["time"]),
            sail_line_numbers=(start, stop - 1),
            shape=tuple(metadata["shape"]),
        ),
    )


@pytest.fixture
def inputs(tmp_path):
    return _inputs(tmp_path)


def _training(inputs, *, scale=7.0):
    return build_c3_poc_trace_graph_training_data(
        inputs,
        amplitude_scale=scale,
        position_scale_m=100.0,
        offset_scale_m=50.0,
        azimuth_min_offset_m=0.1,
    )


def _generator(training, seed=41, fraction=0.5):
    return PocTraceGraphEpisodeGenerator(training, random_seed=seed, missing_fraction=fraction)


def _plan(training, episode):
    domain = training.domain
    positions = {int(value): i for i, value in enumerate(domain.trace_ids)}
    rows = [positions[int(value)] for value in episode.query_trace_ids]
    geometry = compute_trace_graph_geometry(
        domain.source_xy_m, domain.receiver_xy_m, azimuth_min_offset_m=0.1
    )
    queries = compute_trace_graph_geometry(
        domain.source_xy_m[rows], domain.receiver_xy_m[rows], azimuth_min_offset_m=0.1
    )
    return build_trace_graph_subgraph(
        queries,
        episode.query_trace_ids,
        geometry,
        domain.trace_ids,
        episode.visible_mask,
        rounds=2,
        relation_scales_m=np.full((4, 2), 10000.0),
        neighbors_per_relation=2,
    )


def test_domain_contains_exactly_observed_ids_and_owned_physical_samples(inputs, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("PoC graph construction must not read files or fit a training RMS")

    monkeypatch.setattr(np, "load", fail)
    monkeypatch.setattr(
        "seis_interp.processing.trace_graph_preprocessing.fit_trace_graph_preprocessing", fail
    )
    training = _training(inputs)
    volume = inputs.observed_volume
    mask = volume.observed_trace_mask.reshape(-1)
    expected = volume.values.reshape(len(volume.time_s), -1)[:, mask].T
    np.testing.assert_array_equal(training.domain.trace_ids, volume.array_rows.reshape(-1)[mask])
    np.testing.assert_array_equal(training.observed_amplitudes, expected)
    np.testing.assert_array_equal(training.domain.time_s, volume.time_s)
    assert training.domain.time_samples == (0, len(volume.time_s))
    assert training.domain.amplitudes_path is None
    assert training.domain.pool is None
    assert inputs.case["partition"] == "test"
    assert training.preprocessing.amplitude_scale == 7.0
    assert not np.shares_memory(training.observed_amplitudes, volume.values)
    source = training.source(training.domain.observed_mask)
    with pytest.raises(ValueError, match="outside the observed domain"):
        source.read_observed_rows(volume.array_rows.reshape(-1)[~mask][:1])


def test_episode_masks_redraw_with_local_seed_and_ignore_amplitude(inputs):
    training = _training(inputs)
    one, two = _generator(training), _generator(training)
    other = _generator(training, seed=59)
    sequence = []
    different = []
    for number in range(1, 6):
        left = one.next_episode()
        training.observed_amplitudes[:] *= -3
        np.random.seed(number)
        torch.manual_seed(number)
        right = two.next_episode()
        assert left.episode_id == right.episode_id == number
        assert left.kind == "random_trace"
        assert left.missing_fraction == 0.5
        assert 0 < len(left.hidden_trace_ids) < len(training.domain.trace_ids)
        assert set(left.hidden_trace_ids) < set(training.domain.trace_ids)
        np.testing.assert_array_equal(left.visible_mask, right.visible_mask)
        np.testing.assert_array_equal(left.query_trace_ids, right.query_trace_ids)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(left.query_batches(2)))), left.hidden_trace_ids
        )
        sequence.append(tuple(left.hidden_trace_ids))
        different.append(tuple(other.next_episode().hidden_trace_ids))
    assert len(set(sequence)) > 1
    assert sequence != different


def test_hidden_values_are_available_only_to_labels_not_context_or_graph(inputs):
    training = _training(inputs)
    episode = _generator(training).next_episode()
    source = build_poc_trace_graph_context_source(training, episode)
    reader = PocTraceGraphEpisodeLabelReader(training, episode)
    plan = _plan(training, episode)
    graph = source.inputs(plan)
    assert len(plan.edge_type) > 0
    assert not np.isin(plan.trace_ids[plan.edge_index[0]], episode.hidden_trace_ids).any()
    assert torch.count_nonzero(graph.waveforms[graph.query_indices]) == 0
    positions = {int(value): i for i, value in enumerate(training.domain.trace_ids)}
    rows = [positions[int(value)] for value in episode.query_trace_ids]
    expected = (training.observed_amplitudes[rows].astype(np.float64) / 7).astype(np.float32)
    np.testing.assert_array_equal(reader.read(episode.query_trace_ids).numpy(), expected)
    visible_ids = plan.trace_ids[plan.observed_mask]
    expected_context = training.observed_amplitudes[[positions[int(i)] for i in visible_ids]]
    np.testing.assert_array_equal(
        graph.waveforms[graph.observed_mask].numpy(),
        (expected_context.astype(np.float64) / 7).astype(np.float32),
    )
    with pytest.raises(ValueError, match="outside the observed domain"):
        source.read_observed_rows(episode.hidden_trace_ids)
    with pytest.raises(ValueError, match="outside the observed domain"):
        reader.read(training.domain.trace_ids[episode.visible_mask][:1])
    targets = inputs.observed_volume.array_rows[inputs.observed_volume.evaluation_target_trace_mask]
    with pytest.raises(ValueError, match="outside the observed domain"):
        reader.read(targets[:1])
    assert source._amplitudes.shape[0] == episode.visible_mask.sum()
    training.observed_amplitudes[~episode.visible_mask] = 1e20
    after = build_poc_trace_graph_context_source(training, episode).inputs(plan)
    for field in fields(graph):
        if isinstance(getattr(graph, field.name), torch.Tensor):
            assert torch.equal(getattr(graph, field.name), getattr(after, field.name))


def test_full_domain_bounds_do_not_depend_on_outer_mask(inputs):
    first = _training(inputs)
    volume = inputs.observed_volume
    opposite = replace(
        volume,
        observed_trace_mask=volume.evaluation_target_trace_mask.copy(),
        evaluation_target_trace_mask=volume.observed_trace_mask.copy(),
    )
    second = _training(replace(inputs, observed_volume=opposite))
    assert first.preprocessing == second.preprocessing
    table = inputs.index_table
    source = table[["source_x_m", "source_y_m"]].to_numpy()
    relative = table[["relative_receiver_x_m", "relative_receiver_y_m"]].to_numpy()
    geometry = compute_trace_graph_geometry(source, source + relative, azimuth_min_offset_m=0.1)
    midpoint = geometry.midpoint_xy_m
    np.testing.assert_array_equal(
        first.preprocessing.midpoint_origin_m, (midpoint.min(axis=0) + midpoint.max(axis=0)) / 2
    )
    features = []
    for training in (first, second):
        p = training.preprocessing
        features.append(
            build_trace_graph_node_features(
                geometry,
                midpoint_origin_m=np.asarray(p.midpoint_origin_m),
                position_scale_m=p.position_scale_m,
                offset_scale_m=p.offset_scale_m,
                observed_mask=np.ones(len(table), dtype=bool),
            )
        )
    np.testing.assert_array_equal(*features)


def test_target_truth_changes_do_not_reach_training_or_episode_objects(tmp_path):
    left = _training(_inputs(tmp_path / "base"))
    right = _training(_inputs(tmp_path / "changed", target_offset=1000000))
    np.testing.assert_array_equal(left.domain.trace_ids, right.domain.trace_ids)
    np.testing.assert_array_equal(left.observed_amplitudes, right.observed_amplitudes)
    a, b = _generator(left).next_episode(), _generator(right).next_episode()
    np.testing.assert_array_equal(a.hidden_trace_ids, b.hidden_trace_ids)
    ga = build_poc_trace_graph_context_source(left, a).inputs(_plan(left, a))
    gb = build_poc_trace_graph_context_source(right, b).inputs(_plan(right, b))
    assert torch.equal(ga.waveforms, gb.waveforms)
    assert torch.equal(ga.node_features, gb.node_features)
    assert torch.equal(
        PocTraceGraphEpisodeLabelReader(left, a).read(a.query_trace_ids),
        PocTraceGraphEpisodeLabelReader(right, b).read(b.query_trace_ids),
    )
    assert np.max(np.abs(right.observed_amplitudes)) < 1000000


def test_target_storage_sentinels_are_discarded_and_shared_rms_is_not_refitted(inputs):
    volume = inputs.observed_volume
    scale = compute_observed_global_rms(volume.values, volume.observed_trace_mask)
    before = _training(inputs, scale=scale)
    volume.values[:, volume.evaluation_target_trace_mask] = np.nan
    after = _training(inputs, scale=scale)
    assert after.preprocessing.amplitude_scale == scale
    np.testing.assert_array_equal(before.observed_amplitudes, after.observed_amplitudes)
    episode = _generator(after).next_episode()
    reader = PocTraceGraphEpisodeLabelReader(after, episode)
    expected_source = after.source(after.domain.observed_mask)
    expected = (
        expected_source.read_observed_rows(episode.query_trace_ids).astype(np.float64) / scale
    )
    np.testing.assert_array_equal(reader.read(episode.query_trace_ids), expected.astype(np.float32))
    assert torch.isfinite(
        build_poc_trace_graph_context_source(after, episode).inputs(_plan(after, episode)).waveforms
    ).all()


@pytest.mark.parametrize("observed_count", [0, 1, 2])
def test_minimal_observed_domain_requires_query_and_context(inputs, observed_count):
    volume = inputs.observed_volume
    mask = np.zeros_like(volume.observed_trace_mask)
    mask.reshape(-1)[np.flatnonzero(volume.observed_trace_mask)[:observed_count]] = True
    inputs = replace(
        inputs,
        observed_volume=replace(
            volume, observed_trace_mask=mask, evaluation_target_trace_mask=~mask
        ),
    )
    if observed_count < 2:
        with pytest.raises(ValueError, match="at least two traces"):
            _training(inputs)
    else:
        episode = _generator(_training(inputs)).next_episode()
        assert episode.visible_mask.sum() == len(episode.hidden_trace_ids) == 1


@pytest.mark.parametrize("fraction", [0, 1, -0.1, np.nan, True, 0.00001, 0.99999])
def test_invalid_or_degenerate_mask_fraction_fails_before_training(inputs, fraction):
    with pytest.raises(ValueError, match="missing_fraction"):
        _generator(_training(inputs), fraction=fraction)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_is_rejected(inputs, seed):
    with pytest.raises(ValueError, match="random_seed"):
        _generator(_training(inputs), seed=seed)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_external_rms_must_be_valid(inputs, scale):
    with pytest.raises(ValueError, match="amplitude_scale"):
        _training(inputs, scale=scale)


@pytest.mark.parametrize("mutation", ["kind", "ids", "visibility", "hidden", "query"])
def test_episode_binding_rejects_foreign_or_inconsistent_masks(inputs, mutation):
    training = _training(inputs)
    episode = _generator(training).next_episode()
    updates = {
        "kind": {"kind": "random_whole_ffid"},
        "ids": {"trace_ids": episode.trace_ids[::-1]},
        "visibility": {"visible_mask": np.ones_like(episode.visible_mask)},
        "hidden": {"hidden_trace_ids": np.append(episode.hidden_trace_ids, 99999)},
        "query": {"query_trace_ids": episode.query_trace_ids[:-1]},
    }
    episode = replace(episode, **updates[mutation])
    with pytest.raises(ValueError, match="episode"):
        build_poc_trace_graph_context_source(training, episode)
    with pytest.raises(ValueError, match="episode"):
        PocTraceGraphEpisodeLabelReader(training, episode)
