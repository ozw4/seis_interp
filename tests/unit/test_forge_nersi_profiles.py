import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.processing.forge_mvp_contract import AXES, make_geometry, validate_mapping
from seis_interp.processing.forge_nersi_profiles import (
    ExactTraceProfileSampler,
    nersi_profile_layout,
)
from seis_interp.training.forge_mvp_methods import _fit_nersi_profile
from tests.fixtures.forge_mvp import tiny_forge_study

SHAPE = (2, 3, 2, 3)


def _geometry(shape, source_positions, receiver_lines):
    """Every cell of ``shape`` in C order; positions are keyed by grid indices."""
    grid = np.indices(shape).reshape(4, -1).T
    table = pd.DataFrame(grid, columns=AXES)
    table.insert(0, "cell_id", np.arange(len(table)))
    table[["source_line_continuous", "source_point_continuous"]] = [
        source_positions[line, point] for line, point in grid[:, :2]
    ]
    table["receiver_line_continuous"] = [receiver_lines[line, point] for line, point in grid[:, 2:]]
    # Never read by the layout: the receiver point is the profile column.
    table["receiver_point_continuous"] = grid[:, 3] + 0.37
    return table


def _positions(shape, seed=4):
    """Distinct perturbed actual positions of every source cell and receiver station."""
    lines, points, receiver_lines, receiver_points = shape
    rng = np.random.default_rng(seed)
    sources = {
        (line, point): (line + rng.uniform(-0.3, 0.3), point + rng.uniform(-0.3, 0.3))
        for line in range(lines)
        for point in range(points)
    }
    receivers = {
        (line, point): line + rng.uniform(-0.3, 0.3)
        for line in range(receiver_lines)
        for point in range(receiver_points)
    }
    return sources, receivers


def _expected_coordinates(shape, sources, receivers):
    """Profile p = (source_line * points + source_point) * receiver_lines + receiver_line."""
    lines, points, receiver_lines, receiver_points = shape
    rows = []
    for profile in range(lines * points * receiver_lines):
        line, point, receiver_line = np.unravel_index(profile, (lines, points, receiver_lines))
        mean = np.mean([receivers[receiver_line, p] for p in range(receiver_points)])
        rows.append((*sources[line, point], mean))
    return np.array(rows) / np.maximum(np.array([lines, points, receiver_lines]) - 1, 1)


def test_hand_built_profiles_hold_source_and_mean_receiver_line_positions():
    shape = (1, 3, 2, 2)
    sources = {(0, 0): (0.1, 0.4), (0, 1): (-0.2, 1.2), (0, 2): (0.05, 2.2)}
    receivers = {(0, 0): 0.2, (0, 1): -0.1, (1, 0): 1.3, (1, 1): 0.9}
    layout = nersi_profile_layout(_geometry(shape, sources, receivers), shape)
    # Scale max(n - 1, 1) = (1, 2, 1); receiver lines average to 0.05 and 1.1.
    expected = [
        [0.1, 0.2, 0.05],
        [0.1, 0.2, 1.1],
        [-0.2, 0.6, 0.05],
        [-0.2, 0.6, 1.1],
        [0.05, 1.1, 0.05],
        [0.05, 1.1, 1.1],
    ]
    np.testing.assert_allclose(layout.coordinates, expected, rtol=0, atol=1e-12)
    assert layout.receiver_points == 2


@pytest.mark.parametrize("shape", [SHAPE, (1, 4, 3, 2), (3, 1, 1, 4)])
def test_profile_coordinates_follow_profile_order_and_normalization(shape):
    sources, receivers = _positions(shape)
    geometry = _geometry(shape, sources, receivers)
    layout = nersi_profile_layout(geometry, shape)
    lines, points, receiver_lines, receiver_points = shape
    assert layout.coordinates.shape == (lines * points * receiver_lines, 3)
    assert layout.receiver_points == receiver_points
    np.testing.assert_allclose(
        layout.coordinates, _expected_coordinates(shape, sources, receivers), rtol=0, atol=1e-12
    )
    # Geometry row order never changes the layout.
    shuffled = geometry.sample(frac=1.0, random_state=3)
    np.testing.assert_array_equal(
        nersi_profile_layout(shuffled, shape).coordinates, layout.coordinates
    )


def test_receiver_line_mean_weights_each_station_once():
    sources, receivers = _positions(SHAPE)
    geometry = _geometry(SHAPE, sources, receivers)
    # Station (0, 0) survives in a single trace, station (1, 2) in none.
    station = geometry.receiver_grid_line.eq(0) & geometry.receiver_grid_point.eq(0)
    absent = geometry.receiver_grid_line.eq(1) & geometry.receiver_grid_point.eq(2)
    keep = (~station | geometry.cell_id.eq(geometry.cell_id[station].iloc[-1])) & ~absent
    layout = nersi_profile_layout(geometry[keep], SHAPE)
    line_zero = np.mean([receivers[0, p] for p in range(3)])
    line_one = np.mean([receivers[1, p] for p in range(2)])
    np.testing.assert_allclose(layout.coordinates[0::2, 2], line_zero, rtol=0, atol=1e-12)
    np.testing.assert_allclose(layout.coordinates[1::2, 2], line_one, rtol=0, atol=1e-12)


def test_locate_matches_the_c_order_cell_formula_and_the_row_positions():
    sources, receivers = _positions(SHAPE)
    geometry = _geometry(SHAPE, sources, receivers)
    layout = nersi_profile_layout(geometry, SHAPE)
    lines, points, receiver_lines, receiver_points = SHAPE
    profiles, columns = layout.locate(geometry.cell_id.to_numpy()[::-1])
    table = geometry.iloc[::-1]
    expected = (
        table.source_grid_line * points + table.source_grid_point
    ) * receiver_lines + table.receiver_grid_line
    np.testing.assert_array_equal(profiles, expected)
    np.testing.assert_array_equal(columns, table.receiver_grid_point)
    assert profiles.dtype == np.int64 and columns.dtype == np.int64
    for row, profile in zip(table.itertuples(), profiles, strict=True):
        line_mean = np.mean([receivers[row.receiver_grid_line, p] for p in range(receiver_points)])
        position = [row.source_line_continuous, row.source_point_continuous, line_mean]
        np.testing.assert_allclose(
            layout.coordinates[profile] * [lines - 1, points - 1, receiver_lines - 1],
            position,
            rtol=0,
            atol=1e-12,
        )
    single = layout.locate([0, 2, 3, 35])
    np.testing.assert_array_equal(single[0], [0, 0, 1, 11])
    np.testing.assert_array_equal(single[1], [0, 2, 0, 2])


def test_layout_accepts_the_real_mode_contract_geometry(tmp_path):
    _, config, _, _, mapping, stations, cells, grids, _ = tiny_forge_study(tmp_path)
    manifest = validate_mapping(mapping, stations, cells, grids, config)
    geometry = make_geometry(manifest, grids, "real")
    shape = tuple(config["spatial_shape"])
    layout = nersi_profile_layout(geometry, shape)
    scale = np.maximum(np.array(shape[:3]) - 1, 1)
    line_means = geometry.groupby(["receiver_grid_line", "receiver_grid_point"]).first()
    line_means = line_means.receiver_line_continuous.groupby(level=0).mean()
    profiles, _ = layout.locate(geometry.cell_id.to_numpy())
    expected = np.column_stack(
        (
            geometry.source_line_continuous,
            geometry.source_point_continuous,
            line_means[geometry.receiver_grid_line].to_numpy(),
        )
    )
    np.testing.assert_allclose(layout.coordinates[profiles], expected / scale, rtol=0, atol=1e-12)


def _swap_two_cell_ids(geometry):
    geometry.loc[[3, 4], "cell_id"] = geometry.loc[[4, 3], "cell_id"].to_numpy()


def _fortran_cell_ids(geometry):
    indices = geometry[AXES].to_numpy().T
    geometry["cell_id"] = np.ravel_multi_index(tuple(indices), SHAPE, order="F")


def _split_source_line(geometry):
    row = geometry.index[geometry.source_grid_line.eq(0) & geometry.source_grid_point.eq(1)][2]
    geometry.loc[row, "source_line_continuous"] += 1e-6


def _split_source_point(geometry):
    row = geometry.index[geometry.source_grid_line.eq(1) & geometry.source_grid_point.eq(2)][0]
    geometry.loc[row, "source_point_continuous"] -= 0.01


def _split_receiver_line(geometry):
    row = geometry.index[geometry.receiver_grid_line.eq(1) & geometry.receiver_grid_point.eq(0)][4]
    geometry.loc[row, "receiver_line_continuous"] += 0.01


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_swap_two_cell_ids, "C order"),
        (_fortran_cell_ids, "C order"),
        (_split_source_line, "one actual position"),
        (_split_source_point, "one actual position"),
        (_split_receiver_line, "one actual position"),
    ],
)
def test_inconsistent_geometry_is_rejected(mutate, match):
    geometry = _geometry(SHAPE, *_positions(SHAPE))
    mutate(geometry)
    with pytest.raises(ValueError, match=match):
        nersi_profile_layout(geometry, SHAPE)


@pytest.mark.parametrize(
    "missing",
    [
        lambda g: g.receiver_grid_line.eq(1),
        lambda g: g.receiver_grid_line.eq(0),
        lambda g: g.source_grid_line.eq(1) & g.source_grid_point.eq(2),
        lambda g: g.source_grid_line.eq(0) & g.source_grid_point.eq(1),
        lambda g: g.source_grid_line.eq(1),
    ],
    ids=[
        "last_receiver_line",
        "first_receiver_line",
        "last_source_cell",
        "middle_source_cell",
        "source_line",
    ],
)
def test_missing_receiver_line_or_source_cell_is_rejected(missing):
    geometry = _geometry(SHAPE, *_positions(SHAPE))
    with pytest.raises(ValueError, match="every source cell and receiver line"):
        nersi_profile_layout(geometry[~missing(geometry)], SHAPE)


class _RecordedVisits:
    """Generator proxy keeping every profile visit order the sampler draws."""

    def __init__(self, rng):
        self._rng = rng
        self.visits = []

    def permutation(self, n):
        visit = self._rng.permutation(n)
        self.visits.append(visit)
        return visit

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_each_sample_takes_exactly_k_rows_from_the_minimal_profile_prefix():
    profiles = np.array([7, 3, 7, 12, 3, 7, 20, 12, 7, 3, 25, 20])
    count_of = dict(zip(*np.unique(profiles, return_counts=True), strict=True))
    sampler = ExactTraceProfileSampler(profiles, 4, seed=11)
    assert set(sampler.eligible.tolist()) == set(count_of)
    sampler.rng = recorder = _RecordedVisits(sampler.rng)
    total, widths = 0, set()
    for _ in range(300):
        before = sampler.profiles_drawn
        rows = sampler.sample()
        drawn = sampler.profiles_drawn - before
        total += drawn
        assert rows.shape == (4,) and len(np.unique(rows)) == 4
        assert rows.min() >= 0 and rows.max() < len(profiles)
        visit = recorder.visits[-1]
        assert sorted(visit.tolist()) == list(range(len(count_of)))
        prefix = sampler.eligible[visit[:drawn]].tolist()
        held = [count_of[p] for p in prefix]
        assert sum(held) >= 4 > sum(held[:-1])
        assert set(profiles[rows].tolist()) <= set(prefix)
        widths.add(drawn)
    assert sampler.profiles_drawn == total
    assert widths == {1, 2, 3}


def test_equal_profiles_draw_the_ceiling_number_of_profiles():
    profiles = np.random.default_rng(2).permutation(np.repeat([4, 1, 9, 6], 3))
    for traces, drawn in ((1, 1), (3, 1), (4, 2), (7, 3), (12, 4)):
        sampler = ExactTraceProfileSampler(profiles, traces, seed=traces)
        for update in range(1, 21):
            rows = sampler.sample()
            assert sampler.profiles_drawn == update * drawn
            assert len(np.unique(profiles[rows])) <= drawn
            if traces == 3:
                # The single drawn profile is pooled whole: all three of its rows.
                np.testing.assert_array_equal(
                    np.sort(rows), np.flatnonzero(profiles == profiles[rows[0]])
                )
            if traces == 12:
                np.testing.assert_array_equal(np.sort(rows), np.arange(12))


def test_sampling_is_reproducible_by_seed():
    profiles = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 5])

    def draws(seed):
        sampler = ExactTraceProfileSampler(profiles, 4, seed=seed)
        return [sampler.sample().tolist() for _ in range(10)], sampler.profiles_drawn

    assert draws(5) == draws(5)
    assert draws(5)[0] != draws(6)[0]


@pytest.mark.parametrize(("count", "traces"), [(3, 4), (0, 1), (5, 0)])
def test_too_few_observed_traces_are_rejected(count, traces):
    with pytest.raises(ValueError, match="fewer observed traces"):
        ExactTraceProfileSampler(np.arange(count) % 2, traces, seed=0)


def test_balanced_profiles_include_every_row_uniformly():
    profiles = np.random.default_rng(8).permutation(np.repeat(np.arange(6) * 3 + 1, 4))
    sampler = ExactTraceProfileSampler(profiles, 5, seed=17)
    updates = 4000
    hits = np.zeros(len(profiles))
    for _ in range(updates):
        hits[sampler.sample()] += 1
    np.testing.assert_allclose(hits / updates, 5 / 24, atol=0.03)
    assert sampler.profiles_drawn == 2 * updates


def test_profiles_not_rows_are_visited_uniformly():
    # Lone row A and profile B with three rows, two traces per update: order A, B
    # pools all four rows (1/2 each); order B, A pools B alone (2/3 each).
    profiles = np.array([1, 0, 1, 1])
    sampler = ExactTraceProfileSampler(profiles, 2, seed=23)
    updates = 6000
    hits = np.zeros(4)
    for _ in range(updates):
        hits[sampler.sample()] += 1
    np.testing.assert_allclose(hits / updates, [7 / 12, 1 / 4, 7 / 12, 7 / 12], atol=0.03)
    assert abs(sampler.profiles_drawn / updates - 1.5) < 0.04


def _code(profile, column, time):
    """Exact float32 value of time sample ``time`` of column ``column`` of profile ``profile``."""
    return profile * 1000.0 + column * 10.0 + time + 1.0


def test_profile_gather_returns_the_trace_of_each_profile_and_column():
    time_count, receiver_points = 5, 3
    rng = np.random.default_rng(1)
    profiles = rng.integers(10, 20, size=12)
    columns = rng.integers(0, receiver_points, size=12)
    rows = rng.choice(12, size=7, replace=False)
    selected, inverse = np.unique(profiles[rows], return_inverse=True)
    output = torch.full((len(selected), 1, 8, 8), float("nan"))
    for i, profile in enumerate(selected):
        for column in range(receiver_points):
            output[i, 0, :time_count, column] = torch.from_numpy(
                _code(profile, column, np.arange(time_count))
            )
    decoded = output[:, 0, :time_count, :receiver_points].transpose(1, 2)
    gathered = decoded[torch.from_numpy(inverse), torch.from_numpy(columns[rows])]
    expected = _code(profiles[rows, None], columns[rows, None], np.arange(time_count))
    np.testing.assert_array_equal(gathered.numpy(), expected.astype(np.float32))


class _ProfileCode(torch.nn.Module):
    """Decode profile p as ``_code(p, column, time)`` on a padded canvas; padding is NaN."""

    def __init__(self, coordinates, time_count, receiver_points):
        super().__init__()
        self.register_buffer("table", torch.from_numpy(coordinates.astype(np.float32)))
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.time_count, self.receiver_points = time_count, receiver_points
        self.calls = []

    def forward(self, coordinates):
        match = (coordinates[:, None, :] == self.table[None]).all(-1)
        if not torch.equal(match.sum(1), torch.ones(len(coordinates), dtype=torch.long)):
            raise AssertionError("decoded coordinates are not exactly one profile each")
        profile = match.long().argmax(1).numpy()
        self.calls.append(profile.tolist())
        values = torch.full((len(profile), 1, 8, 8), float("nan"))
        grid = _code(
            profile[:, None, None],
            np.arange(self.receiver_points)[None, None, :],
            np.arange(self.time_count)[None, :, None],
        )
        values[:, 0, : self.time_count, : self.receiver_points] = torch.from_numpy(
            grid.astype(np.float32)
        )
        return values + self.offset


def test_fit_supervises_and_predicts_the_trace_of_each_cell():
    shape, time_count, rms = SHAPE, 5, 2.0
    geometry = _geometry(shape, *_positions(shape))
    observed = np.sort(np.random.default_rng(6).choice(36, size=14, replace=False))
    test = np.setdiff1d(np.arange(36), observed)
    layout = nersi_profile_layout(geometry, shape)
    profiles, columns = layout.locate(observed)
    inputs = ForgeModelInputs(
        geometry=geometry,
        observed_cells=observed,
        test_cells=test,
        observed_values=(
            _code(profiles[:, None], columns[:, None], np.arange(time_count)) * rms
        ).astype(np.float32),
        spatial_shape=shape,
        time_s=np.arange(time_count) * 0.002,
        global_rms=rms,
        binding={},
    )
    config = {
        "device": "cpu",
        "model_seed": 7301,
        "loss": "masked_trace_relative_mse",
        "training": {
            "optimizer": "adam",
            "supervised_traces_per_update": 4,
            "learning_rate": 0.0,
            "weight_decay": 0.0,
            "max_steps": 6,
            "report_interval": 2,
            "mixed_precision": "off",
            "ema_decay": None,
        },
        "prediction": {"profile_batch_size": 2},
    }
    model = _ProfileCode(layout.coordinates, time_count, 3)
    prediction, history, shadows, records, states = _fit_nersi_profile(
        model, inputs, config, lambda _: None, (), ()
    )
    # A wrong (profile, column) pair gives a nonzero loss; a padded sample a nonfinite one.
    assert [row["loss"] for row in history] == [0.0, 0.0, 0.0]
    assert records["final_training_loss_window_mean"] == 0.0
    test_profiles, test_columns = layout.locate(test)
    np.testing.assert_array_equal(
        prediction,
        (_code(test_profiles[:, None], test_columns[:, None], np.arange(time_count)) * rms).astype(
            np.float32
        ),
    )
    training_calls, prediction_calls = model.calls[:6], model.calls[6:]
    observed_profiles = set(profiles.tolist())
    for call in training_calls:
        assert call == sorted(set(call)) and set(call) <= observed_profiles and len(call) <= 4
    assert all(len(call) <= 2 for call in prediction_calls)
    assert sum(prediction_calls, []) == sorted(set(test_profiles.tolist()))
    decoded = sum(len(call) for call in training_calls) / 6
    assert records["mean_profiles_decoded_per_update"] == decoded
    assert records["mean_profiles_drawn_per_update"] >= decoded
    assert shadows == {} and states == {}
