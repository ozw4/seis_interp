"""``cinr_real`` reads observed neighbours, never its own trace, and honours its config."""

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.data.forge_mvp_artifacts import ForgeModelInputs
from seis_interp.forge_mvp_config import validate_field_method_config
from seis_interp.models.trace_context_field import TraceContextField
from seis_interp.training import forge_mvp_methods
from seis_interp.training.forge_mvp_methods import _fit_context_field

COLUMNS = [
    f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
]
SHAPE = (2, 4, 2, 4)
TIME_COUNT = 12
TRACES = 4
NEIGHBORS = 3
STEPS = 5
BUDGET = {
    "max_steps": STEPS,
    "learning_rate": 0.001,
    "loss_traces_per_update": TRACES,
    "parameter_count_target": 1500,
    "parameter_count_tolerance": 0.05,
}


# One index step is 10 m along a line and 1000 m along a point, while index
# normalization weights a line step (1 / 1) above a point step (1 / 3). The two
# rankings therefore disagree, which is what makes the declared choice testable.
LINE_SPACING_M = 10.0
POINT_SPACING_M = 1000.0


def _inputs():
    size = int(np.prod(SHAPE))
    axes = np.array(np.unravel_index(np.arange(size), SHAPE)).T
    geometry = pd.DataFrame({"cell_id": np.arange(size)})
    for index, column in enumerate(COLUMNS):
        geometry[column] = axes[:, index].astype(np.float64)
    for role, (line, point) in (("source", (0, 1)), ("receiver", (2, 3))):
        geometry[f"{role}_x_m"] = axes[:, line] * LINE_SPACING_M
        geometry[f"{role}_y_m"] = axes[:, point] * POINT_SPACING_M
    observed = np.arange(0, size, 2, dtype=np.int64)
    test = np.setdiff1d(np.arange(size, dtype=np.int64), observed)
    rng = np.random.default_rng(3)
    return ForgeModelInputs(
        geometry=geometry,
        observed_cells=observed,
        test_cells=test,
        observed_values=rng.normal(size=(len(observed), TIME_COUNT)).astype(np.float32),
        spatial_shape=SHAPE,
        time_s=np.arange(TIME_COUNT) * 0.004,
        global_rms=1.0,
        binding={},
    )


def _model():
    torch.manual_seed(7301)
    return TraceContextField(
        time_sample_count=TIME_COUNT,
        fourier_components=4,
        width=8,
        stem_kernel_size=3,
        time_downsample_factor=2,
    )


def _fit_config(**training):
    return {
        "device": "cpu",
        "model_seed": 7301,
        "loss": "masked_trace_mse",
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.01,
            "max_steps": STEPS,
            "batch_size": TRACES,
            "weight_decay": 0.01,
            "ema_decay": 0.9,
            "report_interval": 1,
        }
        | training,
        "prediction": {"batch_size": 4},
        "context": {"neighbor_count": NEIGHBORS},
    }


def _fit_inr_context(config=None, decays=(), snapshots=()):
    return _fit_context_field(
        _model(), _inputs(), config or _fit_config(), lambda _: None, decays, snapshots
    )


def test_the_fit_returns_test_predictions_shadows_and_the_zero_predictor_gate():
    prediction, history, shadows, records, _ = _fit_inr_context(decays=(0.5, 0.9))
    inputs = _inputs()
    assert prediction.shape == (len(inputs.test_cells), TIME_COUNT)
    assert prediction.dtype == np.float32 and np.isfinite(prediction).all()
    assert len(history) == STEPS
    assert sorted(shadows) == ["0.5"]
    assert records["context_neighbor_count"] == NEIGHBORS
    assert records["runtime_neighbor_selection_seconds"] >= 0
    assert records["g0"]["zero_predictor_loss"] > 0
    assert records["would_clip_threshold"] == 1.0


def test_a_query_is_never_reconstructed_from_its_own_trace(monkeypatch):
    calls = []
    original = forge_mvp_methods.nearest_observed_positions

    def recording(coordinates, observed_rows, query_rows, count, **kwargs):
        calls.append((observed_rows.copy(), query_rows.copy(), count))
        return original(coordinates, observed_rows, query_rows, count, **kwargs)

    monkeypatch.setattr(forge_mvp_methods, "nearest_observed_positions", recording)
    _fit_inr_context()
    inputs = _inputs()
    assert len(calls) == 2
    observed_call, query_call = calls
    # The observed neighbourhood uses the observed set as both query and candidates, so
    # nearest_observed_positions excludes each query from its own context.
    np.testing.assert_array_equal(observed_call[0], observed_call[1])
    np.testing.assert_array_equal(query_call[0], observed_call[0])
    assert len(query_call[1]) == len(inputs.test_cells)
    assert observed_call[2] == query_call[2] == NEIGHBORS
    positions = original(
        np.zeros((int(np.prod(SHAPE)), 4)), observed_call[0], observed_call[1], NEIGHBORS
    )
    assert positions.shape == (len(observed_call[0]), NEIGHBORS)


@pytest.mark.parametrize(
    "optimizer,used,unused",
    [("adamw", torch.optim.AdamW, torch.optim.Adam), ("adam", torch.optim.Adam, torch.optim.AdamW)],
)
def test_the_declared_optimizer_is_the_one_used(monkeypatch, optimizer, used, unused):
    steps = []
    for kind in (used, unused):
        original = kind.step
        monkeypatch.setattr(
            kind,
            "step",
            lambda self, *a, _k=kind, _o=original, **kw: (steps.append(_k), _o(self))[1],
        )
    _fit_inr_context(_fit_config(optimizer=optimizer))
    assert [kind for kind in steps if kind is used] == [used] * STEPS
    assert unused not in steps


def test_an_invalid_optimizer_is_rejected():
    with pytest.raises(ValueError, match="adam or adamw"):
        _fit_inr_context(_fit_config(optimizer="sgd"))


def test_the_declared_loss_is_the_one_used(monkeypatch):
    seen = []
    original = forge_mvp_methods.masked_trace_loss

    def recording(prediction, target, *args, **kwargs):
        seen.append((kwargs.get("loss_name"), tuple(prediction.shape)))
        return original(prediction, target, *args, **kwargs)

    monkeypatch.setattr(forge_mvp_methods, "masked_trace_loss", recording)
    _fit_inr_context()
    assert {name for name, _ in seen} == {"masked_trace_mse"}
    assert {shape for _, shape in seen} == {(TRACES, TIME_COUNT)}


def _method_config(**overrides):
    config = {
        "method_id": "cinr_real",
        "geometry_mode": "real",
        "device": "cpu",
        "loss": "masked_trace_mse",
        "checkpoint_rule": "fixed_final_ema",
        "model_seed": 7301,
        "model": {
            "input_features": 4,
            "fourier_components": 4,
            "frequency_base": 1.1,
            "width": 8,
            "stem_kernel_size": 3,
            "time_downsample_factor": 2,
            "activation": "gelu",
            "output_activation": "linear",
            "time_sample_count": TIME_COUNT,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "max_steps": STEPS,
            "weight_decay": 0.0,
            "report_interval": 1,
            "learning_rate_schedule": "constant",
            "mixed_precision": "off",
            "ema_decay": 0.9,
            "batch_size": TRACES,
        },
        "prediction": {"batch_size": 4},
        "context": {"neighbor_count": NEIGHBORS},
    }
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(section), dict):
            config[section].update(value)
        else:
            config[section] = value
    return config


def _contract():
    return {
        "model_seed": 7301,
        "time_sample_count": TIME_COUNT,
        "matched_training": BUDGET,
        "observed_trace_count": 16,
    }


def test_the_declared_method_config_is_accepted():
    validate_field_method_config(_method_config(), _contract(), loss_name="masked_trace_mse")


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"context": {"radius": 2}}, "context.neighbor_count"),
        ({"context": {"neighbor_count": 0}}, "context.neighbor_count"),
        ({"context": {"neighbor_count": True}}, "context.neighbor_count"),
        ({"model": {"time_sample_count": TIME_COUNT + 1}}, "complete traces"),
        ({"geometry_mode": "grid"}, "unsupported field method or geometry"),
        ({"training": {"batch_size": TRACES + 1}}, "budgeted loss traces"),
        ({"loss": "masked_trace_relative_mse"}, "declared training loss"),
    ],
)
def test_structural_mistakes_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_field_method_config(
            _method_config(**overrides), _contract(), loss_name="masked_trace_mse"
        )


def test_only_the_conditioned_field_declares_a_context():
    config = _method_config()
    del config["context"]
    with pytest.raises(ValueError, match="only the conditioned field declares a context"):
        validate_field_method_config(config, _contract(), loss_name="masked_trace_mse")


def test_the_declared_distance_selects_the_neighbourhood(monkeypatch):
    """``physical`` ranks on station metres; the field still reads the index coordinates."""
    seen = {}
    original = forge_mvp_methods.nearest_observed_positions

    def recording(coordinates, observed_rows, query_rows, count, **kwargs):
        seen.setdefault("ranking", np.array(coordinates, dtype=np.float64))
        return original(coordinates, observed_rows, query_rows, count, **kwargs)

    monkeypatch.setattr(forge_mvp_methods, "nearest_observed_positions", recording)
    inputs = _inputs()
    rankings, neighbourhoods = {}, {}
    for distance in ("index", "physical"):
        seen.clear()
        config = _fit_config()
        config["context"] = {"neighbor_count": NEIGHBORS, "distance": distance}
        _, _, _, records, _ = _fit_inr_context(config)
        assert records["context_distance"] == distance
        rankings[distance] = seen["ranking"]
        neighbourhoods[distance] = original(
            seen["ranking"],
            np.arange(len(inputs.observed_cells)) * 2,
            np.arange(len(inputs.observed_cells)) * 2,
            NEIGHBORS,
        )
    # the index ranking is the normalized local grid; the physical one is in metres
    assert rankings["index"].max() <= 1.0
    assert rankings["physical"].max() >= POINT_SPACING_M
    # and they disagree, so the declaration is not cosmetic
    assert not np.array_equal(neighbourhoods["index"], neighbourhoods["physical"])


def test_an_unknown_distance_is_rejected():
    config = _fit_config()
    config["context"] = {"neighbor_count": NEIGHBORS, "distance": "offset"}
    with pytest.raises(ValueError, match="index or physical"):
        _fit_inr_context(config)
    with pytest.raises(ValueError, match="index or physical"):
        validate_field_method_config(
            _method_config(context={"distance": "offset"}),
            _contract(),
            loss_name="masked_trace_mse",
        )


def test_the_declared_distance_defaults_to_the_index_ranking():
    _, _, _, records, _ = _fit_inr_context(_fit_config())
    assert records["context_distance"] == "index"
    validate_field_method_config(_method_config(), _contract(), loss_name="masked_trace_mse")
