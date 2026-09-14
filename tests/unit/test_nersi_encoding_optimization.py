from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.configuration import REPOSITORY_ROOT, load_resolved_config
from seis_interp.models.nersi import FourierFeatureMapping, Nersi
from seis_interp.nersi_config import validate_nersi_poc_config
from seis_interp.processing.nersi_coordinate_mapping import nersi_profile_encoding
from seis_interp.training.c3_volume_nersi_data import C3VolumeNersiData
from seis_interp.training.fixed_step_nersi import train_nersi_fixed_steps
from seis_interp.training.nersi_checkpoints import (
    load_fixed_step_nersi_checkpoint,
    save_fixed_step_nersi_checkpoint,
)
from seis_interp.training.nersi_optimization import (
    nersi_step_learning_rate,
    validate_nersi_optimization,
)


def geometry():
    shape = (2, 3, 2, 8)
    sl, shot, rx, ry = np.indices(shape).reshape(4, -1)
    return pd.DataFrame(
        dict(
            source_x_m=100 + 80 * sl,
            source_y_m=1000 + 40 * shot,
            relative_receiver_x_m=-20 + 40 * rx,
            relative_receiver_y_m=-160 + 40 * ry,
        )
    ), shape


def options(schedule="cosine", decay=None):
    return dict(
        learning_rate_schedule=schedule,
        minimum_learning_rate=0.00001 if schedule == "cosine" else 0.001,
        ema_decay=decay,
    )


def test_cartesian_geometry_and_nyquist_are_mask_independent():
    table, shape = geometry()
    result = nersi_profile_encoding(table, shape, cartesian=True, fractions=[1, 0.5, 1])
    mapping = result["coordinate_mapping"]
    actual = np.array(mapping["normalized_anchors"])
    anchors = table.iloc[::8]
    physical = np.column_stack(
        (
            anchors.source_x_m + anchors.relative_receiver_x_m / 2,
            anchors.source_y_m + anchors.relative_receiver_y_m / 2,
            anchors.relative_receiver_x_m / 2,
        )
    )
    np.testing.assert_allclose(
        actual, (physical - physical.min(0)) / np.ptp(physical, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(result["axis_frequency_limits"], [np.pi / 0.8, np.pi, np.pi])
    assert mapping["decoder_half_offset_y_m"] == list(np.arange(-80, 80, 20))
    table["target_amplitudes"] = np.nan
    assert nersi_profile_encoding(table, shape, cartesian=True, fractions=[1, 0.5, 1]) == result


def test_non_affine_geometry_is_preserved():
    table, shape = geometry()
    before = nersi_profile_encoding(table, shape, cartesian=True)
    table.loc[:7, "source_y_m"] += 1
    after = nersi_profile_encoding(table, shape, cartesian=True)
    assert before != after
    assert after["coordinate_mapping"]["normalized_anchors"][0][1] == 1 / 80


def test_fourier_bandlimit_zeroes_features_above_each_axis_limit():
    mapping = FourierFeatureMapping(
        fourier_components=3,
        frequency_base=2,
        axis_frequency_limits=[0, 2 * np.pi, 4 * np.pi],
    )
    coords = torch.rand(2, 3, requires_grad=True)
    actual = mapping(coords).reshape(2, 3, 3, 2)
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert torch.count_nonzero(actual[:, 1, 1:]) == 0
    assert torch.count_nonzero(actual[:, 2, 2:]) == 0
    assert mapping.frequency_mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]
    actual.sum().backward()
    assert torch.isfinite(coords.grad).all()


def test_schedule_endpoints_and_monotonicity():
    rates = [nersi_step_learning_rate(s, 5, 0.001, options()) for s in range(1, 6)]
    assert rates[0] == 0.001 and rates[-1] == 0.00001
    assert all(a > b for a, b in zip(rates, rates[1:], strict=False))


@pytest.mark.parametrize(
    "changes",
    [
        {"ema_decay": 1},
        {"ema_decay": -1},
        {"ema_decay": True},
        {"minimum_learning_rate": 0.01},
        {"learning_rate_schedule": "unknown"},
    ],
)
def test_invalid_optimization(changes):
    with pytest.raises(ValueError):
        validate_nersi_optimization({**options(), **changes}, 0.001)


@pytest.mark.parametrize(
    "schedule,decay", [("constant", None), ("cosine", None), ("constant", 0.9), ("cosine", 0.9)]
)
def test_optimizer_ema_sampling_and_checkpoint_roundtrip(tmp_path, monkeypatch, schedule, decay):
    table, shape = geometry()
    encoded = nersi_profile_encoding(table, shape, cartesian=True, fractions=[1, 1, 1])
    coords = (np.indices(shape[:3]).reshape(3, -1).T / (np.array(shape[:3]) - 1)).astype(np.float32)
    mask = np.zeros((12, 8), dtype=bool)
    mask[:, ::2] = True
    profiles = np.ones((12, 1, 8, 8), dtype=np.float32)
    profiles[..., 1::2] = np.nan
    data = C3VolumeNersiData(
        coords, profiles, mask, np.arange(12, dtype=np.int64), 1.0, shape, (8, 8)
    )
    torch.manual_seed(1)
    model = Nersi(
        fourier_components=3,
        encoder_width=4,
        latent_channels=1,
        decoder_channels=(1, 1, 1),
        profile_shape=(8, 8),
        kernel_size=1,
        **encoded,
    )
    states = []
    step = torch.optim.Adam.step

    def capture(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        states.append([p.detach().clone() for p in model.parameters()])
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", capture)
    result = train_nersi_fixed_steps(
        model,
        data,
        device="cpu",
        learning_rate=0.001,
        profiles_per_step=2,
        max_steps=3,
        report_interval=1,
        random_seed=201,
        loss_name="masked_trace_mse",
        optimization=options(schedule, decay),
    )
    assert result.steps_completed == 3 and result.supervised_trace_presentations == 24
    expected = states[0]
    for state in states[1:]:
        expected = (
            state
            if decay is None
            else [a.lerp(b, 1 - decay) for a, b in zip(expected, state, strict=True)]
        )
    for actual, want in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    path = tmp_path / "final.pt"
    save_fixed_step_nersi_checkpoint(
        path,
        model,
        data,
        global_step=3,
        final_batch_loss=result.final_batch_loss,
        input_binding={
            "case_id": "case",
            "volume_id": "volume",
            "input_hashes": {"benchmark_case": "a" * 64, "benchmark_volume": {"index": "b" * 64}},
        },
        model_initialization_seed=1,
        sampling_seed=201,
        optimization=options(schedule, decay),
    )
    restored = load_fixed_step_nersi_checkpoint(path)
    assert restored.optimization == options(schedule, decay)
    torch.testing.assert_close(
        restored.model(torch.from_numpy(coords)), model(torch.from_numpy(coords)), rtol=0, atol=0
    )
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["coordinate_anchors"][0, 0] += 0.1
    torch.save(payload, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="fixed encoding buffer"):
        load_fixed_step_nersi_checkpoint(tmp_path / "tampered.pt")


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_coordinates", "cartesian_cmp_half_offset"),
        ("fourier_bandlimit", {"nyquist_fraction": [1, 0.5, 0.25]}),
        ("optimization", options("cosine", 0.99)),
    ],
)
def test_new_options_leave_frozen_configuration_unchanged(field, value):
    baseline = load_resolved_config(
        REPOSITORY_ROOT / "studies/study_037_c3_neural_mse_loss_ablation/formal/nersi.yaml"
    )
    original = deepcopy(baseline)
    validate_nersi_poc_config({**baseline, field: value})
    assert baseline == original
