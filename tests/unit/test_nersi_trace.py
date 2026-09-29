import hashlib
from math import atanh
from math import ceil as math_ceil

import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.models.nersi_trace import NersiTrace, band_taper
from seis_interp.processing.nersi_trace_coordinates import (
    head_samples_per_radius,
    linear_moveout_samples_per_radius,
    midpoint_offset_transform,
    slowness_samples_per_radius,
)


def test_four_continuous_coordinates_and_odd_full_time_axis():
    torch.manual_seed(7301)
    model = NersiTrace(
        time_sample_count=9,
        fourier_components=2,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
    )
    coordinates = torch.tensor([[0.15, 0.2, 0.3, 0.4], [0.15, 0.2, 0.3, 0.401]], requires_grad=True)
    prediction = model(coordinates)
    assert prediction.shape == (2, 9)
    assert not torch.equal(prediction[0], prediction[1])
    prediction.square().mean().backward()
    assert torch.isfinite(coordinates.grad).all()
    assert coordinates.grad.abs().sum(dim=0).gt(0).all()


def test_optional_affine_encoding_matches_explicit_coordinates_and_preserves_default_state():
    config = dict(
        time_sample_count=9,
        fourier_components=2,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
    )
    torch.manual_seed(7301)
    original = NersiTrace(**config)
    matrix = torch.tensor(
        [[0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [-0.5, 0, 0.5, 0, 0.5], [0, -0.5, 0, 0.5, 0.5]]
    )
    torch.manual_seed(7301)
    transformed = NersiTrace(**config, coordinate_transform=matrix)
    assert "coordinate_transform" not in original.state_dict()
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(transformed.state_dict()[name], tensor, rtol=0, atol=0)
    coordinates = torch.rand(3, 4, requires_grad=True)
    expected = original(coordinates @ matrix[:, :4].T + matrix[:, 4])
    actual = transformed(coordinates)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert torch.isfinite(coordinates.grad).all()


@pytest.mark.parametrize("matrix", [np.zeros((3, 5)), np.full((4, 5), np.nan)])
def test_invalid_coordinate_transform(matrix):
    with pytest.raises(ValueError, match="finite 4 by 5"):
        NersiTrace(time_sample_count=9, coordinate_transform=matrix)


@pytest.mark.parametrize("polar", [False, True])
def test_midpoint_offset_map_reproduces_actual_geometry_without_waveforms(polar):
    rng = np.random.default_rng(7)
    local = rng.uniform(-0.1, 1.1, size=(32, 4))
    source = local[:, :2] @ np.array([[20, 100], [70, -15]]) + [335000, 4260000]
    receiver = local[:, 2:] @ np.array([[15, 50], [110, -30]]) + [335050, 4260040]
    shape = (6, 32, 8, 32)
    columns = [
        f"{role}_{axis}_continuous" for role in ("source", "receiver") for axis in ("line", "point")
    ]
    geometry = pd.DataFrame(local * (np.array(shape) - 1), columns=columns)
    geometry[["source_x_m", "source_y_m"]] = source
    geometry[["receiver_x_m", "receiver_y_m"]] = receiver
    physical = np.column_stack(((source + receiver) / 2, (receiver - source) / 2))
    expected = (physical - physical.min(axis=0)) / np.ptp(physical, axis=0)
    if polar:
        expected[:, 2:] = physical[:, 2:] / np.linalg.norm(physical[:, 2:], axis=1).max()
    matrix = midpoint_offset_transform(geometry, shape, polar=polar)
    np.testing.assert_allclose(local @ matrix[:, :4].T + matrix[:, 4], expected, atol=1e-9)
    geometry.loc[0, "source_x_m"] += 1
    with pytest.raises(ValueError, match="not affine"):
        midpoint_offset_transform(geometry, shape)


def test_polar_fourier_coordinates_preserve_radius_and_signed_azimuth():
    identity = torch.cat((torch.eye(4), torch.zeros(4, 1)), dim=1)
    model = NersiTrace(
        time_sample_count=9,
        fourier_components=2,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
        coordinate_transform=identity,
        polar_offset=True,
    )
    coordinates = torch.tensor(
        [[0.2, 0.3, 0.3, 0.4], [0.1, 0.5, -0.3, -0.4], [0.1, 0.5, 0, 0]], requires_grad=True
    )
    captured = []
    handle = model.fourier_mapping.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach())
    )
    prediction = model(coordinates)
    handle.remove()
    torch.testing.assert_close(captured[0][:, 2], torch.tensor([0.5, 0.5, 0.0]))
    torch.testing.assert_close(captured[0][:, 3], torch.tensor([0.29516724, -0.70483276, 0.0]))
    prediction.square().mean().backward()
    assert torch.isfinite(coordinates.grad).all()


def test_fractional_moveout_keeps_every_requested_time_sample_and_gradients():
    model = NersiTrace(
        time_sample_count=9,
        fourier_components=2,
        encoder_width=4,
        latent_channels=2,
        decoder_channels=[2, 2, 2],
        kernel_size=1,
        coordinate_transform=torch.cat((torch.eye(4), torch.zeros(4, 1)), dim=1),
        polar_offset=True,
        moveout_samples_per_radius=2.5,
    )
    radius = torch.tensor([0, 0.5, 1.0])
    decoded = torch.arange(model.decoder_time_count, dtype=torch.float32).repeat(3, 1)
    decoded.requires_grad_()
    prediction = model._apply_moveout(decoded, radius)
    expected = torch.arange(9)[None, :] + model.time_padding - 2.5 * radius[:, None]
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    prediction.sum().backward()
    torch.testing.assert_close(decoded.grad.sum(dim=1), torch.full((3,), 9.0))
    assert model(torch.tensor([[0.2, 0.3, 0.3, 0.4]])).shape == (1, 9)


@pytest.mark.parametrize("moveout", [-1, float("nan"), 2.5])
def test_moveout_requires_valid_polar_geometry(moveout):
    with pytest.raises(ValueError, match="moveout requires"):
        NersiTrace(time_sample_count=9, moveout_samples_per_radius=moveout)


def test_moveout_velocity_converts_to_samples_per_unit_polar_radius():
    # Half-offsets are 60 m and 100 m; the larger one is the unit polar radius.
    geometry = pd.DataFrame(
        {
            "source_x_m": [0.0, 10.0],
            "source_y_m": [0.0, 0.0],
            "receiver_x_m": [120.0, 10.0],
            "receiver_y_m": [0.0, 200.0],
        }
    )
    assert linear_moveout_samples_per_radius(geometry, 1000.0, 0.002) == pytest.approx(100.0)
    for velocity in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="velocity"):
            linear_moveout_samples_per_radius(geometry, velocity, 0.002)
    with pytest.raises(ValueError, match="sample interval"):
        linear_moveout_samples_per_radius(geometry, 1000.0, 0.0)
    flat = geometry.assign(receiver_x_m=geometry.source_x_m, receiver_y_m=geometry.source_y_m)
    with pytest.raises(ValueError, match="half-offset"):
        linear_moveout_samples_per_radius(flat, 1000.0, 0.002)


SMALL = dict(
    fourier_components=2,
    encoder_width=4,
    latent_channels=2,
    decoder_channels=[2, 2, 2],
    kernel_size=1,
)
IDENTITY = torch.cat((torch.eye(4), torch.zeros(4, 1)), dim=1)
DEFAULT_STATE_KEYS = [
    "coordinate_transform",
    "fourier_mapping.frequencies",
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
]
OFF = dict(
    head_samples_per_radius=None,
    static_shift_samples=0.0,
    residual_slowness_samples_per_radius=0.0,
    spatial_shape=None,
    output_band_hz=None,
    sample_interval_s=None,
    output_window=None,
)


def _heads_model(**options):
    return NersiTrace(
        time_sample_count=9,
        **SMALL,
        coordinate_transform=IDENTITY,
        polar_offset=True,
        head_samples_per_radius=[2.0, -3.0],
        **options,
    )


@pytest.mark.parametrize("moveout", [0.0, 2.5])
def test_explicit_off_options_are_bit_identical_to_the_default_model(moveout):
    config = dict(
        time_sample_count=9,
        **SMALL,
        coordinate_transform=IDENTITY,
        polar_offset=True,
        moveout_samples_per_radius=moveout,
    )
    torch.manual_seed(7301)
    default = NersiTrace(**config)
    after_default = torch.rand(3)
    torch.manual_seed(7301)
    explicit = NersiTrace(**config, **OFF)
    assert torch.equal(torch.rand(3), after_default)  # same RNG consumption
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert list(default.state_dict())[: len(DEFAULT_STATE_KEYS)] == DEFAULT_STATE_KEYS
    assert list(default.state_dict())[-2:] == [
        "output_convolution.weight",
        "output_convolution.bias",
    ]
    for name, tensor in default.state_dict().items():
        assert torch.equal(explicit.state_dict()[name], tensor), name
    coordinates = torch.rand(5, 4)
    assert torch.equal(default(coordinates), explicit(coordinates))
    assert default.output_convolution.out_channels == 1
    assert (default.pad_left, default.pad_right) == (default.time_padding, 0)


GOLDEN_COORDINATES = torch.tensor(
    [[0.15, 0.2, 0.3, 0.4], [0.7, 0.1, -0.5, 0.25], [0.4, 0.9, 0.0, 0.0]]
)
# Captured once from the NersiTrace before heads, statics, band and window existed
# (seed 7301, SMALL, T 12, GOLDEN_COORDINATES): sha256 of the state_dict and the forward output.
GOLDEN_DEFAULTS = {
    "plain": (
        {},
        "91f1f7567a80021cae1a8493b5dc52c172af50449b0b996e6e61b5df80b59817",
        [
            [0.52648401, 0.38052195, 0.52750993, 0.39737549, 0.5337128, 0.37717715],
            [0.52693063, 0.40085009, 0.53011638, 0.3795397, 0.52796304, 0.3979435],
            [0.52615619, 0.38044488, 0.52722359, 0.39774233, 0.53300619, 0.37759408],
            [0.52702075, 0.40049839, 0.52946156, 0.37973532, 0.52790558, 0.39779124],
            [0.52713919, 0.38004461, 0.52715874, 0.39819601, 0.53128654, 0.37814996],
            [0.52678668, 0.4002142, 0.52989894, 0.37960964, 0.52795029, 0.39788121],
        ],
    ),
    "polar": (
        {"coordinate_transform": IDENTITY, "polar_offset": True},
        "9be3ca4e5fc866c892e864643a19ec78421e0459b7e1dfdd82f302998879059c",
        [
            [0.52653223, 0.38052911, 0.52754569, 0.39733142, 0.53379393, 0.37712455],
            [0.52691698, 0.40089253, 0.53006232, 0.37955955, 0.52796304, 0.39792228],
            [0.52609634, 0.38044077, 0.52718586, 0.39778405, 0.53294253, 0.37763754],
            [0.52703685, 0.40045524, 0.52960688, 0.37968382, 0.52790725, 0.39784393],
            [0.52713919, 0.38004461, 0.52715874, 0.39819601, 0.53128654, 0.37814996],
            [0.52678668, 0.4002142, 0.52989894, 0.37960964, 0.52795029, 0.39788121],
        ],
    ),
    "polar_moveout": (
        {"coordinate_transform": IDENTITY, "polar_offset": True, "moveout_samples_per_radius": 3.3},
        "85eddae6c7f31b28dc7cac34cb549751477c2fdb8a24774cc5bbcf9c217db745",
        [
            [-0.28441474, -0.31792691, -0.30187362, -0.23965579, -0.27676815, -0.29834852],
            [-0.28989854, -0.24240528, -0.28679183, -0.32177991, -0.30380848, -0.23821233],
            [-0.27917355, -0.31044737, -0.32632443, -0.22905537, -0.27447277, -0.29283631],
            [-0.31154054, -0.23100835, -0.27938718, -0.31302443, -0.32846558, -0.22927737],
            [-0.34074181, -0.22129861, -0.27255085, -0.2879509, -0.3294462, -0.22211002],
            [-0.27347502, -0.30540651, -0.34704676, -0.22262394, -0.27373382, -0.28647989],
        ],
    ),
}


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("case", sorted(GOLDEN_DEFAULTS))
def test_default_model_matches_the_pre_change_golden_reference(case):
    options, state_sha256, output = GOLDEN_DEFAULTS[case]
    torch.manual_seed(7301)
    model = NersiTrace(time_sample_count=12, **SMALL, **options)
    assert _state_sha256(model) == state_sha256
    with torch.no_grad():
        prediction = model(GOLDEN_COORDINATES)
    expected = torch.tensor(output).reshape(3, 12)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=1e-6)


def _impulse(position, length=9):
    """Linear-interpolation spread of a unit impulse at a fractional sample position."""
    expected = torch.zeros(length)
    lower = int(np.floor(position))
    expected[lower] += 1 - (position - lower)
    if position != lower:
        expected[lower + 1] += position - lower
    return expected


@pytest.mark.parametrize("radius", [0.5, 0.75])
def test_one_active_head_shifts_an_impulse_by_its_moveout(radius):
    model = _heads_model()
    # m = [2, -3]: pad_left = ceil(3) + 1 = 4, pad_right = ceil(2) + 1 = 3.
    assert (model.pad_left, model.pad_right, model.decoder_time_count) == (4, 3, 16)
    assert model.output_convolution.out_channels == 2
    decoded = torch.zeros(1, 2, model.decoder_time_count)
    decoded[0, 0, model.pad_left + 5] = 1.0
    radius_tensor = torch.tensor([radius])
    polar = torch.tensor([[0.2, 0.3, radius, 0.0]])
    output = model._apply_heads(decoded, polar, radius_tensor, torch.rand(1, 4))
    # Head 0 reads t + pad_left + 2 r, so its impulse appears at t = 5 - 2 r.
    torch.testing.assert_close(output[0], _impulse(5 - 2 * radius), rtol=0, atol=1e-6)
    # Head 1 reads t + pad_left - 3 r; the head reads add.
    decoded[0, 1, model.pad_left + 5] = 1.0
    both = model._apply_heads(decoded, polar, radius_tensor, torch.rand(1, 4))
    expected = _impulse(5 - 2 * radius) + _impulse(5 + 3 * radius)
    torch.testing.assert_close(both[0], expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("bias", "correction"),
    [
        ((atanh(0.5), atanh(-0.8)), (0.25, -0.4)),  # M * tanh(b) with M = 0.5
        ((50.0, -50.0), (0.5, -0.5)),  # tanh bounds the correction at +-M
    ],
)
def test_residual_slowness_shifts_each_head_by_its_bounded_correction(bias, correction):
    model = _heads_model(residual_slowness_samples_per_radius=0.5)
    last = model.residual_slowness_network[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor(bias))
    radius = 0.5
    polar = torch.tensor([[0.2, 0.3, radius, 0.29516724]])
    for head, slope in enumerate((2.0, -3.0)):
        decoded = torch.zeros(1, 2, model.decoder_time_count)
        decoded[0, head, model.pad_left + 5] = 1.0
        output = model._apply_heads(decoded, polar, torch.tensor([radius]), torch.rand(1, 4))
        position = 5 - (slope + correction[head]) * radius
        torch.testing.assert_close(output[0], _impulse(position), rtol=0, atol=1e-6)


def test_residual_slowness_reads_the_polar_coordinates():
    model = _heads_model(residual_slowness_samples_per_radius=0.5)
    captured = {}
    hooks = [
        module.register_forward_pre_hook(
            lambda _, args, name=name: captured.setdefault(name, args[0].detach())
        )
        for name, module in (
            ("encoder", model.fourier_mapping),
            ("residual", model.residual_slowness_fourier),
        )
    ]
    model(torch.tensor([[0.2, 0.3, 0.3, 0.4], [0.1, 0.5, -0.3, -0.4]]))
    for hook in hooks:
        hook.remove()
    polar = torch.tensor([[0.2, 0.3, 0.5, 0.29516724], [0.1, 0.5, 0.5, -0.70483276]])
    torch.testing.assert_close(captured["residual"], polar)
    torch.testing.assert_close(captured["residual"], captured["encoder"], rtol=0, atol=0)


def test_zero_initialized_statics_and_residual_slowness_start_at_the_plain_heads():
    shape = (2, 3, 4, 5)
    torch.manual_seed(7301)
    model = _heads_model(
        static_shift_samples=1.5, residual_slowness_samples_per_radius=0.5, spatial_shape=shape
    )
    # margin = 2 * 1.5 + 0.5 = 3.5 on both sides.
    assert (model.pad_left, model.pad_right) == (math_ceil(3 + 3.5) + 1, math_ceil(2 + 3.5) + 1)
    assert model.src_static.shape == (1, 1, 2, 3) and model.rcv_static.shape == (1, 1, 4, 5)
    assert not model.src_static.any() and not model.rcv_static.any()
    coordinates = torch.rand(4, 4)
    with_priors = model(coordinates)
    model.static_shift_samples = 0.0
    model.residual_slowness_samples_per_radius = 0.0
    torch.testing.assert_close(model(coordinates), with_priors, rtol=0, atol=0)
    model.static_shift_samples, model.residual_slowness_samples_per_radius = 1.5, 0.5
    model(coordinates).square().sum().backward()
    assert model.src_static.grad.abs().sum() > 0 and model.rcv_static.grad.abs().sum() > 0
    assert model.residual_slowness_network[-1].weight.grad.abs().sum() > 0


def test_static_grids_are_sampled_at_the_local_input_coordinates():
    # Source lines x points 2 x 3, receiver lines x points 4 x 5.
    model = _heads_model(static_shift_samples=2.0, spatial_shape=(2, 3, 4, 5))
    assert model.src_static.shape == (1, 1, 2, 3) and model.rcv_static.shape == (1, 1, 4, 5)
    with torch.no_grad():
        model.src_static.copy_(torch.arange(6.0).reshape(1, 1, 2, 3) / 10)
        model.rcv_static.zero_()
    # (line, point) = (1, 0.5) of the source grid in [0, 1] units -> row 1, column 1.
    local = torch.tensor([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7]])
    expected = 2.0 * torch.tanh(torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(model._static_shift(local), expected)
    with torch.no_grad():
        model.src_static.zero_()
        model.rcv_static.copy_(torch.arange(20.0).reshape(1, 1, 4, 5) / 10)
    # Receiver (line, point) = (1/3, 0.5) -> row 1, column 2; (1, 1) -> row 3, column 4.
    # The source coordinates differ, so a receiver read at local[:, :2] would disagree.
    local = torch.tensor([[1.0, 0.5, 1 / 3, 0.5], [0.0, 0.0, 1.0, 1.0]])
    expected = 2.0 * torch.tanh(torch.tensor([0.7, 1.9]))
    torch.testing.assert_close(model._static_shift(local), expected)


def _reference_taper(frequency, low, high):
    weight = np.zeros_like(frequency)
    rise = (frequency >= low - 2) & (frequency < low)
    weight[rise] = 0.5 - 0.5 * np.cos(np.pi * (frequency[rise] - (low - 2)) / 2)
    weight[(frequency >= low) & (frequency <= high)] = 1.0
    fall = (frequency > high) & (frequency < high + 5)
    weight[fall] = 0.5 - 0.5 * np.cos(np.pi * ((high + 5) - frequency[fall]) / 5)
    return weight


def test_output_band_is_a_fixed_zero_phase_taper_matching_numpy():
    frequency = np.fft.rfftfreq(64, 0.004)
    np.testing.assert_allclose(
        band_taper(64, (8, 40), 0.004), _reference_taper(frequency, 8, 40), atol=1e-12
    )
    model = NersiTrace(
        time_sample_count=64, **SMALL, output_band_hz=[8, 40], sample_interval_s=0.004
    )
    assert not any("band" in name for name, _ in model.named_parameters())
    coordinates = torch.rand(3, 4)
    banded = model(coordinates).detach().numpy()
    model.output_band = None
    raw = model(coordinates).detach().numpy()
    weight = _reference_taper(frequency, 8, 40)
    expected = np.fft.irfft(np.fft.rfft(raw, axis=1) * weight, n=64, axis=1)
    np.testing.assert_allclose(banded, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("delay", "starts"),
    [
        (4.0, (6, 2, 4)),  # 6 - rint(r * 4) for r = 0, 1, 0.5
        (4.8, (6, 1, 4)),  # r * delay = 4.8, 2.4: rounds up, then down
        (5.2, (6, 1, 3)),  # r * delay = 5.2, 2.6: rounds down, then up
    ],
)
def test_output_window_zeros_samples_outside_each_trace_window(delay, starts):
    window = {"max_delay_samples": 6, "window_samples": 4, "delay_samples_per_radius": delay}
    model = NersiTrace(
        time_sample_count=12,
        **SMALL,
        coordinate_transform=IDENTITY,
        polar_offset=True,
        output_window=window,
    )
    coordinates = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.1, 0.2, 0.6, 0.8], [0.3, 0.2, 0.3, 0.4]])
    masked = model(coordinates).detach()
    model.output_window = None
    raw = model(coordinates).detach()
    for row, start in enumerate(starts):
        keep = torch.zeros(12, dtype=torch.bool)
        keep[start : start + 4] = True
        assert torch.equal(masked[row, ~keep], torch.zeros(int((~keep).sum())))
        assert torch.equal(masked[row, keep], raw[row, keep])


def test_band_then_window_follow_the_head_sum():
    window = {"max_delay_samples": 20, "window_samples": 20, "delay_samples_per_radius": 8.0}
    model = NersiTrace(
        time_sample_count=64,
        **SMALL,
        coordinate_transform=IDENTITY,
        polar_offset=True,
        head_samples_per_radius=[2.0, -3.0],
        output_band_hz=[8, 40],
        sample_interval_s=0.004,
        output_window=window,
    )
    coordinates = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.1, 0.2, 0.6, 0.8], [0.3, 0.2, 0.3, 0.4]])
    output = model(coordinates).detach().numpy()
    model.output_band, model.output_window = None, None
    raw = model(coordinates).detach().numpy()
    weight = _reference_taper(np.fft.rfftfreq(64, 0.004), 8, 40)
    banded = np.fft.irfft(np.fft.rfft(raw, axis=1) * weight, n=64, axis=1)
    keep = np.zeros((3, 64), dtype=bool)
    for row, start in enumerate((20, 12, 16)):  # 20 - rint(r * 8) for r = 0, 1, 0.5
        keep[row, start : start + 20] = True
    np.testing.assert_allclose(output, banded * keep, atol=1e-6)
    assert not output[~keep].any()
    # The reverse order (window, then band) spreads energy outside the recorded window.
    reverse = np.fft.irfft(np.fft.rfft(raw * keep, axis=1) * weight, n=64, axis=1)
    assert np.abs(reverse[~keep]).max() > 1e-3


def test_head_parameter_count_formula():
    shape = (2, 3, 4, 5)
    model = NersiTrace(
        time_sample_count=20,
        fourier_components=3,
        encoder_width=7,
        latent_channels=2,
        decoder_channels=[4, 3, 2],
        kernel_size=3,
        coordinate_transform=IDENTITY,
        polar_offset=True,
        head_samples_per_radius=[5.5, 0.0, -2.2],
        static_shift_samples=1.0,
        residual_slowness_samples_per_radius=0.25,
        spatial_shape=shape,
    )
    pad_left, pad_right = math_ceil(2.2 + 2.25) + 1, math_ceil(5.5 + 2.25) + 1
    base_time = math_ceil((20 + pad_left + pad_right) / 8)
    reference = NersiTrace(
        time_sample_count=8 * base_time,
        fourier_components=3,
        encoder_width=7,
        latent_channels=2,
        decoder_channels=[4, 3, 2],
        kernel_size=3,
    )
    decoder = sum(p.numel() for p in reference.decoder.parameters())
    expected = (
        (3 * 8 * 7 + 7)
        + (7 * 2 * base_time + 2 * base_time)
        + decoder
        + (2 * 3 + 3)
        + (2 * 3 + 4 * 5)
        + (64 * 128 + 128 + 128 * 3 + 3)
    )
    assert model.base_time == base_time
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"head_samples_per_radius": []}, "nonempty list"),
        ({"head_samples_per_radius": [1.0, float("nan")]}, "finite numbers"),
        ({"head_samples_per_radius": [1.0], "polar_offset": False}, "requires polar_offset"),
        ({"head_samples_per_radius": [1.0], "moveout_samples_per_radius": 2.0}, "exclusive"),
        ({"static_shift_samples": 1.0}, "static_shift_samples requires head_samples"),
        ({"residual_slowness_samples_per_radius": 1.0}, "requires head_samples"),
        ({"head_samples_per_radius": [1.0], "static_shift_samples": -1.0}, ">= 0"),
        ({"head_samples_per_radius": [1.0], "static_shift_samples": 1.0}, "spatial_shape of four"),
        (
            {
                "head_samples_per_radius": [1.0],
                "static_shift_samples": 1.0,
                "spatial_shape": (2, 3),
            },
            "spatial_shape of four",
        ),
        ({"spatial_shape": (2, 3, 2, 3)}, "used only with static_shift_samples"),
        ({"output_band_hz": [8, 40]}, "sample_interval_s"),
        ({"sample_interval_s": 0.004}, "used only with output_band_hz"),
        ({"output_band_hz": [2, 40], "sample_interval_s": 0.004}, "2 < lo < hi"),
        ({"output_band_hz": [8, 121], "sample_interval_s": 0.004}, "Nyquist"),
        ({"output_window": {"max_delay_samples": 1}}, "exactly max_delay_samples"),
        (
            {
                "output_window": {
                    "max_delay_samples": 1,
                    "window_samples": 0,
                    "delay_samples_per_radius": 1.0,
                }
            },
            "window_samples >= 1",
        ),
        (
            {
                "polar_offset": False,
                "output_window": {
                    "max_delay_samples": 1,
                    "window_samples": 2,
                    "delay_samples_per_radius": 1.0,
                },
            },
            "output_window requires polar_offset",
        ),
    ],
)
def test_prior_option_validation(options, message):
    config = dict(time_sample_count=9, **SMALL, coordinate_transform=IDENTITY, polar_offset=True)
    with pytest.raises(ValueError, match=message):
        NersiTrace(**{**config, **options})


def test_slowness_conversions_use_the_unit_polar_radius():
    geometry = pd.DataFrame(
        {
            "source_x_m": [0.0, 10.0],
            "source_y_m": [0.0, 0.0],
            "receiver_x_m": [120.0, 10.0],
            "receiver_y_m": [0.0, 200.0],
        }
    )
    # R_max = 100 m and dt = 2 ms: 1 ms/m is 1e-3 * 200 / 0.002 = 100 samples per unit radius.
    aligned = head_samples_per_radius(geometry, [0.5, 1.0, 2.0], 0.002, 1000.0)
    np.testing.assert_allclose(aligned, [50.0, 0.0, -100.0], rtol=1e-12)
    raw = head_samples_per_radius(geometry, [0.5, 2.0], 0.002)
    np.testing.assert_allclose(raw, [-50.0, -200.0], rtol=1e-12)
    assert slowness_samples_per_radius(geometry, 0.2, 0.002) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="finite and positive"):
        head_samples_per_radius(geometry, [0.0], 0.002)
    with pytest.raises(ValueError, match="nonnegative"):
        slowness_samples_per_radius(geometry, -0.1, 0.002)
