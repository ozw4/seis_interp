import numpy as np
import pandas as pd
import pytest
import torch

from seis_interp.models.nersi_trace import NersiTrace
from seis_interp.processing.nersi_trace_coordinates import midpoint_offset_transform


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
