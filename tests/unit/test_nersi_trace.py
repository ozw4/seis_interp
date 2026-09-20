import torch

from seis_interp.models.nersi_trace import NersiTrace


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
