import pytest
import torch

from seis_interp.models.spectral_trace_graph import SpectralTraceGraphInputBlock


@pytest.mark.parametrize("time_count", [1, 7, 8, 384])
def test_initial_spectral_block_is_observed_average_and_ignores_hidden_values(time_count):
    torch.manual_seed(11)
    block = SpectralTraceGraphInputBlock()
    waveforms = torch.randn(4, time_count)
    observed = torch.tensor([True, True, False, False])
    edges = torch.tensor([[0, 1], [2, 2]])
    types = torch.tensor([0, 1])
    features = torch.zeros(2, 15)
    output = block(waveforms, observed, edges, types, features)
    torch.testing.assert_close(output[:2], waveforms[:2], rtol=0, atol=0)
    torch.testing.assert_close(output[2], waveforms[:2].mean(0), rtol=1e-5, atol=1e-6)
    assert torch.equal(output[3], torch.zeros(time_count))
    changed = waveforms.clone()
    changed[~observed] = float("nan")
    torch.testing.assert_close(block(changed, observed, edges, types, features), output)
    output.square().mean().backward()
    gradient = block.edge_response[-1].weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_spectral_block_is_linear_in_global_normalized_amplitude():
    block = SpectralTraceGraphInputBlock()
    torch.nn.init.normal_(block.edge_response[-1].weight, std=0.01)
    waveforms = torch.randn(3, 17)
    observed = torch.tensor([True, True, False])
    edges = torch.tensor([[0, 1], [2, 2]])
    types = torch.tensor([0, 1])
    features = torch.zeros(2, 15)
    output = block(waveforms, observed, edges, types, features)
    scaled = block(waveforms * 3, observed, edges, types, features)
    torch.testing.assert_close(scaled, output * 3, rtol=1e-5, atol=1e-6)
