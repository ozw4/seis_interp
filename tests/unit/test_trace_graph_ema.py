import pytest
import torch

from seis_interp.training.trace_graph_ema import TraceGraphEMA, validate_trace_graph_ema_decay


@pytest.mark.parametrize("value", [True, False, 0, 1, -0.1, float("nan"), float("inf"), "0.9", {}])
def test_invalid_decay(value):
    with pytest.raises(ValueError, match="ema_decay"):
        validate_trace_graph_ema_decay(value)


def test_post_update_recurrence_buffers_snapshot_and_rng():
    model = torch.nn.Linear(1, 1, bias=False)
    model.register_buffer("frequencies", torch.tensor([1.25]))
    model.register_buffer("counter", torch.tensor(0))
    rng = torch.get_rng_state().clone()
    ema = TraceGraphEMA(0.75)
    with pytest.raises(ValueError, match="no successful"):
        ema.state_dict()
    with torch.no_grad():
        for index, weight in enumerate([2.0, 6.0, 10.0]):
            model.weight.fill_(weight)
            model.counter.fill_(index)
            ema.update(model)
    assert ema.updates == 3
    state = ema.state_dict()
    # First post-update value 2 -> 3 -> 4.75; the original initialization is excluded.
    torch.testing.assert_close(state["weight"], torch.tensor([[4.75]]), rtol=0, atol=0)
    assert model.weight.item() == 10
    assert state["counter"].item() == 2
    assert torch.equal(state["frequencies"], model.frequencies)
    assert not state["weight"].requires_grad
    state["weight"].zero_()
    assert ema.state_dict()["weight"].item() == 4.75
    assert torch.equal(rng, torch.get_rng_state())
    assert validate_trace_graph_ema_decay(None) is None
