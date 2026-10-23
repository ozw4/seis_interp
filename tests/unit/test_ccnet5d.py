from __future__ import annotations

import math
from itertools import product

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import seis_interp.models.ccnet5d as ccnet5d_module
from seis_interp.models.ccnet5d import CC3D2D, CCNet5D, ccnet5d_method_variant


def _loop_reference(
    values: torch.Tensor,
    weight_3d: torch.Tensor,
    bias_3d: torch.Tensor,
    weight_2d: torch.Tensor,
    bias_2d: torch.Tensor,
    *,
    output_activation: str,
) -> torch.Tensor:
    batch, _, time, source_x, source_y, receiver_x, receiver_y = values.shape
    output = values.new_empty(
        (batch, weight_2d.shape[0], time, source_x, source_y, receiver_x, receiver_y)
    )
    for b in range(batch):
        features = values.new_empty(
            (weight_3d.shape[0], time, source_x, source_y, receiver_x, receiver_y)
        )
        for rx, ry in product(range(receiver_x), range(receiver_y)):
            block = F.conv3d(
                values[b, :, :, :, :, rx, ry].unsqueeze(0),
                weight_3d,
                bias_3d,
                padding=weight_3d.shape[-1] // 2,
            )
            features[:, :, :, :, rx, ry] = torch.relu(block[0])
        for t, sx, sy in product(range(time), range(source_x), range(source_y)):
            block = F.conv2d(
                features[:, t, sx, sy, :, :].unsqueeze(0),
                weight_2d,
                bias_2d,
                padding=weight_2d.shape[-1] // 2,
            )[0]
            output[b, :, t, sx, sy, :, :] = (
                torch.relu(block) if output_activation == "relu" else block
            )
    return output


@pytest.mark.parametrize("output_activation", ["linear", "relu"])
def test_cc3d2d_matches_independent_loop_forward_and_gradients(output_activation: str) -> None:
    generator = torch.Generator().manual_seed(214)
    module = CC3D2D(2, 3, 2, kernel_size=3, output_activation=output_activation).double()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float64, generator=generator))
    values = torch.randn(2, 2, 2, 3, 4, 5, 6, dtype=torch.float64, generator=generator)
    values.requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    actual = module(values)
    expected = _loop_reference(
        reference_values,
        module.conv3d.weight,
        module.conv3d.bias,
        module.conv2d.weight,
        module.conv2d.bias,
        output_activation=output_activation,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    probe = torch.randn(actual.shape, dtype=torch.float64, generator=generator)
    actual_gradients = torch.autograd.grad(actual, (values, *module.parameters()), probe)
    reference_gradients = torch.autograd.grad(
        expected, (reference_values, *module.parameters()), probe
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-10)


def test_cc3d2d_preserves_shape_batch_independence_and_convolution_contract() -> None:
    module = CC3D2D(2, 3, 2, kernel_size=3)
    values = torch.randn(2, 2, 2, 3, 4, 5, 6)
    output = module(values)
    changed = values.clone()
    changed[1] = -7.0

    assert output.shape == values.shape
    assert output.dtype == values.dtype
    assert output.device == values.device
    assert output.is_contiguous()
    torch.testing.assert_close(module(changed)[0], output[0], rtol=0.0, atol=0.0)
    for convolution, dimensions in ((module.conv3d, 3), (module.conv2d, 2)):
        assert convolution.bias is not None
        assert convolution.groups == 1
        assert convolution.kernel_size == (3,) * dimensions
        assert convolution.stride == convolution.dilation == (1,) * dimensions
        assert convolution.padding == (1,) * dimensions
        assert convolution.padding_mode == "zeros"


def test_cc3d2d_accepts_noncontiguous_input_without_reordering_values() -> None:
    module = CC3D2D(2, 3, 2, kernel_size=3, output_activation="linear")
    values = torch.randn(2, 2, 2, 3, 4, 5, 12)[..., ::2]

    assert not values.is_contiguous()
    torch.testing.assert_close(module(values), module(values.contiguous()))


def test_cc3d2d_linear_regime_matches_explicit_five_dimensional_cross_correlation() -> None:
    generator = torch.Generator().manual_seed(319)
    module = CC3D2D(2, 2, 2, kernel_size=3).double()
    with torch.no_grad():
        module.conv3d.weight.copy_(
            torch.rand(module.conv3d.weight.shape, dtype=torch.float64, generator=generator)
        )
        module.conv2d.weight.copy_(
            torch.rand(module.conv2d.weight.shape, dtype=torch.float64, generator=generator)
        )
        module.conv3d.bias.zero_()
        module.conv2d.bias.zero_()
    values = torch.rand(1, 2, 2, 1, 3, 2, 1, dtype=torch.float64, generator=generator)
    kernel = torch.einsum("oruv,rcijk->ocijkuv", module.conv2d.weight, module.conv3d.weight)
    padded = F.pad(values, (1, 1) * 5)
    expected = torch.empty_like(values)
    for b, out_channel in product(range(values.shape[0]), range(2)):
        for position in product(*(range(length) for length in values.shape[2:])):
            region = tuple(slice(start, start + 3) for start in position)
            expected[(b, out_channel, *position)] = (
                padded[(b, slice(None), *region)] * kernel[out_channel]
            ).sum()

    torch.testing.assert_close(module(values), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"in_channels": 0}, "in_channels"),
        ({"intermediate_channels": True}, "intermediate_channels"),
        ({"out_channels": -1}, "out_channels"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"kernel_size": 2}, "kernel_size"),
        ({"kernel_size": 1.5}, "kernel_size"),
        ({"output_activation": "tanh"}, "output_activation"),
    ],
)
def test_cc3d2d_rejects_invalid_constructor_values(changed: dict[str, object], match: str) -> None:
    arguments = {"in_channels": 2, "intermediate_channels": 3, "out_channels": 2} | changed

    with pytest.raises(ValueError, match=match):
        CC3D2D(**arguments)


@pytest.mark.parametrize("shape", [(2, 2, 3, 4, 5, 6), (2, 1, 2, 3, 4, 5, 6)])
def test_cc3d2d_rejects_wrong_input_rank_or_channels(shape: tuple[int, ...]) -> None:
    module = CC3D2D(2, 3, 2)

    with pytest.raises(ValueError, match="shape|channels"):
        module(torch.zeros(shape))


def test_ccnet5d_has_exactly_four_modules_and_the_full_width_parameter_count() -> None:
    model = CCNet5D()
    channels = [(1, 64, 64), (64, 64, 64), (64, 64, 64), (64, 64, 1)]

    assert list(dict(model.named_children())) == ["network"]
    assert len(model.network) == 4
    assert [
        (module.conv3d.in_channels, module.conv3d.out_channels, module.conv2d.out_channels)
        for module in model.network
    ] == channels
    assert [module.output_activation for module in model.network] == [
        "relu",
        "relu",
        "relu",
        "linear",
    ]
    assert all(
        type(module) in {CCNet5D, nn.Sequential, CC3D2D, nn.Conv3d, nn.Conv2d}
        for module in model.modules()
    )
    expected_count = sum(
        intermediate * incoming * 5**3 + intermediate + outgoing * intermediate * 5**2 + outgoing
        for incoming, intermediate, outgoing in channels
    )
    assert expected_count == 1_853_249
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count


def test_ccnet5d_tiny_forward_preserves_shape_and_gradients() -> None:
    model = CCNet5D(hidden_channels=3, intermediate_channels=2, kernel_size=3)
    values = torch.randn(2, 1, 2, 3, 4, 2, 3, requires_grad=True)

    output = model(values)
    output.square().mean().backward()

    assert output.shape == values.shape
    assert output.dtype == values.dtype == torch.float32
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_ccnet5d_only_final_output_activation_controls_signed_amplitudes() -> None:
    linear = CCNet5D(hidden_channels=3, intermediate_channels=2, kernel_size=1)
    relu = CCNet5D(
        hidden_channels=3, intermediate_channels=2, kernel_size=1, output_activation="relu"
    )
    with torch.no_grad():
        linear.network[-1].conv2d.weight.zero_()
        linear.network[-1].conv2d.bias.fill_(-1.0)
    relu.load_state_dict(linear.state_dict(), strict=True)
    values = torch.randn(2, 1, 2, 3, 1, 2, 4)

    torch.testing.assert_close(linear(values), -torch.ones_like(values), rtol=0.0, atol=0.0)
    torch.testing.assert_close(relu(values), torch.zeros_like(values), rtol=0.0, atol=0.0)
    assert [module.output_activation for module in relu.network] == ["relu"] * 4


@pytest.mark.parametrize("output_activation", ["linear", "relu"])
def test_ccnet5d_constructor_and_state_round_trip(output_activation: str) -> None:
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=2,
        kernel_size=3,
        output_activation=output_activation,
    )
    config = model.constructor_config()
    assert config == {
        "hidden_channels": 3,
        "intermediate_channels": 2,
        "kernel_size": 3,
        "output_activation": output_activation,
    }
    restored = CCNet5D(**config)
    restored.load_state_dict(model.state_dict(), strict=True)
    values = torch.randn(1, 1, 2, 3, 1, 2, 4)

    torch.testing.assert_close(restored(values), model(values), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("kernel_size", "radius"), [(5, 8), (3, 4)])
def test_ccnet5d_halo_radius(kernel_size: int, radius: int) -> None:
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=kernel_size)

    assert model.halo_radius == radius


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("linear", "supervised_train_partition_linear_output"),
        ("relu", "supervised_train_partition_paper_relu_output"),
    ],
)
def test_ccnet5d_method_variant_matches_final_activation(activation: str, expected: str) -> None:
    assert ccnet5d_method_variant(activation) == expected


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"hidden_channels": 0}, "hidden_channels"),
        ({"intermediate_channels": False}, "intermediate_channels"),
        ({"output_activation": "tanh"}, "output_activation"),
    ],
)
def test_ccnet5d_rejects_invalid_constructor_values(changed: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CCNet5D(**changed)


def test_ccnet5d_time_axis_options_default_to_the_isotropic_model() -> None:
    isotropic = CCNet5D(hidden_channels=3, intermediate_channels=2, kernel_size=3)
    explicit = CCNet5D(
        hidden_channels=3,
        intermediate_channels=2,
        kernel_size=3,
        time_kernel_size=3,
        time_dilation=1,
    )
    explicit.load_state_dict(isotropic.state_dict(), strict=True)
    values = torch.randn(1, 1, 6, 2, 3, 2, 3)

    torch.testing.assert_close(explicit(values), isotropic(values), rtol=0.0, atol=0.0)
    assert explicit.constructor_config() == isotropic.constructor_config()
    assert "time_kernel_size" not in isotropic.constructor_config()
    assert explicit.halo_radius == explicit.time_halo_radius == 4


def test_ccnet5d_time_kernel_and_dilation_widen_only_the_time_axis() -> None:
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=2,
        kernel_size=3,
        time_kernel_size=9,
        time_dilation=2,
    )
    values = torch.randn(1, 1, 40, 2, 3, 2, 3, requires_grad=True)

    output = model(values)
    output.square().mean().backward()

    assert output.shape == values.shape
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(module.conv3d.kernel_size == (9, 3, 3) for module in model.network)
    assert all(module.conv3d.dilation == (2, 1, 1) for module in model.network)
    assert all(module.conv3d.padding == (8, 1, 1) for module in model.network)
    assert all(module.conv2d.kernel_size == (3, 3) for module in model.network)
    assert model.time_halo_radius == 4 * 2 * 4
    assert model.halo_radius == 32
    config = model.constructor_config()
    assert config["time_kernel_size"] == 9 and config["time_dilation"] == 2
    restored = CCNet5D(**config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(values), output, rtol=0.0, atol=0.0)


def test_ccnet5d_time_receptive_field_is_bounded_by_the_time_halo() -> None:
    model = CCNet5D(
        hidden_channels=2,
        intermediate_channels=2,
        kernel_size=1,
        time_kernel_size=5,
        time_dilation=3,
    )
    with torch.no_grad():
        # Positive weights and zero biases keep every ReLU open, so the difference
        # marks exactly the samples the time-axis receptive field can reach.
        for parameter in model.parameters():
            parameter.copy_(parameter.abs() if parameter.ndim > 1 else torch.zeros_like(parameter))
    time = 4 * model.time_halo_radius + 1
    base = torch.zeros(1, 1, time, 1, 1, 1, 1)
    perturbed = base.clone()
    perturbed[0, 0, 0] = 1.0

    with torch.no_grad():
        difference = (model(perturbed) - model(base))[0, 0, :, 0, 0, 0, 0]

    assert model.time_halo_radius == 24
    assert torch.any(difference[: model.time_halo_radius + 1] != 0)
    assert torch.all(difference[model.time_halo_radius + 1 :] == 0)


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"time_kernel_size": 4}, "time_kernel_size"),
        ({"time_kernel_size": 0}, "time_kernel_size"),
        ({"time_dilation": 0}, "time_dilation"),
        ({"time_dilation": True}, "time_dilation"),
    ],
)
def test_ccnet5d_rejects_invalid_time_axis_values(changed: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=3, **changed)


@pytest.fixture(scope="module")
def single_thread_convolutions():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _assert_polyphase_matches_reference(model: CCNet5D, values: torch.Tensor) -> None:
    actual_values = values.detach().requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    actual = model(actual_values)
    expected = model.network(reference_values)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn(
        actual.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(41)
    )
    actual_gradients = torch.autograd.grad(actual, (actual_values, *model.parameters()), probe)
    expected_gradients = torch.autograd.grad(
        expected, (reference_values, *model.parameters()), probe
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


POLYPHASE_CASES = [
    (time, dilation, index)
    for index, (time, dilation) in enumerate(
        [(time, dilation) for dilation in (1, 2, 3, 4, 6) for time in (1, 2, 3, 5, 8, 9, 17, 43)]
        + [(4, 6), (6, 6)]
    )
]


@pytest.mark.usefixtures("single_thread_convolutions")
@pytest.mark.parametrize(("time", "dilation", "case_index"), POLYPHASE_CASES)
def test_polyphase_forward_and_all_gradients_match_reference(
    time: int, dilation: int, case_index: int
) -> None:
    generator = torch.Generator().manual_seed(104 + case_index)
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=4,
        kernel_size=(3, 5)[case_index % 2],
        time_kernel_size=(3, 9)[case_index // 2 % 2],
        time_dilation=dilation,
        output_activation=("linear", "relu")[case_index // 4 % 2],
    ).double()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.copy_(
                    0.3
                    + 0.4 * torch.randn(parameter.shape, dtype=torch.float64, generator=generator)
                )
    spatial = ((1, 2, 3, 1), (2, 1, 1, 3), (1, 1, 2, 3))[case_index % 3]
    batch = (1, 3)[case_index % 2]
    values = torch.randn(batch, 1, time, *spatial, dtype=torch.float64, generator=generator)
    if case_index % 3 == 0:
        values = values.transpose(-1, -2)
    _assert_polyphase_matches_reference(model, values)


@pytest.mark.usefixtures("single_thread_convolutions")
@pytest.mark.parametrize("dilation", [1, 4])
@pytest.mark.parametrize("receiver_layout", ["flat", "source_batch"])
def test_polyphase_batch_and_sample_element_splits_are_exact(
    monkeypatch, dilation: int, receiver_layout: str
) -> None:
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=4,
        kernel_size=3,
        time_kernel_size=9,
        time_dilation=dilation,
    ).double()
    generator = torch.Generator().manual_seed(72)
    values = torch.randn(3, 1, 9, 2, 1, 2, 3, dtype=torch.float64, generator=generator)
    monkeypatch.setattr(ccnet5d_module, "_TIME_CONV_BATCH_CHUNK", 5)
    folded_time = 9 if dilation == 1 else 28
    element_limit = 121 if receiver_layout == "flat" else 24 * folded_time * 3 // 2
    monkeypatch.setattr(ccnet5d_module, "_CONV_SAMPLE_ELEMENT_LIMIT", element_limit)
    convolution_shapes = []
    original = F.conv3d

    def record_convolution(inputs, weight, *args, **kwargs):
        if "padding" in kwargs:
            convolution_shapes.append((inputs.shape, weight.shape))
        return original(inputs, weight, *args, **kwargs)

    monkeypatch.setattr(F, "conv3d", record_convolution)
    _assert_polyphase_matches_reference(model, values)
    # The optimized convolutions pass padding by keyword, while the reference
    # uses positional arguments. The expanded receiver kernel ends in one.
    time_shapes = [shape for shape, weight in convolution_shapes if weight[-3] == 9]
    assert any(shape[0] == 5 for shape in time_shapes)
    assert all(shape[0] <= 5 for shape in time_shapes)
    receiver_shapes = [shape for shape, weight in convolution_shapes if weight[-1] == 1]
    assert len(receiver_shapes) > len(model.network)
    assert all(math.prod(shape[1:]) < element_limit for shape in receiver_shapes)
    if receiver_layout == "source_batch":
        assert any(shape[0] == 5 for shape in receiver_shapes)
        assert all(shape[-1] == folded_time for shape in receiver_shapes)


@pytest.mark.usefixtures("single_thread_convolutions")
def test_polyphase_gaps_and_short_phase_padding_are_zero_after_each_hidden_module(monkeypatch):
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=4,
        kernel_size=3,
        time_kernel_size=9,
        time_dilation=4,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(0.3)
    original = ccnet5d_module._zero_time_phase_padding
    checked_modules = []

    def verify_padding(values, **kwargs):
        original(values, **kwargs)
        phases = values.unflatten(3, (4, values.shape[3] // 4))
        assert torch.count_nonzero(phases[:, :, :, :, 3:]) == 0
        assert torch.count_nonzero(phases[:, :, :, 1:, 2]) == 0
        checked_modules.append(values.shape[-1])

    monkeypatch.setattr(ccnet5d_module, "_zero_time_phase_padding", verify_padding)
    _assert_polyphase_matches_reference(model, torch.randn(1, 1, 9, 1, 2, 2, 1).double())
    assert checked_modules == [3, 3, 3]


def test_polyphase_preserves_state_keys_shapes_order_and_constructor_config() -> None:
    model = CCNet5D(
        hidden_channels=3,
        intermediate_channels=4,
        kernel_size=3,
        time_kernel_size=9,
        time_dilation=4,
        output_activation="relu",
    )
    expected = []
    for index, (incoming, outgoing) in enumerate([(1, 3), (3, 3), (3, 3), (3, 1)]):
        expected.extend(
            [
                (f"network.{index}.conv3d.weight", (4, incoming, 9, 3, 3)),
                (f"network.{index}.conv3d.bias", (4,)),
                (f"network.{index}.conv2d.weight", (outgoing, 4, 3, 3)),
                (f"network.{index}.conv2d.bias", (outgoing,)),
            ]
        )
    assert [(name, tuple(value.shape)) for name, value in model.state_dict().items()] == expected
    assert not list(model.buffers())
    assert model.constructor_config() == {
        "hidden_channels": 3,
        "intermediate_channels": 4,
        "kernel_size": 3,
        "output_activation": "relu",
        "time_kernel_size": 9,
        "time_dilation": 4,
    }


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 1, 2, 1, 1, 1), "values must have shape (B, C, T, Sx, Sy, Rx, Ry)"),
        ((1, 2, 2, 1, 1, 1, 1), "values must have 1 input channels"),
    ],
)
def test_polyphase_preserves_input_validation_messages(shape, message):
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=3)
    with pytest.raises(ValueError) as error:
        model(torch.zeros(shape))
    assert str(error.value) == message


@pytest.mark.parametrize("dilation", [1, 4])
def test_polyphase_preserves_empty_time_exception(dilation):
    model = CCNet5D(
        hidden_channels=1,
        intermediate_channels=1,
        kernel_size=3,
        time_kernel_size=9,
        time_dilation=dilation,
    )
    values = torch.zeros(1, 1, 0, 1, 1, 1, 1)
    with pytest.raises(RuntimeError) as reference_error:
        model.network(values)
    with pytest.raises(RuntimeError) as actual_error:
        model(values)
    assert str(actual_error.value) == str(reference_error.value)


@pytest.mark.parametrize("axis", [0, 3, 4, 5, 6])
def test_polyphase_preserves_other_empty_dimension_results(axis):
    model = CCNet5D(
        hidden_channels=1,
        intermediate_channels=1,
        kernel_size=3,
        time_dilation=4,
    )
    shape = [1, 1, 3, 2, 2, 2, 2]
    shape[axis] = 0
    values = torch.zeros(shape)
    if axis == 0:
        torch.testing.assert_close(model(values), model.network(values), rtol=0, atol=0)
    else:
        with pytest.raises(RuntimeError) as reference_error:
            model.network(values)
        with pytest.raises(RuntimeError) as actual_error:
            model(values)
        assert str(actual_error.value) == str(reference_error.value)
