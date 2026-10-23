from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import seis_interp.training.ccnet5d_prediction as prediction_module
from seis_interp.data.c3_volume_adapter import ObservedC3Volume
from seis_interp.models.ccnet5d import CCNet5D
from seis_interp.training.ccnet5d_prediction import predict_ccnet5d_volume


def _volume(shape=(13, 2, 3, 1, 4), dtype=np.float64):
    rng = np.random.default_rng(21)
    mask = rng.random(shape[1:]) > 0.5
    values = rng.normal(size=shape).astype(dtype)
    return ObservedC3Volume(
        values=values,
        time_s=np.arange(shape[0]) * 0.004,
        array_rows=np.arange(np.prod(shape[1:])).reshape(shape[1:]),
        observed_trace_mask=mask,
        evaluation_target_trace_mask=~mask,
    )


def test_halo_matches_full_forward_including_true_boundaries_with_bias() -> None:
    volume = _volume()
    torch.manual_seed(7)
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=3).double()
    # Positive biases and mixed-sign weights exercise internal ReLU and make
    # externally pre-padding the volume measurably wrong at its boundaries.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.3)
    inputs = torch.from_numpy(np.where(volume.observed_trace_mask[None], volume.values, 0) / 2)
    with torch.no_grad():
        full = model(inputs[None, None])[0, 0].numpy() * 2
        padded = F.pad(inputs, (4, 4) * 5)
        externally_padded = model(padded[None, None])[0, 0][(slice(4, -4),) * 5].numpy() * 2
    assert not np.allclose(full, externally_padded, rtol=1e-5, atol=1e-5)
    seen = []
    handle = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[2:]))
    result = predict_ccnet5d_volume(
        model, volume, amplitude_rms=2, core_shape=(3, 2, 2, 1, 2), device="cpu"
    )
    handle.remove()
    errors = full[:, volume.observed_trace_mask] - volume.values[:, volume.observed_trace_mask]
    full[:, volume.observed_trace_mask] = volume.values[:, volume.observed_trace_mask]
    np.testing.assert_allclose(result.values, full, rtol=1e-12, atol=1e-12)
    assert result.observed_model_rmse_before_reinsertion == pytest.approx(
        np.sqrt(np.mean(errors**2))
    )
    assert result.observed_model_max_abs_error_before_reinsertion == pytest.approx(
        np.max(abs(errors))
    )
    assert result.tile_count == len(seen) == 20
    np.testing.assert_array_equal(
        result.coverage_counts,
        np.ones(volume.observed_trace_mask.shape, dtype=result.coverage_counts.dtype),
    )
    assert max(shape[0] for shape in seen) == 11 < volume.values.shape[0]
    assert result.maximum_input_shape == (11, 2, 3, 1, 4)
    assert model.training


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_signed_physical_output_empty_tiles_reinsertion_and_target_isolation(dtype) -> None:
    volume = _volume(shape=(3, 2, 2, 1, 3), dtype=dtype)
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=1).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        list(model.modules())[-1].bias.fill_(-1)
    outputs = []
    for target in (0, 1e20, np.nan):
        values = volume.values.copy()
        values[:, ~volume.observed_trace_mask] = target
        before = values.copy()
        result = predict_ccnet5d_volume(
            model,
            replace(volume, values=values),
            amplitude_rms=3,
            core_shape=(2, 1, 1, 1, 1),
            device="cpu",
        )
        np.testing.assert_array_equal(values, before)
        expected = np.where(volume.observed_trace_mask[None], volume.values, -3)
        np.testing.assert_array_equal(result.values, expected)
        assert result.values.dtype == dtype
        assert result.values.flags.c_contiguous
        assert result.tile_count == 24
        assert not model.training
        outputs.append(result.values)
    np.testing.assert_array_equal(outputs[0], outputs[2])
    empty = replace(volume, observed_trace_mask=np.zeros_like(volume.observed_trace_mask))
    result = predict_ccnet5d_volume(
        model, empty, amplitude_rms=3, core_shape=(2, 1, 1, 1, 1), device="cpu"
    )
    np.testing.assert_array_equal(result.values, -3)
    assert result.tile_count == 24
    assert result.observed_model_rmse_before_reinsertion == 0


def test_missing_boundary_tile_is_rejected_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = _volume(shape=(3, 2, 2, 1, 3), dtype=np.float32)
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=1)
    original_tiles = prediction_module.iter_ccnet5d_tiles

    def omit_last_tile(*args, **kwargs):
        yield from list(original_tiles(*args, **kwargs))[:-1]

    monkeypatch.setattr(prediction_module, "iter_ccnet5d_tiles", omit_last_tile)

    with pytest.raises(ValueError, match="cover every output sample"):
        predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=1.0,
            core_shape=(2, 1, 1, 1, 1),
            device="cpu",
        )


def test_model_mode_is_restored_on_forward_failure() -> None:
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=1)

    def fail(module, args):
        assert not module.training
        raise RuntimeError("forward failed")

    handle = model.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        predict_ccnet5d_volume(model, _volume(), amplitude_rms=1, core_shape=(2,) * 5, device="cpu")
    handle.remove()
    assert model.training


def test_dilated_nondivisible_halo_tiles_match_full_reference_forward() -> None:
    volume = _volume(shape=(53, 2, 1, 1, 3))
    torch.manual_seed(57)
    model = CCNet5D(
        hidden_channels=2,
        intermediate_channels=2,
        kernel_size=3,
        time_kernel_size=3,
        time_dilation=4,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(0.3)
    normalized = np.where(volume.observed_trace_mask[None], volume.values, 0) / 2.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    input_times = []
    handle = model.register_forward_pre_hook(
        lambda module, args: input_times.append(args[0].shape[2])
    )
    try:
        with torch.no_grad():
            expected = model.network(torch.from_numpy(normalized)[None, None])[0, 0].numpy() * 2.0
        result = predict_ccnet5d_volume(
            model, volume, amplitude_rms=2.0, core_shape=(7, 1, 1, 1, 2), device="cpu"
        )
    finally:
        handle.remove()
        torch.set_num_threads(previous_threads)
    assert any(time % model.time_dilation for time in input_times)
    assert max(input_times) < volume.values.shape[0]
    expected[:, volume.observed_trace_mask] = volume.values[:, volume.observed_trace_mask]
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.coverage_counts, np.ones(volume.values.shape[1:]))


@pytest.mark.parametrize("mode", ["fp16", "invalid", None])
def test_unsupported_prediction_precision_is_rejected(mode) -> None:
    model = CCNet5D(hidden_channels=1, intermediate_channels=1, kernel_size=1)
    with pytest.raises(ValueError, match="prediction.mixed_precision"):
        predict_ccnet5d_volume(
            model,
            _volume(),
            amplitude_rms=1,
            core_shape=(2,) * 5,
            device="cpu",
            mixed_precision=mode,
        )


def test_full_time_prediction_uses_spatial_halo_instead_of_dilated_time_halo() -> None:
    volume = _volume(shape=(17, 1, 19, 1, 21))
    torch.manual_seed(43)
    model = CCNet5D(
        hidden_channels=2, intermediate_channels=2, kernel_size=3, time_dilation=3
    ).double()
    source = np.where(volume.observed_trace_mask[None], volume.values, 0) / 2
    with torch.no_grad():
        expected = model(torch.from_numpy(source)[None, None])[0, 0].numpy() * 2
    expected[:, volume.observed_trace_mask] = volume.values[:, volume.observed_trace_mask]
    result = predict_ccnet5d_volume(
        model, volume, amplitude_rms=2, core_shape=(17, 1, 3, 1, 4), device="cpu"
    )
    assert model.time_halo_radius == 12
    assert result.maximum_input_shape == (17, 1, 11, 1, 12)
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="BF16 CUDA unavailable",
)
def test_bf16_prediction_uses_spatial_tiles_with_full_time_and_restores_float32() -> None:
    volume = _volume(shape=(13, 3, 5, 2, 5), dtype=np.float32)
    volume.values[:, ~volume.observed_trace_mask] = np.nan
    torch.manual_seed(31)
    model = CCNet5D(hidden_channels=2, intermediate_channels=2, kernel_size=3).cuda()
    source = np.where(volume.observed_trace_mask[None], volume.values, 0) / 2
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(torch.from_numpy(source).cuda()[None, None])[0, 0]
    expected = reference.float().cpu().numpy() * 2
    expected[:, volume.observed_trace_mask] = volume.values[:, volume.observed_trace_mask]
    forwards = []
    handle = model.register_forward_hook(
        lambda module, args, output: forwards.append((args[0].shape[2:], output.dtype))
    )
    try:
        result = predict_ccnet5d_volume(
            model,
            volume,
            amplitude_rms=2,
            core_shape=(13, 2, 3, 1, 3),
            device="cuda",
            mixed_precision="bf16",
        )
    finally:
        handle.remove()
    assert result.tile_count == len(forwards) > 1
    assert all(shape[0] == 13 and dtype == torch.bfloat16 for shape, dtype in forwards)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert result.values.dtype == np.float32 and np.isfinite(result.values).all()
    np.testing.assert_allclose(result.values, expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(
        result.values[:, volume.observed_trace_mask], volume.values[:, volume.observed_trace_mask]
    )
    np.testing.assert_array_equal(result.coverage_counts, np.ones(volume.values.shape[1:]))
    assert model.training
