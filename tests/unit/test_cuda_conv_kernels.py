from types import SimpleNamespace

import pytest

from seis_interp.training.cuda_conv_kernels import (
    classify_cuda_conv_kernel,
    summarize_cuda_conv_kernels,
)

TF32_FPROP = (
    "sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_"
    "nhwckrsc_nhwc_tilesize128x32x64"
)
TF32_DGRAD = "sm90_xmma_dgrad_implicit_gemm_indexed_f32f32_tf32f32_f32_tilesize256x64x32"
TF32_WGRAD = "sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_tilesize64x64x64"
FFMA_FPROP = (
    "sm80_xmma_fprop_implicit_gemm_indexed_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize32x32x8_ffma"
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (TF32_FPROP, "tensor_core"),
        (TF32_DGRAD, "tensor_core"),
        (TF32_WGRAD, "tensor_core"),
        ("sm80_xmma_fprop_bf16bf16_bf16bf16_f32", "tensor_core"),
        ("sm80_xmma_wgrad_f16f16_f16f16_f32", "tensor_core"),
        ("cutlass_tensorop_fprop", "tensor_core"),
        ("cutlass_s1688gemm_fprop", "tensor_core"),
        ("conv_16x8x16", "tensor_core"),
        (FFMA_FPROP, "non_tensor_core"),
        ("sm80_xmma_wgrad_f32f32_f32f32_f32", "non_tensor_core"),
        ("dgrad2d_grouped_direct", "non_tensor_core"),
        ("implicit_convolveNd_sgemm<float, 3>", "non_tensor_core"),
        ("at::native::vol2im_kernel<float,float>", "non_tensor_core"),
        ("at::native::vol2col_kernel<float,float>", "non_tensor_core"),
        ("unrecognized_conv_kernel", "unknown"),
    ],
)
def test_classify_cuda_conv_kernel(name, expected):
    assert classify_cuda_conv_kernel(name) == expected


def _event(names, *, backward=False, weight=(4, 3, 9, 3, 3)):
    shapes = [[1, 4, 8, 3, 3], [1, 3, 8, 3, 3], list(weight)] if backward else [[], list(weight)]
    parent = SimpleNamespace(
        name="aten::convolution_backward" if backward else "aten::convolution",
        input_shapes=shapes,
        cpu_parent=None,
    )
    return SimpleNamespace(
        name="aten::cudnn_convolution",
        kernels=[SimpleNamespace(name=name) for name in names],
        cpu_parent=parent,
    )


def test_summary_uses_time_weight_shape_and_requires_all_three_passes():
    events = [
        _event([TF32_FPROP, TF32_FPROP, "nhwcAddPaddingKernel", "tensorCopyKernel"]),
        _event([TF32_DGRAD, TF32_WGRAD], backward=True),
        _event([FFMA_FPROP], weight=(3, 4, 3, 3, 1)),
    ]
    names, tensor_cores = summarize_cuda_conv_kernels(events, time_kernel_shape=(9, 3, 3))
    assert names == tuple(sorted((TF32_FPROP, TF32_DGRAD, TF32_WGRAD, FFMA_FPROP)))
    assert tensor_cores is True
    assert summarize_cuda_conv_kernels(events[:1], time_kernel_shape=(9, 3, 3))[1] is False


def test_summary_excludes_single_input_channel_time_conv_from_tensor_core_check():
    first_layer = "void implicit_convolveNd_sgemm<__nv_bfloat16, 3, 1024>"
    fprop = "sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32"
    dgrad = "sm80_xmma_dgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32"
    wgrad = "sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32"
    events = [
        _event([first_layer], weight=(4, 1, 9, 3, 3)),
        _event([fprop]),
        _event([dgrad, wgrad], backward=True),
    ]
    names, tensor_cores = summarize_cuda_conv_kernels(events, time_kernel_shape=(9, 3, 3))
    assert first_layer in names
    assert tensor_cores is True

    events.append(_event(["sm80_xmma_wgrad_f32f32_f32f32_f32"], backward=True))
    assert summarize_cuda_conv_kernels(events, time_kernel_shape=(9, 3, 3))[1] is False


@pytest.mark.parametrize(
    "failure",
    [
        _event([FFMA_FPROP]),
        _event(["sm80_xmma_wgrad_f32f32_f32f32_f32"], backward=True),
        _event(["at::native::vol2im_kernel<float,float>"], backward=True),
        _event(["unrecognized_conv_kernel"], backward=True),
    ],
)
def test_summary_rejects_any_non_tensor_core_or_unattributed_time_compute(failure):
    events = [_event([TF32_FPROP]), _event([TF32_DGRAD, TF32_WGRAD], backward=True), failure]
    names, tensor_cores = summarize_cuda_conv_kernels(iter(events), time_kernel_shape=(9, 3, 3))
    assert failure.kernels[0].name in names
    assert tensor_cores is False


def test_summary_ignores_kernels_outside_convolutions():
    event = SimpleNamespace(name="Optimizer.step", kernels=[SimpleNamespace(name=FFMA_FPROP)])
    assert summarize_cuda_conv_kernels([event], time_kernel_shape=(9, 3, 3)) == ((), False)
