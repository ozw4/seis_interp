"""Classify convolution CUDA kernels and identify time-convolution passes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

_TENSOR_CORE_MARKERS = ("tf32", "bf16", "f16", "tensorop", "s1688", "16x8x")
_NON_TENSOR_CORE_MARKERS = ("ffma", "f32f32_f32f32", "direct", "convolvend")
_NATIVE_MARKERS = ("vol2im", "im2vol", "vol2col", "col2vol", "im2col", "col2im")
_CONVOLUTION_OPS = ("aten::convolution", "aten::_convolution", "aten::convolution_backward")


def classify_cuda_conv_kernel(name: str) -> Literal["tensor_core", "non_tensor_core", "unknown"]:
    """Classify arithmetic from kernel names, including PyTorch's native fallback."""
    lowered = name.lower()
    if any(marker in lowered for marker in _NATIVE_MARKERS):
        return "non_tensor_core"
    if any(marker in lowered for marker in _TENSOR_CORE_MARKERS):
        return "tensor_core"
    if any(marker in lowered for marker in _NON_TENSOR_CORE_MARKERS):
        return "non_tensor_core"
    return "unknown"


def summarize_cuda_conv_kernels(
    events: Iterable[object],
    *,
    time_kernel_shape: tuple[int, int, int],
) -> tuple[tuple[str, ...], bool]:
    """Return unique compute names and whether all hidden time-conv passes use TC.

    Profiler CPU ancestry and recorded weight shapes distinguish the time
    convolution from the receiver convolution, which also runs as Conv3d.
    Single-input-channel convolutions are excluded from the Tensor Core check.
    Missing passes and unrecognized arithmetic never count as Tensor Core use.
    """
    names: set[str] = set()
    time_passes: dict[str, list[str]] = {name: [] for name in ("fprop", "dgrad", "wgrad")}
    unidentified_time_pass = False
    for event in events:
        weight_shape, backward = _convolution_context(event)
        if weight_shape is None:
            continue
        for kernel in getattr(event, "kernels", ()) or ():
            name = kernel.name
            if not _is_compute_kernel(name):
                continue
            names.add(name)
            if weight_shape[1] != 1 and tuple(weight_shape[2:]) == time_kernel_shape:
                pass_name = _convolution_pass(name, backward)
                if pass_name is None:
                    unidentified_time_pass = True
                else:
                    time_passes[pass_name].append(classify_cuda_conv_kernel(name))
    all_tensor_core = not unidentified_time_pass and all(
        classifications and all(value == "tensor_core" for value in classifications)
        for classifications in time_passes.values()
    )
    return tuple(sorted(names)), bool(all_tensor_core)


def _convolution_context(event: object) -> tuple[list[int] | None, bool]:
    while event is not None:
        if event.name in _CONVOLUTION_OPS:
            backward = event.name == "aten::convolution_backward"
            index = 2 if backward else 1
            shapes = getattr(event, "input_shapes", ()) or ()
            if len(shapes) > index and shapes[index]:
                return shapes[index], backward
        event = getattr(event, "cpu_parent", None)
    return None, False


def _is_compute_kernel(name: str) -> bool:
    lowered = name.lower()
    if any(
        marker in lowered
        for marker in (
            "copy",
            "elementwise",
            "transpose",
            "nchwtonhwc",
            "nhwctonchw",
            "convert",
            "reduce",
            "padding",
        )
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "gemm",
            "conv",
            "fprop",
            "dgrad",
            "wgrad",
            "xmma",
            "cutlass",
            "engine",
            "direct",
            *_NATIVE_MARKERS,
        )
    )


def _convolution_pass(name: str, backward: bool) -> str | None:
    lowered = name.lower()
    for pass_name in ("fprop", "dgrad", "wgrad"):
        if pass_name in lowered:
            return pass_name
    if not backward:
        return "fprop"
    if any(marker in lowered for marker in ("vol2im", "col2vol", "col2im")):
        return "dgrad"
    if any(marker in lowered for marker in ("im2vol", "vol2col", "im2col")):
        return "wgrad"
    return None
