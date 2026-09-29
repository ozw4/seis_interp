"""Cross-convolution on time, source, and relative-receiver axes."""

from __future__ import annotations

import torch
from torch import nn


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_output_activation(value: str) -> None:
    if value not in ("relu", "linear"):
        raise ValueError("output_activation must be 'relu' or 'linear'")


def _validate_odd_kernel(name: str, value: int) -> None:
    _validate_positive_integer(name, value)
    if value % 2 != 1:
        raise ValueError(f"{name} must be odd")


def _time_kernel_size(kernel_size: int, time_kernel_size: int | None) -> int:
    if time_kernel_size is None:
        return kernel_size
    _validate_odd_kernel("time_kernel_size", time_kernel_size)
    return time_kernel_size


class CC3D2D(nn.Module):
    """Apply Conv3D/ReLU, then Conv2D over the remaining two spatial axes.

    Input and output use ``(B, C, T, Sx, Sy, Rx, Ry)``. Both convolutions
    preserve their spatial shapes with same zero padding. The intermediate
    channels are reduced by Conv2D, without an additional branch sum.

    ``time_kernel_size`` and ``time_dilation`` widen only the Conv3D time axis;
    the default keeps the isotropic, undilated kernel.
    """

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        output_activation: str = "relu",
        time_kernel_size: int | None = None,
        time_dilation: int = 1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("intermediate_channels", intermediate_channels),
            ("out_channels", out_channels),
        ):
            _validate_positive_integer(name, value)
        _validate_odd_kernel("kernel_size", kernel_size)
        time_kernel = _time_kernel_size(kernel_size, time_kernel_size)
        _validate_positive_integer("time_dilation", time_dilation)
        _validate_output_activation(output_activation)
        self.output_activation = output_activation
        self.conv3d = nn.Conv3d(
            in_channels,
            intermediate_channels,
            (time_kernel, kernel_size, kernel_size),
            padding=(time_dilation * (time_kernel // 2), kernel_size // 2, kernel_size // 2),
            dilation=(time_dilation, 1, 1),
            bias=True,
        )
        self.conv2d = nn.Conv2d(
            intermediate_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 7:
            raise ValueError("values must have shape (B, C, T, Sx, Sy, Rx, Ry)")
        if values.shape[1] != self.conv3d.in_channels:
            raise ValueError(f"values must have {self.conv3d.in_channels} input channels")
        batch, channels, time, source_x, source_y, receiver_x, receiver_y = values.shape
        intermediate = self.conv3d.out_channels
        values_3d = values.permute(0, 5, 6, 1, 2, 3, 4).reshape(
            batch * receiver_x * receiver_y, channels, time, source_x, source_y
        )
        features = torch.relu(self.conv3d(values_3d))
        features = features.reshape(
            batch, receiver_x, receiver_y, intermediate, time, source_x, source_y
        )
        values_2d = features.permute(0, 4, 5, 6, 3, 1, 2).reshape(
            batch * time * source_x * source_y, intermediate, receiver_x, receiver_y
        )
        output = self.conv2d(values_2d)
        if self.output_activation == "relu":
            output = torch.relu(output)
        return (
            output.reshape(
                batch, time, source_x, source_y, self.conv2d.out_channels, receiver_x, receiver_y
            )
            .permute(0, 4, 1, 2, 3, 5, 6)
            .contiguous()
        )


def ccnet5d_method_variant(output_activation: str) -> str:
    """Identify the supervised signed-output or paper-ReLU audit variant."""
    _validate_output_activation(output_activation)
    if output_activation == "linear":
        return "supervised_train_partition_linear_output"
    return "supervised_train_partition_paper_relu_output"


class CCNet5D(nn.Module):
    """Four cross-convolution modules with one amplitude input/output channel.

    The default final linear activation is an explicit signed-amplitude
    modification, not a claim of complete paper reproduction. ``relu`` keeps
    the final Conv2D activation for an audit variant. All earlier convolution
    activations remain ReLU; there are no skip or normalization layers.

    ``time_kernel_size`` and ``time_dilation`` enlarge only the time-axis
    receptive field of every Conv3D; the defaults reproduce the isotropic model.
    """

    def __init__(
        self,
        *,
        hidden_channels: int = 64,
        intermediate_channels: int = 64,
        kernel_size: int = 5,
        output_activation: str = "linear",
        time_kernel_size: int | None = None,
        time_dilation: int = 1,
    ) -> None:
        super().__init__()
        _validate_positive_integer("hidden_channels", hidden_channels)
        _validate_positive_integer("intermediate_channels", intermediate_channels)
        _validate_output_activation(output_activation)
        self.hidden_channels = hidden_channels
        self.intermediate_channels = intermediate_channels
        self.kernel_size = kernel_size
        self.output_activation = output_activation
        self.time_kernel_size = _time_kernel_size(kernel_size, time_kernel_size)
        _validate_positive_integer("time_dilation", time_dilation)
        self.time_dilation = time_dilation
        time_axis = {"time_kernel_size": time_kernel_size, "time_dilation": time_dilation}
        self.network = nn.Sequential(
            CC3D2D(1, intermediate_channels, hidden_channels, kernel_size=kernel_size, **time_axis),
            CC3D2D(
                hidden_channels,
                intermediate_channels,
                hidden_channels,
                kernel_size=kernel_size,
                **time_axis,
            ),
            CC3D2D(
                hidden_channels,
                intermediate_channels,
                hidden_channels,
                kernel_size=kernel_size,
                **time_axis,
            ),
            CC3D2D(
                hidden_channels,
                intermediate_channels,
                1,
                kernel_size=kernel_size,
                output_activation=output_activation,
                **time_axis,
            ),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    @property
    def time_halo_radius(self) -> int:
        """Return the time-axis receptive-field radius for four stride-one modules."""
        return 4 * self.time_dilation * (self.time_kernel_size // 2)

    @property
    def halo_radius(self) -> int:
        """Return the largest per-axis receptive-field radius; spatial axes need at most this."""
        return max(4 * (self.kernel_size // 2), self.time_halo_radius)

    def constructor_config(self) -> dict[str, object]:
        """Return every constructor value needed to restore the same function.

        Time-axis keys appear only when they differ from the isotropic default,
        so existing checkpoint payloads are unchanged.
        """
        config: dict[str, object] = {
            "hidden_channels": self.hidden_channels,
            "intermediate_channels": self.intermediate_channels,
            "kernel_size": self.kernel_size,
            "output_activation": self.output_activation,
        }
        if self.time_kernel_size != self.kernel_size:
            config["time_kernel_size"] = self.time_kernel_size
        if self.time_dilation != 1:
            config["time_dilation"] = self.time_dilation
        return config
