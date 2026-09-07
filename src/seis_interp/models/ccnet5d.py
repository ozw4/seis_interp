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


class CC3D2D(nn.Module):
    """Apply Conv3D/ReLU, then Conv2D over the remaining two spatial axes.

    Input and output use ``(B, C, T, Sx, Sy, Rx, Ry)``. Both convolutions
    preserve their spatial shapes with same zero padding. The intermediate
    channels are reduced by Conv2D, without an additional branch sum.
    """

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("intermediate_channels", intermediate_channels),
            ("out_channels", out_channels),
            ("kernel_size", kernel_size),
        ):
            _validate_positive_integer(name, value)
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        _validate_output_activation(output_activation)
        self.output_activation = output_activation
        self.conv3d = nn.Conv3d(
            in_channels, intermediate_channels, kernel_size, padding=kernel_size // 2, bias=True
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
    """

    def __init__(
        self,
        *,
        hidden_channels: int = 64,
        intermediate_channels: int = 64,
        kernel_size: int = 5,
        output_activation: str = "linear",
    ) -> None:
        super().__init__()
        _validate_positive_integer("hidden_channels", hidden_channels)
        _validate_positive_integer("intermediate_channels", intermediate_channels)
        _validate_output_activation(output_activation)
        self.hidden_channels = hidden_channels
        self.intermediate_channels = intermediate_channels
        self.kernel_size = kernel_size
        self.output_activation = output_activation
        self.network = nn.Sequential(
            CC3D2D(1, intermediate_channels, hidden_channels, kernel_size=kernel_size),
            CC3D2D(
                hidden_channels, intermediate_channels, hidden_channels, kernel_size=kernel_size
            ),
            CC3D2D(
                hidden_channels, intermediate_channels, hidden_channels, kernel_size=kernel_size
            ),
            CC3D2D(
                hidden_channels,
                intermediate_channels,
                1,
                kernel_size=kernel_size,
                output_activation=output_activation,
            ),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    @property
    def halo_radius(self) -> int:
        """Return each axis' receptive-field radius for four stride-one modules."""
        return 4 * (self.kernel_size // 2)

    def constructor_config(self) -> dict[str, object]:
        """Return every constructor value needed to restore the same function."""
        return {
            "hidden_channels": self.hidden_channels,
            "intermediate_channels": self.intermediate_channels,
            "kernel_size": self.kernel_size,
            "output_activation": self.output_activation,
        }
