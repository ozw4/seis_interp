"""Implicit neural field emitting one complete trace per continuous coordinate."""

from numbers import Integral

from torch import nn

from seis_interp.models.nersi import FourierFeatureMapping


class TraceCoordinateField(nn.Module):
    """Map four continuous trace coordinates to the whole trace through one linear head.

    Four unrounded local coordinates identify a trace. An exponential Fourier
    encoding feeds a GELU multilayer perceptron whose output layer emits exactly
    ``time_sample_count`` samples, so no sample of the output is discarded and
    every parameter contributes to the returned trace. There is no
    two-dimensional canvas, no transposed convolution and no pixel shuffle; the
    time axis is an output axis of a single linear map rather than an upsampled
    image dimension.

    Only :class:`~seis_interp.models.nersi.FourierFeatureMapping` is shared with
    the NeRSI models. It holds no parameters, and sharing it keeps
    ``frequency_base`` and ``fourier_components`` identical in meaning across the
    two neural-implicit parameterizations; the architecture shares nothing.
    """

    def __init__(
        self,
        *,
        time_sample_count,
        input_features=4,
        fourier_components=40,
        frequency_base=1.1,
        hidden_width=256,
        hidden_layers=4,
        activation="gelu",
        output_activation="linear",
    ):
        super().__init__()
        if input_features != 4:
            raise ValueError("the trace coordinate field requires four coordinates")
        if activation != "gelu" or output_activation != "linear":
            raise ValueError("the trace coordinate field requires GELU and linear output")
        self.time_sample_count = _positive_integer(time_sample_count, "time_sample_count")
        self.hidden_width = _positive_integer(hidden_width, "hidden_width")
        self.hidden_layers = _positive_integer(hidden_layers, "hidden_layers")
        self.encoding = FourierFeatureMapping(
            input_features=input_features,
            fourier_components=fourier_components,
            frequency_base=frequency_base,
        )
        incoming = (
            self.encoding.output_features,
            *((self.hidden_width,) * (self.hidden_layers - 1)),
        )
        self.trunk = nn.Sequential(
            *(
                layer
                for width in incoming
                for layer in (nn.Linear(width, self.hidden_width), nn.GELU())
            )
        )
        self.output_layer = nn.Linear(self.hidden_width, self.time_sample_count)

    def forward(self, coordinates):
        """``(batch, 4)`` coordinates to ``(batch, time_sample_count)`` amplitudes."""
        return self.output_layer(self.trunk(self.encoding(coordinates)))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def field_parameter_count(
    *, time_sample_count, fourier_components, hidden_width, hidden_layers, input_features=4
):
    """Parameters of the declared field without constructing it.

    Used to solve for the width that meets a matched parameter budget on a given
    time canvas; it mirrors the module above exactly.
    """
    if not isinstance(time_sample_count, Integral) or not isinstance(hidden_width, Integral):
        raise ValueError("time_sample_count and hidden_width must be integers")
    encoded = input_features * 2 * fourier_components
    first = encoded * hidden_width + hidden_width
    hidden = (hidden_layers - 1) * (hidden_width * hidden_width + hidden_width)
    head = hidden_width * time_sample_count + time_sample_count
    return first + hidden + head


def matched_hidden_width(
    *, time_sample_count, fourier_components, hidden_layers, parameter_count_target, **kwargs
):
    """Smallest absolute-error width meeting the matched parameter target."""
    best = None
    for width in range(1, 4096):
        count = field_parameter_count(
            time_sample_count=time_sample_count,
            fourier_components=fourier_components,
            hidden_width=width,
            hidden_layers=hidden_layers,
            **kwargs,
        )
        error = abs(count - parameter_count_target)
        if best is None or error < best[1]:
            best = (width, error, count)
        if count > parameter_count_target and best[0] != width:
            break
    width, _, count = best
    return width, count
