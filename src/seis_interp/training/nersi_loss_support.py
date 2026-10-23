"""Select the recorded time samples without changing the complete-trace objective."""

import torch


def recorded_trace_samples(prediction, target, starts=None, sample_count=None):
    """Gather each trace's contiguous recording; bypass gathering for a full canvas."""
    if starts is None:
        return prediction, target
    indices = torch.as_tensor(starts, device=prediction.device)[:, None] + torch.arange(
        sample_count, device=prediction.device
    )
    return prediction.gather(1, indices), target.gather(1, indices)
