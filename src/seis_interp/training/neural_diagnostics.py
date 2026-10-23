"""Read-only training diagnostics for the predeclared zero-predictor gate."""

import math

import torch

from seis_interp.training.trace_relative_loss import masked_trace_loss


class NeuralTrainingDiagnostics:
    """Accumulate G0 on updates 901–1000 and would-clip counts without changing gradients.

    Short probes report their last (up to) 100 updates, labelled insufficient for
    G0. Device scalars are read only when reporting, not once per update.
    """

    def __init__(self, updates, loss_name):
        self.last = min(updates, 1000)
        self.first = max(1, self.last - 99)
        self.loss_name = loss_name
        self.samples = []
        self.clipped = None
        self.gradient_updates = 0

    @torch.no_grad()
    def observe(self, step, prediction, target, loss):
        if not self.first <= step <= self.last:
            return
        prediction, target = prediction.detach(), target.detach()
        zero_loss = masked_trace_loss(
            torch.zeros_like(target), target, loss_name=self.loss_name, check_finite=False
        )
        self.samples.append(
            torch.stack(
                (
                    loss.detach().double(),
                    zero_loss,
                    prediction.double().square().mean(),
                    target.double().square().mean(),
                )
            )
        )

    @torch.no_grad()
    def gradients(self, model):
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if norms:
            above = (torch.stack(norms).norm() > 1.0).to(torch.int64)
            self.clipped = above if self.clipped is None else self.clipped + above
            self.gradient_updates += 1

    def records(self):
        result = {}
        if self.gradient_updates:
            result.update(
                would_clip_threshold=1.0,
                would_clip_fraction=float(self.clipped.cpu()) / self.gradient_updates,
                gradient_diagnostic_updates=self.gradient_updates,
            )
        if not self.samples:
            return result
        values = torch.stack(self.samples).cpu()
        loss, zero_loss, output_energy, target_energy = values.mean(0).tolist()
        finite = bool(torch.isfinite(values).all()) and zero_loss > 0 and target_energy > 0
        ratio = math.sqrt(output_energy / target_energy) if finite else None
        enough = self.last == 1000 and len(values) == 100
        result["g0"] = {
            "first_update": self.first,
            "last_update": self.last,
            "updates": len(values),
            "zero_predictor_loss": zero_loss if math.isfinite(zero_loss) else None,
            "mean_loss": loss if math.isfinite(loss) else None,
            "loss_over_L0": loss / zero_loss if finite else None,
            "output_rms_over_target_rms": ratio,
            "all_finite": finite,
            "status": (
                "insufficient_updates"
                if not enough
                else "passed"
                if finite and loss <= 0.95 * zero_loss and ratio >= 0.2
                else "failed"
            ),
        }
        return result
