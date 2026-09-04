# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Compare only two losses on one fixed subset

**Status:** active

**Decision:** Compare MSE alone against MSE plus correlation weight 0.1 on the identical
full-batch eight-trace subset, and keep this POC-specific loss out of the production trainer until
there is evidence for it.

**Reason:** Eight traces is the first Study 004 subset that failed, so it isolates the loss term
without changing the trace count. Adding an unproven auxiliary objective to the shared trainer
would spread an unvalidated condition across other studies.

**Evidence:** Study 004 recorded `near_zero` for the seed-42 eight-trace prefix.

**Impact:** The reusable trainer keeps its pointwise losses; the correlation term stays a study
condition.

## 2026-08-26 — Close the correlation-loss path

**Status:** active

**Decision:** Stop this path and attribute no causal benefit to the correlation loss.

**Reason:** The MSE control escaped collapse on its own, so the auxiliary term is not what changed
the outcome. The full-batch change is the remaining candidate cause and belongs to a separate
probe.

**Evidence:** Summary decision `full_batch_control_succeeds`; best median trace S/N and
correlation were 32.48 dB / 0.9997 for `mse_control` and 33.15 dB / 0.9998 for `mse_corr_0p1`.

**Impact:** The batching axis becomes the next study; no production loss setting is selected here.
