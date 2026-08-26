# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Isolate exact coverage from batch size

**Status:** active

**Decision:** Compare exact full-batch coverage with random replacement at the same 5,000 points
per update and 50,000 updates on the identical seed-42 eight-trace subset.

**Reason:** Study 005 showed that full-batch MSE fits the subset strongly, but that change also
increased batch size, reduced gradient variance, and guaranteed complete point coverage. Holding
batch size and point budget fixed isolates whether exact coverage is necessary.

**Evidence:** Study 005 run `20260826T020640Z_b550db8_mse_control` reached 32.48 dB best
median training-trace S/N.

**Impact:** Production training remains unchanged while the batching cause is diagnosed.
