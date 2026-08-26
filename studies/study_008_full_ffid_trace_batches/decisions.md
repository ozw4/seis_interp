# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Probe the full training split with random complete-trace batches

**Context:** Study 007 remained near zero when each update independently sampled 5,000 points
with replacement from all 435 training traces. Study 005 showed that a fixed complete
eight-trace batch can be fit with pure MSE, so correlation loss is not required for this probe.

**Decision:** Run one seed-42 condition over all 435 training traces and 625 samples. At each
update, uniformly select eight distinct training traces without replacement within the update
and train on all 625 samples from each trace. Use pure MSE, batch size 5,000, and 50,000 updates;
do not apply correlation loss. Evaluate full-training fit every 500 updates using the existing
classification thresholds.

**Evidence:** Study 007 run `20260826T061901Z_b0af699_random5000_trace435` reached only -0.01808
dB best median training-trace S/N. Study 005 run `20260826T020640Z_b550db8_mse_control` reached
32.48 dB on its fixed eight-trace complete batch with pure MSE.

**Impact:** This diagnostic tests whether preserving complete trace structure within each update
changes full-FFID training fit at the same 5,000-point batch size and 250,000,000 point budget.
It does not change production training or evaluate interpolation performance.
