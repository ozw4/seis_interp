# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Probe the full training split with the successful large random batch

**Context:** Study 004 remained near zero for 435 traces with 1,024-point batches and 50,000
updates. Study 006 strongly fit the fixed eight-trace subset with 5,000-point random-replacement
batches, but that condition also increased the total point-evaluation budget.

**Decision:** Run one seed-42 condition over all 435 training traces and 625 samples with
random-replacement batch size 5,000, pure MSE, and 50,000 updates. Evaluate full-training fit
every 500 updates and classify the best median training-trace S/N report using the fixed
strong-fit, escaped-zero-predictor, and near-zero thresholds.

**Evidence:** Study 006 summary `20260826T045538Z_55b4b9d_summary.json` recorded 30.14 dB best
median training-trace S/N for its eight-trace random-replacement condition. The existing FFID
2348 split contains 435 training traces and 625 samples per trace.

**Impact:** This diagnostic tests whether the successful large random batch scales to the full
training split. It does not separate batch-size and total-point-budget effects and does not
change production training.
