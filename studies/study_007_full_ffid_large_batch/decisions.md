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

## 2026-08-26 — Full training split remains near zero

**Context:** The fixed single condition completed all 50,000 updates and 250,000,000 training
point evaluations over 435 traces by 625 samples. All 100 full-training report points were
finite.

**Decision:** Record `full_ffid_near_zero`. The best report occurred at step 19,000 with -0.01808
dB median training-trace S/N, -0.001365 dB global S/N, 0.001756 median trace correlation, and a
0.02225 prediction/target RMS ratio. Final median training-trace S/N was -0.03997 dB.

**Evidence:** Run `20260826T061901Z_b0af699_random5000_trace435`; summary
`20260826T061901Z_b0af699_summary.json`.

**Impact:** The large random batch that strongly fit eight traces did not fit the full 435-trace
training set under this fixed setup. This result does not separate the effects of batch size and
total point budget and does not evaluate interpolation performance.
