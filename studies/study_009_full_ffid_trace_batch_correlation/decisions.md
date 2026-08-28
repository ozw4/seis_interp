# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Add trace-wise correlation to random complete-trace batches

**Context:** Study 008 remained near zero when every update trained with pure MSE on eight
randomly selected complete traces from the 435-trace training pool. Study 005 already defines a
trace-wise correlation loss for complete fixed-length traces, so the same loss has an
unambiguous trace axis under the Study 008 batching structure.

**Decision:** Run one seed-42 condition with the Study 008 sampling, model, optimizer, batch
size, update budget, and full-training evaluation unchanged. Replace the pure-MSE objective
with MSE plus `0.1` times mean trace-wise `1 - correlation` over the eight current complete
traces, using epsilon `1.0e-4`. Keep `training.loss: l2` for the primary term and record the
auxiliary term explicitly in the experiment contract.

**Evidence:** Study 008 run `20260826T065417Z_fa548ba_tracebatch8_trace435` reached -0.001160 dB
best median training-trace S/N and remained `near_zero`. Study 005 used correlation weight
`0.1` and epsilon `1.0e-4` on eight complete 625-sample traces; its MSE control already fit that
fixed subset, so it did not determine whether correlation loss can help across the full
training pool.

**Impact:** This diagnostic tests the correlation auxiliary term where its trace-wise meaning
is preserved without changing Study 008's 5,000-point sampling structure or point budget. It
does not modify production training or evaluate interpolation performance.

## 2026-08-26 — Trace-wise correlation does not escape the near-zero predictor

**Context:** The single fixed condition completed all 50,000 updates and 250,000,000 point
evaluations. Every update used eight complete traces, and all 100 full-training report points
and loss components were finite.

**Decision:** Record `full_ffid_near_zero`. The best report occurred at step 3,500 with
-0.017734 dB median training-trace S/N, -0.001588 dB global S/N, 0.000583 median trace
correlation, and a 0.018018 prediction/target RMS ratio. Final median training-trace S/N was
-0.098969 dB.

**Evidence:** Run `20260826T071528Z_3013774_tracebatch8_corr0p1_trace435`; summary
`20260826T071528Z_3013774_summary.json`.

**Impact:** Adding the Study 005 trace-wise correlation term did not make the Study 008-style
full-training probe fit under this one-seed fixed budget. This result does not establish how the
loss behaves across seeds or under temporal patching and does not evaluate interpolation.
