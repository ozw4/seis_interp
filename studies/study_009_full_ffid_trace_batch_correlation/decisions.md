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
