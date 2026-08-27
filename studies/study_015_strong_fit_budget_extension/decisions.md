# Decisions

## 2026-08-27: New study, not another Study 014 run

`docs/repository_layout.md` keeps seed- or budget-only variations of the same question inside
the same study as new runs. This experiment changes the research question, not just the budget:
Study 014 asked which ingredient causes the 435-trace escape, while this study asks whether the
recommended recipe converges to `strong_fit` at all and at what step. Re-running Study 014's
five-condition runner at 400,000 updates would also waste about six GPU hours on the four
conditions whose verdicts (control, correlation, attribution) are already settled. A separate
single-condition study keeps the completed Study 014 contract immutable.

## 2026-08-27: 400,000-update budget

Study 014's `full_trace_batch_per_trace_rms` curve gains roughly 1.3 dB per doubling of updates
(12.1 dB at step 5,000 to 16.14 dB at step 44,500). Reaching 20 dB needs about three more
doublings from 44,500, i.e. about 360,000 updates, so the budget is set to 400,000 (8x
Study 014) — large enough to cover the extrapolation with margin, small enough to remain one
~2 hour single-GPU run.

## 2026-08-27: Single condition with a reproduction gate instead of a control

The near-zero control and the ingredient attribution are settled Study 014 facts; repeating
them at 8x budget adds cost without information. Instead, determinism is the gate: with seed,
model, data, and batch schedule identical to Study 014, the first 50,000 updates must reproduce
Study 014's recorded best (16.13774316268436 dB within 0.05 dB) or the decision is
`baseline_not_reproduced`. The tolerance is deliberately tight because Study 014 itself
reproduced the informal scratch run to four decimal places under the same setup.
