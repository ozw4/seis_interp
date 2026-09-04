# study_015_strong_fit_budget_extension

## Status

`completed`

## Research question

Study 014 attributed the first 435-trace escape from the near-zero predictor to the full
complete-trace batch, with per-trace RMS balancing as a strong amplifier, and recommended the
recipe "full complete-trace batch plus per-trace RMS, no correlation loss". That condition
reached a best median training-trace S/N of 16.14 dB within 50,000 updates and was still
climbing at budget exhaustion. Does the recommended recipe reach the 20 dB `strong_fit` bar
when the update budget is extended, and at what step?

This is a new research question — budget sufficiency for convergence to `strong_fit` — not a
seed or epoch variation of Study 014's ingredient-attribution question, which is why it is a
separate study rather than another Study 014 run.

## Motivation

Study 014's `full_trace_batch_per_trace_rms` learning curve is close to log-linear in updates:
about 12.1 dB at step 5,000, 13.5 dB at 10,000, 14.5 dB at 25,500, and 16.14 dB at 44,500 —
roughly 1.3 dB per doubling of updates. Extrapolating that slope, the 20 dB threshold needs
about three more doublings from step 44,500, landing near 360,000 updates. An 8x budget of
400,000 updates covers that estimate with margin while remaining a single ~2 hour run.

## Fixed conditions

The experiment reuses Study 014's data and training contract exactly: every `train` row of the
FFID 2348 trace split (435 traces, all 625 samples), training-min/max linear coordinate scaling
plus azimuth sine/cosine features, training-only global RMS amplitude normalization followed by
per-trace unit-RMS balancing, the 6-input SIREN with width 256, four sine layers,
`omega_0: 30.0`, and `hidden_omega: 30.0`, Adam at learning rate `1.0e-4`, and seed 42 for
NumPy, PyTorch, CUDA, and the trace sampler. Every 500 updates, chunked prediction in batches
of 65,536 evaluates all 435 by 625 training points against the per-trace-scaled target.

## Condition

Exactly one condition is run: `full_trace_batch_per_trace_rms`, identical to Study 014's
condition of the same name — every update trains on all 435 complete traces (271,875 points,
`random_complete_traces` without replacement) against the per-trace unit-RMS target with the
plain L2 loss and no correlation term — except that `training.total_updates` is 400,000
instead of 50,000.

## Acceptance and interpretation

The report with maximum median training-trace S/N is classified with the shared thresholds:

- `strong_fit`: median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

One gate precedes the decision. Because the seed, model, data, and batch schedule are identical
to Study 014's run, the first 50,000 updates must reproduce it: the best median training-trace
S/N among reports at or before step 50,000 must match Study 014's recorded 16.1377 dB
within 0.05 dB, otherwise the decision is `baseline_not_reproduced`. Otherwise the decision
follows the classification: `extended_budget_strong_fit`, `extended_budget_escaped_zero_predictor`,
or `extended_budget_near_zero`. The summary also records `first_strong_fit_step`, the first
report step whose median training-trace S/N reaches 20 dB (null if never reached).

## Reproduction

```bash
python scripts/run_study_015_strong_fit_budget_extension.py \
  --config studies/study_015_strong_fit_budget_extension/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_015_strong_fit_budget_extension \
  --device cuda:0
```

## Expected outputs

One immutable run directory containing only `config.resolved.yaml`, `inputs.lock.json`,
`metrics.json`, and `run.json`, plus one sibling summary JSON recording the baseline gate, the
decision, and `first_strong_fit_step`. No checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only and does not evaluate validation/test
interpolation. It extends the budget of the single recommended condition; it does not revisit
the other Study 014 conditions, separate batch size from trace completeness, or sweep the
learning rate, so a failure to reach 20 dB within 400,000 updates would mean "not with this
recipe and budget", not "never".

## Current conclusion

Run `20260827T054748Z_94b479e_full_trace_batch_per_trace_rms`; decision
`extended_budget_strong_fit`.

The baseline gate passed exactly: the best median training-trace S/N within the first 50,000
updates was 16.1377 dB, identical to Study 014's recorded value to full float
precision, confirming a bit-exact reproduction of the shared seed-42 schedule.

The recommended recipe reaches `strong_fit`. The median training-trace S/N first crossed 20 dB
at step 199,000 (20.03 dB, median trace correlation 0.9950), about 4x Study 014's budget. The
best report came at step 356,500 with 21.29 dB median trace S/N, 0.9963 median trace
correlation, 20.95 dB global S/N, and a 0.9934 prediction/target RMS ratio. The final report at
step 400,000 was 19.40 dB — late-training reports oscillate roughly 19-21 dB around the best
while the mean training loss still creeps down (0.0184 at step 400,000).

Returns diminish steeply: the 8x budget added 5.16 dB over Study 014's 16.14 dB best, and the
last 200,000 updates contributed only about 1.3 dB of that while the curve flattened into an
oscillating plateau near 20-21 dB. Under this fixed learning rate (`1.0e-4`) and model size,
more updates alone are unlikely to buy much beyond ~21 dB; a learning-rate schedule or larger
capacity would be the lever if a higher training fit were ever needed. The training-fit
question for the 435-trace pool is now closed — the natural next step is interpolation quality
on held-out traces, which first requires a coordinate-dependent amplitude model because the
per-trace RMS scales do not exist at held-out positions.

Historical rationale belongs in [`decisions.md`](decisions.md).
