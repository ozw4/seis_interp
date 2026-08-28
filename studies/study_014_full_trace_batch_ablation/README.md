# study_014_full_trace_batch_ablation

## Status

`completed`

## Research question

An informal 2026-08-27 scratch run produced the first escape from the near-zero predictor on
the full 435-trace FFID 2348 training pool, but it changed three things at once relative to
every failing baseline: complete-trace batches covering the whole pool each update, a trace
correlation loss with weight 0.3, and per-trace RMS amplitude balancing. Which of these
ingredients is necessary, and is the full complete-trace batch alone sufficient?

## Motivation

Studies 004-013 never escaped the near-zero predictor on the fresh 435-trace pool: batching
structure (007, 008, 010), a correlation loss with weight 0.1 on small trace batches (009),
staged continuation (011), the official SIREN parameterization (012), and per-trace RMS
balancing or a Huber loss under 5,000-point random-replacement batches (013) all stayed near
zero, while pools up to 64 traces reached strong fits. The strongest surviving discriminator
was per-trace sample density per update: successful small-pool fits saw all 625 samples of
every trace in effect each update, while failing 435-trace runs sampled about 11.5 points per
trace per 5,000-point batch.

The informal scratch run combined `random_complete_traces` with all 435 traces per update
(271,875 points, restoring 625 samples per trace per update), a trace correlation loss with
weight 0.3, and per-trace RMS balancing, under `omega_0: 30.0`, `hidden_omega: 30.0`, and
learning rate `1.0e-4`. It reached a best median training-trace S/N of 16.39 dB at step 47,500
(median trace correlation 0.985, prediction/target RMS ratio 0.984) and was still improving at
50,000 updates. The same model and learning rate under 5,000-point random-replacement batches
stayed near zero, so the batch structure and/or the auxiliary ingredients did the work. Study
013 already showed per-trace RMS alone (under small batches) is not sufficient, and Study 009
showed a weight-0.1 correlation loss on 8-trace batches is not sufficient; this study isolates
the ingredients under the full-pool batch.

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split and all 625 time
samples. Coordinates use training-min/max linear scaling plus azimuth sine/cosine features, and
amplitudes use the training-only global RMS. Validation and test amplitudes are used only to
validate the existing split contract; they do not enter training, model selection, or metrics.
The split contract fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to
validation.

Every condition uses the 6-input SIREN with width 256, four sine layers, `omega_0: 30.0`, and
`hidden_omega: 30.0`, trained with Adam at learning rate `1.0e-4` for 50,000 updates on
`cuda:0` — exactly the settings of the informal escape run. Each condition starts with a fresh
model and optimizer. NumPy, PyTorch, and CUDA are reseeded to 42 immediately before model
construction, and each sampler uses seed 42. Every 500 updates, chunked prediction in batches
of 65,536 evaluates all 435 by 625 training points against that condition's own training
target.

## Conditions

Exactly five conditions are run in this order:

| Condition | Batch per update | Correlation weight | Amplitude target |
|---|---|---|---|
| `small_batch_control` | 5,000 random points with replacement | 0.0 | global RMS |
| `full_trace_batch` | all 435 complete traces (271,875 points) | 0.0 | global RMS |
| `full_trace_batch_correlation` | all 435 complete traces | 0.3 | global RMS |
| `full_trace_batch_per_trace_rms` | all 435 complete traces | 0.0 | per-trace unit RMS |
| `full_trace_batch_correlation_per_trace_rms` | all 435 complete traces | 0.3 | per-trace unit RMS |

`small_batch_control` reproduces the failing small-batch baseline under this exact model and
learning rate and gates the comparison. `full_trace_batch` tests whether the full complete-
trace batch alone is sufficient. The next two conditions each add exactly one auxiliary
ingredient, and the final condition reproduces the informal escape exactly. The correlation
conditions add `0.3 * (1 - trace correlation)` per trace to the MSE with epsilon `1.0e-4`; all
conditions use the L2 base loss. For per-trace-RMS conditions, training-fit metrics are
computed against the per-trace-scaled target, whose median-trace S/N and correlation
thresholds keep their meaning as in Study 013.

## Acceptance and interpretation

For each condition, the report with maximum median training-trace S/N is classified as
follows:

- `strong_fit`: median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

Two gates precede the decision. If `small_batch_control` is not `near_zero`, the decision is
`small_batch_control_not_reproduced`. Otherwise, if `full_trace_batch_correlation_per_trace_rms`
is `near_zero`, the informal escape did not replicate and the decision is
`combined_escape_not_reproduced`. Otherwise the decision follows the `full_trace_batch`
classification: `full_trace_batch_strong_fit`, `full_trace_batch_escaped_zero_predictor`, or
`full_trace_batch_near_zero`. The single-ingredient conditions are recorded in the summary but
do not select the decision; they attribute the escape when `full_trace_batch` alone stays near
zero.

## Reproduction

```bash
python scripts/run_study_014_full_trace_batch_ablation.py \
  --config studies/study_014_full_trace_batch_ablation/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_014_full_trace_batch_ablation \
  --device cuda:0
```

## Expected outputs

Each of the five conditions produces one immutable run directory containing only
`config.resolved.yaml`, `inputs.lock.json`, `metrics.json`, and `run.json`. One sibling
summary JSON records all condition outcomes, the gates, and the summary decision. No
checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only and does not evaluate validation/test
interpolation. The full-trace-batch conditions and the control differ in both points per
update (271,875 versus 5,000) and sampling scheme (complete traces without replacement versus
uniform with replacement), so a positive `full_trace_batch` result does not by itself separate
batch size from trace completeness; that split is left to a follow-up. The correlation weight
tests the single value 0.3 and is not a weight sweep. All conditions share one seed, one model
parameterization, and one update budget.

## Current conclusion

Run `20260827T041050Z_925e8e4_*`; decision `full_trace_batch_escaped_zero_predictor`.

| Condition | Classification | Best step | Best median S/N | Best median correlation | Best RMS ratio |
|---|---|---:|---:|---:|---:|
| `small_batch_control` | `near_zero` | 14,000 | 0.0039 dB | 0.0395 | 0.0177 |
| `full_trace_batch` | `escaped_zero_predictor` | 45,500 | 8.91 dB | 0.9349 | 0.9226 |
| `full_trace_batch_correlation` | `escaped_zero_predictor` | 2,000 | 8.19 dB | 0.9272 | 0.6074 |
| `full_trace_batch_per_trace_rms` | `escaped_zero_predictor` | 44,500 | 16.14 dB | 0.9878 | 0.9857 |
| `full_trace_batch_correlation_per_trace_rms` | `escaped_zero_predictor` | 47,500 | 16.39 dB | 0.9885 | 0.9874 |

Both gates held: the control reproduced the near-zero baseline under this exact model and
learning rate, and the combined condition reproduced the informal escape exactly (best median
S/N 16.3882 dB at step 47,500, matching the scratch run to four decimals, as expected from the
shared seed and settings).

The full complete-trace batch alone is sufficient to escape the near-zero predictor: with no
auxiliary ingredient it reached 8.91 dB (median trace correlation 0.935, RMS ratio 0.92) and
was still improving at 50,000 updates. This is the first attribution of the 435-trace escape
and confirms the per-trace sample-density hypothesis that survived Studies 004-013.

The ingredient attribution is clean. Per-trace RMS balancing roughly doubles the fit quality
under the full-trace batch (16.14 dB versus 8.91 dB), even though Study 013 showed it does
nothing under small random batches — amplitude imbalance is an amplifier of the small-batch
failure, not its cause. The correlation loss contributes almost nothing: it slightly hurt the
global-RMS condition (best 8.19 dB at step 2,000, final 7.07 dB versus 8.74 dB without) and
added only 0.25 dB on top of per-trace balancing (16.39 versus 16.14 dB). The recommended
recipe going forward is the full complete-trace batch plus per-trace RMS balancing, with no
correlation loss. None of the conditions reached the 20 dB `strong_fit` bar within 50,000
updates, but the two per-trace conditions were still climbing at budget exhaustion; a longer
budget is the natural follow-up, alongside separating batch size from trace completeness.

Historical rationale belongs in [`decisions.md`](decisions.md).
