# study_013_amplitude_balancing

## Status

`completed`

## Research question

Can per-trace RMS amplitude balancing let one SIREN escape the near-zero predictor and fit all
435 FFID 2348 training traces under the same 5,000-point random-replacement and 50,000-update
budget as Studies 007 and 012?

## Motivation

Studies 007-012 remained near zero on the full 435-trace training pool across every tested
batching structure, auxiliary loss, continuation schedule, and SIREN parameterization, while
pools of up to 64 traces reached strong fits. A 2026-08-27 analysis of the prepared FFID 2348
dataset localizes a candidate cause in the amplitude distribution rather than in optimization
structure: after the global-RMS normalization, per-trace RMS spans 0.15 to 15.6, the ten
highest-energy traces hold 79% of the total training energy, and the top 1% of points hold 94%.
The highest-RMS traces are the near-offset traces (20-82 m; correlation of log RMS with log
offset is -0.946), and 90% of the strongest trace's energy lies in its first 16 of 625 samples.
The Study 011 nested pools kept maximum trace RMS at or below 1.12 through the 64-trace stage
(all strong fits), while the 128-trace stage introduced an RMS-5.5 trace holding 48% of that
pool's energy, exactly where continuation collapsed. In the Study 007/012 failures, the mean
training loss never left the data variance near 1.0, and the legacy and official-SIREN loss
histories were identical to about 1e-4, showing that model output contributed almost nothing to
the loss. This study tests whether removing the amplitude imbalance is sufficient to escape the
near-zero predictor.

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split and all 625 time
samples. Coordinates use training-min/max linear scaling plus azimuth sine/cosine features, and
amplitudes use the training-only global RMS. Validation and test amplitudes are used only to
validate the existing split contract; they do not enter training, model selection, or metrics.
The split contract fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to
validation.

Every condition uses the legacy 6-input SIREN with width 256, four sine layers,
`omega_0: 300.0`, and `hidden_omega: 1.0` — the parameterization under which pools up to 64
traces reached strong fits and which Study 012 showed behaves identically to the official
package on this probe. Each condition starts with a fresh model, Adam optimizer, and
`RandomPointSampler`. NumPy, PyTorch, and CUDA are reseeded to 42 immediately before model
construction, and each sampler uses seed 42, so all three conditions draw identical point
sequences. Training uses learning rate `1.0e-3`, uniform sampling with replacement, batch size
5,000, and 50,000 updates on `cuda:0`, for 250,000,000 sampled point evaluations per condition.
Every 500 updates, chunked prediction in batches of 65,536 evaluates all 435 by 625 training
points against that condition's own training target.

## Conditions

Exactly three conditions are run in this order:

| Condition | Amplitude target | Loss |
|---|---|---|
| `global_rms_control` | global-RMS normalization (unchanged) | pure MSE |
| `per_trace_rms` | global-RMS then each training trace scaled to unit RMS | pure MSE |
| `huber_global_rms` | global-RMS normalization (unchanged) | Huber, delta 1.0 |

`global_rms_control` is an exact re-execution of the Study 007/012 near-zero condition and
gates the comparison. `per_trace_rms` removes the trace-energy imbalance from the regression
target; the per-trace scales are derived from training traces only. `huber_global_rms` is a
secondary paper-sanctioned robust-loss observation that bounds the influence of extreme points
without changing the target; it does not enter the summary decision.

For `per_trace_rms`, training-fit metrics are computed against the per-trace-scaled target.
Median trace S/N and median trace correlation are invariant to scaling each trace's target and
prediction by the same factor, so the classification thresholds keep their meaning; global S/N
and the prediction/target RMS ratio reweight traces and are not directly comparable across
scalings.

## Acceptance and interpretation

For each condition, the report with maximum median training-trace S/N is classified as follows:

- `strong_fit`: median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

If `global_rms_control` is not `near_zero`, the summary decision is
`global_rms_control_not_reproduced`, regardless of the other conditions. Otherwise,
`per_trace_rms` maps to `per_trace_rms_strong_fit`, `per_trace_rms_escaped_zero_predictor`, or
`per_trace_rms_near_zero` according to its classification. The `huber_global_rms`
classification is recorded in the summary but does not select the decision. Best reports are
used only for diagnostic classification; no report selects a checkpoint or changes training.

## Reproduction

```bash
python scripts/run_study_013_amplitude_balancing.py \
  --config studies/study_013_amplitude_balancing/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_013_amplitude_balancing \
  --device cuda:0
```

## Expected outputs

Each of the three conditions produces one immutable run directory containing only
`config.resolved.yaml`, `inputs.lock.json`, `metrics.json`, and `run.json`. One sibling summary
JSON records all condition outcomes, the per-trace scale statistics, and the summary decision.
No checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only and does not evaluate validation/test
interpolation. Per-trace RMS scales are unavailable at held-out coordinates, so a positive
result motivates a coordinate-dependent gain model (for example, a smooth offset-dependent
spreading correction fitted on training traces) before interpolation can be evaluated; that
model is out of scope here. The Huber condition tests a single delta of 1.0 and is not a delta
sweep. A positive per-trace result supports the amplitude-imbalance explanation under this
fixed budget but does not decompose which part of the imbalance (dynamic range, gradient
domination, or SIREN output-scale limits) is causal.

## Current conclusion

| Condition | Classification | Best step | Best median S/N | Best median correlation | Best RMS ratio |
|---|---|---:|---:|---:|---:|
| `global_rms_control` | `near_zero` | 19,000 | -0.0181 dB | 0.0018 | 0.0222 |
| `per_trace_rms` | `near_zero` | 4,000 | -0.0029 dB | 0.0003 | 0.0265 |
| `huber_global_rms` | `near_zero` | 36,500 | -0.0013 dB | 0.0075 | 0.0096 |

The control reproduced the Study 007/012 near-zero result exactly, validating the comparison.
Scaling every training trace to unit RMS did not let training escape the zero predictor: the
per-trace condition's mean training loss stayed at the unit data variance near 1.0 for all
50,000 updates, exactly as the unbalanced control did. The Huber condition also remained near
zero with a flat loss. The summary decision is `per_trace_rms_near_zero`: amplitude imbalance is
not a sufficient explanation for the fresh full-pool failure, so removing it alone does not fix
optimization under this budget. The amplitude-concentration observations remain factual but
their causal role, if any, is limited to at most an interaction with another condition (for
example the Study 011 continuation collapse), which this study does not test.

Historical rationale belongs in [`decisions.md`](decisions.md).
