# study_all_ffid_temp

## Purpose

This is a scratch workspace for repeated survey-wide SIREN experiments using the prepared
`all_ffids_per_ffid_random_split_amplitude_qc` dataset. It is not a numbered study and its latest
output is not an immutable research record.

Edit [`config.yaml`](config.yaml) directly between runs. Its values reuse the prepared-data and
model conditions from `study_016_all_ffid_siren`, while exposing per-trace RMS, batch mode, and
trace-correlation settings as experiment knobs. Study 016 itself remains a pure-L2,
train-global-RMS condition and is not changed by this scratch experiment.

## Fixed prepared-data contract

The existing processed dataset was prepared with the following values. Keep them unchanged when
reusing it:

- `project.random_seed`
- `sampling.random_trace_holdout_fraction`
- `sampling.validation_fraction_of_holdout`
- `sampling.split_scope`
- `sampling.trace_amplitude_filter`
- `normalization.coordinates`
- `normalization.amplitude`

Changing one of those values requires preparing a different processed dataset. Model and training
values can be changed without rebuilding the split or normalization.

## Trace-amplitude eligibility

Amplitude eligibility is determined from raw interim traces before split assignment and before
normalization is fitted. This rule does not depend on `training.amplitude_scaling`:

- exclude a trace when all 625 samples are exactly zero;
- exclude a trace when any sample has absolute amplitude greater than `1.0e4`;
- never clip amplitudes or remove individual time samples.

For the locked SEG C3 NA input this excludes exactly 544 traces: 107 all-zero traces and 437
excessive-amplitude traces, all belonging to FFID 1746. The remaining 2,303,480 traces across
4,780 FFIDs are split and normalized. Excluded rows remain recorded with the `excluded` split
label but are used by neither training nor validation.

## Training-target amplitude scaling

`training.amplitude_scaling` is independent of the fixed prepared-data normalization above and
accepts two values:

- `train_global_rms` divides every target by the RMS fitted from all training samples. This is
  the physical-amplitude interpolation contract used by Study 016.
- `per_trace_rms` divides each complete trace by that trace's own RMS at training time.

For `per_trace_rms`, validation traces are also divided by their own target RMS so that early
stopping measures waveform fit in the unit-RMS target domain. This is an oracle-normalized
diagnostic: the validation metric is not directly comparable with Study 016's global-RMS S/N,
and a held-out trace's physical amplitude cannot be reconstructed because its RMS is unavailable
at inference. The checkpoint records the scaling name but cannot supply that unknown scale. Do
not promote this condition as a physical-amplitude interpolation result without a separate gain
model fitted only from training data. Generated metrics, run metadata, and the checkpoint label
this metric domain as `oracle_per_trace_unit_rms`.

## Batch modes and staged FFID scope

`training.batch_mode` supports three diagnostic paths:

- `random_points` samples independent trace/time points and selects checkpoints by median
  validation trace S/N. It retains the original materialized validation path.
- `full_ffid_epoch` uses one complete training FFID per update and selects checkpoints by streamed
  global validation S/N.
- `random_complete_traces` samples `training.traces_per_update` distinct rows uniformly from the
  selected training-trace pool, includes all 625 time samples from each row, and runs
  `training.steps_per_epoch` updates per epoch. Targets are scaled only after their rows are
  selected, and checkpoints and early stopping use only streamed global validation S/N.

For the two streamed modes, optional inclusive `training.ffid_range: [min, max]` applies the same
FFID selection to train, validation, and test groups without rewriting the prepared artifacts.
Coordinate bounds and the prepared training-global amplitude RMS remain those of the original
survey-wide preparation. The configured and effective FFID ranges and effective FFID count are
locked in the generated run records.

`random_complete_traces` can additionally set `training.evaluate_training_snr: true` to stream
global S/N over the selected training traces after every epoch. This is an optimization diagnostic
and does not affect checkpoint selection. Leave it false for survey-wide runs when the extra full
training pass is too expensive.

## Training loss

When `training.correlation_weight` is positive, the scratch condition uses the configured `l2`
loss as its pointwise MSE term and optimizes

```text
L = MSE + correlation_weight * mean_trace(1 - corr)
```

This auxiliary term requires `batch_mode: full_ffid_epoch`. For each full-FFID update, the already
amplitude-scaled targets and matching predictions are
interpreted in trace-major order as `(eligible_trace_count, 625)`. Each correlation is centered
over the 625 time samples of one complete trace, and the resulting `1 - corr` values are averaged
equally over the eligible traces in that FFID. The auxiliary term is therefore computed after
`training.amplitude_scaling`; with a `per_trace_rms` setting it operates in the
per-trace unit-RMS target domain.

The stabilized correlation adds `correlation_eps: 1.0e-4` to each centered squared norm before
forming the denominator. This keeps the loss and its zero-prediction gradient finite, but makes
the regularized correlation only approximately scale invariant: the epsilon belongs to the
chosen scaled-amplitude domain, and the same value does not have identical behavior under
`train_global_rms` and `per_trace_rms`.

Correlation is a training-only auxiliary term. Validation and early stopping are unchanged and
continue to use streamed global S/N; with `per_trace_rms`, that remains the oracle per-trace
unit-RMS validation domain described above.

With `batch_mode: random_points` or `batch_mode: random_complete_traces`, set
`correlation_weight: 0`. The reusable `train siren` path currently enables the auxiliary
correlation objective only for `full_ffid_epoch`; the other modes train with the configured
pointwise loss alone.

## Run

Prepare the amplitude-filtered split once, or regenerate it after changing a fixed prepared-data
condition:

```bash
python -m seis_interp.cli data prepare-baseline \
  --config studies/study_all_ffid_temp/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc \
  --overwrite
```

From the repository root:

```bash
python scripts/run_study_all_ffid_temp.py
```

All three batch modes print an immediately flushed start and end line for each epoch. The
`random_points` end line includes the epoch mean training loss and both validation S/N metrics.
For example, a `per_trace_rms` run prints:

```text
random_points 1/10 start: steps_per_epoch=500 batch_size=3000000 amplitude_scaling=per_trace_rms validation_metric_domain=oracle_per_trace_unit_rms
random_points 1/10 end: train_loss=... oracle_per_trace_unit_rms_median_trace_snr_db=... oracle_per_trace_unit_rms_global_snr_db=...
```

To override only the execution device:

```bash
python scripts/run_study_all_ffid_temp.py --device cuda:1
```

The runner trains into a staging directory. After a successful run it replaces
`runs/study_all_ffid_temp/current`; if training fails, the previous successful output remains.
The output contains the normal `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
`run.json`, and `artifacts/best.pt` files.

Do not use this overwriteable directory as formal evidence. Copy an accepted condition into a
numbered study and run it with an immutable run ID when the result needs to be retained.
