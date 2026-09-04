# study_all_ffid_temp

## Purpose

Scratch workspace for repeated survey-wide SIREN experiments on the prepared
`all_ffids_per_ffid_random_split_amplitude_qc` dataset. Not a numbered study; its output is not an
immutable research record. Edit [`config.yaml`](config.yaml) directly between runs.

Do not use this overwriteable directory as formal evidence. Copy an accepted condition into a
numbered study and run it with an immutable run ID when the result needs to be retained.

## Fixed prepared-data contract

Changing any of these requires preparing a different processed dataset; every other model and
training value can change without rebuilding the split or normalization.

`project.random_seed`, `sampling.random_trace_holdout_fraction`,
`sampling.validation_fraction_of_holdout`, `sampling.split_scope`,
`sampling.trace_amplitude_filter`, `normalization.coordinates`, `normalization.amplitude`.

## Trace-amplitude eligibility

Determined from raw interim traces before split assignment and before normalization is fitted, so
it does not depend on `training.amplitude_scaling`: a trace is excluded when all 625 samples are
exactly zero, or when any sample exceeds `1.0e4` in absolute value. Amplitudes are never clipped
and individual time samples are never removed.

For the locked SEG C3 NA input this excludes 544 traces (107 all-zero, 437 excessive), all in
FFID 1746, leaving 2,303,480 traces across 4,780 FFIDs. Excluded rows keep the `excluded` split
label and are used by neither training nor validation.

## Training-time coordinate features

`model.coordinate_features` selects the model inputs derived at training time and never requires
rebuilding the prepared dataset. `model.input_features` must match the resulting count.

| Value | Inputs | `input_features` |
|---|---|---:|
| `cmp_offset_azimuth` (default) | time, cmp_x, cmp_y, offset, azimuth_sin, azimuth_cos | 6 |
| `cmp_cartesian_half_offset` | time, cmp_x, cmp_y, half_offset_x, half_offset_y | 5 |
| `cmp_cartesian_half_offset_radius` | the above plus normalized offset | 6 |

Cartesian components come from the stored physical headers as
`half_offset = 0.5 * (source - receiver)`. Both half-offset axes are divided by the same symmetric
scale `0.5 * prepared_training_max_offset_m` rather than being fitted per axis: this keeps offset
magnitude and azimuth geometry coupled while avoiding held-out fitting. Time and CMP use the
min/max fitted from prepared training traces.

`model.time_coordinate_scale` multiplies only the already-normalized time coordinate, leaving every
spatial feature and the fitted physical bounds untouched. It must stay a positive, finite, nonzero
value in `float32`. It is a model-input transform, so changing it does not require rebuilding the
prepared split or fitted normalization.

`model.layer_omega_schedule: exponential` assigns a geometric progression of activation frequencies
across the sine layers from `model.omega_0` to `model.hidden_omega`, and requires at least two sine
layers. Otherwise `omega_0` applies to the first sine layer and `hidden_omega` to the rest.

`model.skip_connections: dense` makes each sine layer after the first — and the final linear layer
— consume the concatenated activations of every preceding sine layer. It combines with the
frequency schedule, but parameter and activation cost grow quadratically with `model.hidden_layers`.

## Training-target amplitude scaling

`training.amplitude_scaling` is independent of the fixed prepared-data normalization:

- `train_global_rms` divides every target by the RMS fitted from all training samples. This is the
  physical-amplitude interpolation contract used by Study 016.
- `per_trace_rms` divides each complete trace by its own RMS at training time, and divides
  validation traces by their own target RMS so early stopping measures waveform fit in the unit-RMS
  domain.

`per_trace_rms` is an **oracle-normalized diagnostic**. Its validation metric is not comparable
with Study 016's global-RMS S/N, and a held-out trace's physical amplitude cannot be reconstructed
because its RMS is unavailable at inference — the checkpoint records the scaling name but cannot
supply that scale. Do not promote it as a physical-amplitude interpolation result without a
separate gain model fitted only from training data. Artifacts label this domain
`oracle_per_trace_unit_rms`.

## Batch modes and staged FFID scope

`training.batch_mode` supports three diagnostic paths:

- `random_points` samples independent trace/time points and selects checkpoints by median
  validation trace S/N, over the materialized validation path.
- `full_ffid_epoch` uses one complete training FFID per update and selects checkpoints by streamed
  global validation S/N.
- `random_complete_traces` samples `training.traces_per_update` distinct rows uniformly from the
  training-trace pool with all 625 samples each, running `training.steps_per_epoch` updates per
  epoch. Targets are scaled only after their rows are selected; checkpoints and early stopping use
  streamed global validation S/N only.

For the two streamed modes, inclusive `training.ffid_range: [min, max]` applies the same FFID
selection to train, validation, and test without rewriting the prepared artifacts. Coordinate
bounds and the prepared training-global amplitude RMS stay those of the original survey-wide
preparation.

`random_complete_traces` accepts `training.evaluate_training_snr: true` to stream global S/N over
the selected training traces after every epoch. It does not affect checkpoint selection; leave it
false for survey-wide runs when the extra full training pass is too expensive.

It also accepts a cosine learning-rate decay, rejected by the other batch modes:

```yaml
training:
  learning_rate: 1.0e-4
  learning_rate_schedule: cosine
  minimum_learning_rate: 1.0e-6
```

The schedule advances after every optimizer update. Its horizon is the configured
`max_epochs * steps_per_epoch` even when early stopping ends the run sooner. The minimum must be
positive, finite, and strictly below `training.learning_rate`.

## Training loss

When `training.correlation_weight` is positive the condition optimizes

```text
L = MSE + correlation_weight * mean_trace(1 - corr)
```

This auxiliary term requires `batch_mode: full_ffid_epoch`; set `correlation_weight: 0` for the
other two modes. Each full-FFID update interprets the amplitude-scaled targets and matching
predictions in trace-major order as `(eligible_trace_count, 625)`, centers each correlation over
one trace's 625 samples, and averages `1 - corr` equally over that FFID's eligible traces. It is
computed after `training.amplitude_scaling`, so under `per_trace_rms` it operates in the per-trace
unit-RMS domain.

The stabilized correlation adds `correlation_eps: 1.0e-4` to each centered squared norm before
forming the denominator. This keeps the loss and its zero-prediction gradient finite, but makes the
regularized correlation only approximately scale invariant: the epsilon belongs to the chosen
scaled-amplitude domain, so the same value does not behave identically under `train_global_rms` and
`per_trace_rms`.

Correlation is training-only. Validation and early stopping continue to use streamed global S/N.

## Run

Prepare the amplitude-filtered split once, or regenerate it after changing a fixed prepared-data
condition:

```bash
python -m seis_interp.cli data prepare-baseline \
  --config studies/study_all_ffid_temp/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc \
  --overwrite

python scripts/run_study_all_ffid_temp.py [--device cuda:1]
```

All three batch modes print a flushed start and end line per epoch. The runner trains into a
staging directory and replaces `runs/study_all_ffid_temp/current` only on success, so a failed run
leaves the previous output in place.
