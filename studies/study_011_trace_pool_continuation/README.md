# study_011_trace_pool_continuation

## Status

`completed`

## Research question

Can a single SIREN reach a strong fit on all 435 FFID 2348 training traces when training starts
on the Study 006 eight-trace subset and continues through fixed, nested expansions of the
training-trace pool?

## Fixed conditions

The experiment uses the existing FFID 2348 trace split and all 625 time samples. One seed-42
permutation of sorted `train` array rows defines the nested pool sizes
`[8, 16, 32, 64, 128, 256, 435]`; the first eight rows therefore match the Study 006 subset.
Coordinates use training-min/max linear scaling plus azimuth sine/cosine features, and
amplitudes use the training-only global RMS. Validation and test amplitudes are used only to
validate the split contract and do not enter training, model selection, or metrics. The split
contract fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to validation.

One freshly initialized seed-42 6-input SIREN has width 256, four sine layers, and
`omega_0: 300.0`. It uses pure point-wise L2 loss, Adam, and learning rate `1.0e-3` on `cuda:0`.
At every update, `RandomPointSampler` draws 5,000 points uniformly with replacement from the
current nested trace pool. Each stage runs exactly 50,000 updates, for 350,000 updates and
1,750,000,000 point evaluations over all seven stages. Stage sampler seeds are the project base
seed plus the zero-based stage index: `[42, 43, 44, 45, 46, 47, 48]`.

The same model parameters and Adam state at the final update of one stage are carried directly
into the next stage. The model and optimizer are not reset, and training never rewinds to a
best report point. No checkpoint is saved or loaded. The first stage's final median
training-trace S/N must be at least 20 dB; failure of this validity gate means the Study 006
strong-fit anchor was not reproduced. In that case, the run stops before later stages and
records `stage8_anchor_failed` while retaining the completed first-stage outputs.

Every 500 updates, chunked prediction in batches of 65,536 evaluates all points in the current
stage pool only. Reports contain median training-trace S/N, global training S/N, median
training-trace correlation, prediction/target RMS ratio, and mean training loss since the prior
report. Stage best points maximize median training-trace S/N, while the final state—not a best
state—is passed to the next stage.

## Acceptance and interpretation

After the first-stage validity gate passes, the best report point from the final 435-trace stage
is classified with the existing thresholds:

- `strong_fit`: median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at the same report exceeds 0.1.
- `near_zero`: otherwise.

The corresponding successful-gate summary decisions are `full_ffid_strong_fit`,
`full_ffid_escaped_zero_predictor`, and `full_ffid_near_zero`. Final metrics for every stage are
also retained because they describe the states actually carried forward.

## Reproduction

```bash
python scripts/run_study_011_trace_pool_continuation.py \
  --config studies/study_011_trace_pool_continuation/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_011_trace_pool_continuation \
  --device cuda:0
```

## Expected outputs

One immutable run directory contains `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
and `run.json`; `metrics.json` contains every stage result. One sibling summary JSON records the
validity-gate result, completed stage count, final completed-stage outcome, and summary decision.
No checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only. It does not evaluate validation/test
interpolation, alternative pool schedules, or alternative optimizers and losses. Its
1.75-billion-point budget is seven times the Study 007 budget, and it does not include a
350,000-update fresh full-pool control. A successful result would therefore show that this
fixed continuation path can fit the training set, but would not separate continuation from
additional compute or establish a general causal advantage.

## Current conclusion

The eight-trace anchor was reproduced, and continuation retained `strong_fit` through 64
traces. Training collapsed to `near_zero` at the 128-trace stage and did not recover at 256 or
435 traces; the final full-pool decision is `full_ffid_near_zero`. This one-seed run does not
separate the continuation schedule from its larger compute budget.

Historical rationale belongs in [`decisions.md`](decisions.md).
