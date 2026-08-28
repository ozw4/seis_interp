# study_009_full_ffid_trace_batch_correlation

## Status

`completed`

## Research question

Can the Study 005 trace-wise correlation auxiliary loss help one SIREN fit all 435 FFID 2348
training traces under the random complete-trace batches used by Study 008?

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split as the sampling
pool and retains all 625 time samples. Coordinates use training-min/max linear scaling plus
azimuth sine/cosine features, and amplitudes use the training-only global RMS. Validation and
test amplitudes are used only to validate the existing split contract; they do not enter
training, model selection, or metrics. The split contract fixes a 0.20 random trace holdout and
assigns 0.25 of that holdout to validation.

One seed-42 6-input SIREN has width 256, four sine layers, and `omega_0: 300.0`. It is freshly
initialized and trained with Adam and learning rate `1.0e-3`. Every update uniformly selects
eight distinct training traces without replacement within that update, then uses all 625
samples from each selected trace. Trace selection starts again from the full 435-trace pool on
the next update, so traces may recur across updates. The batch size is 5,000, the update count
is 50,000, and the total point-evaluation budget is 250,000,000.

The training objective is point-wise MSE plus `0.1` times the mean Study 005 trace-wise
`1 - correlation` loss over the eight complete traces in the current batch. The correlation
epsilon is `1.0e-4`. The `training.loss: l2` field identifies the primary MSE term; the fixed
auxiliary term is recorded separately by `experiment.correlation_weight` and
`experiment.correlation_eps`.

Every 500 updates, chunked prediction in batches of 65,536 over all 435 training traces and all
625 samples reports training fit on `cuda:0`. The reported fit metrics are median
training-trace S/N, global training S/N, median training-trace correlation, and
prediction/target RMS ratio; training-loss components are also recorded. The best point
maximizes median training-trace S/N.

## Acceptance and interpretation

- `strong_fit`: best median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, best median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

The corresponding summary decisions are `full_ffid_strong_fit`,
`full_ffid_escaped_zero_predictor`, and `full_ffid_near_zero`.

## Reproduction

```bash
python scripts/run_study_009_full_ffid_trace_batch_correlation.py \
  --config studies/study_009_full_ffid_trace_batch_correlation/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_009_full_ffid_trace_batch_correlation \
  --device cuda:0
```

## Expected outputs

One immutable run directory contains `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
and `run.json`. One sibling summary JSON records the single run and summary decision. No
checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only. It does not evaluate interpolation,
validation/test performance, temporal patches, or alternative optimizers and batches. Its
comparison with Study 008 targets the added correlation term, but this single run does not
establish broad causal behavior across seeds or batching strategies. The auxiliary correlation
loss is a POC-specific objective and is outside the source paper's reported L1, L2, and Huber
losses.

## Current conclusion

Trace-wise correlation loss did not make Study 008-style complete-trace batches escape the
near-zero predictor on all 435 training traces. The best median training-trace S/N was
-0.01773 dB with a 0.01802 prediction/target RMS ratio, yielding `full_ffid_near_zero`.

Historical rationale belongs in [`decisions.md`](decisions.md).
