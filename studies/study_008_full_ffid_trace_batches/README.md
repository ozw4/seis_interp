# study_008_full_ffid_trace_batches

## Status

`completed`

## Research question

Can one SIREN fit all 435 FFID 2348 training traces when every update uses eight uniformly
selected complete traces, for a 5,000-point batch and a fixed 50,000-update budget?

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split as the sampling
pool and retains all 625 time samples. Coordinates use training-min/max linear scaling plus
azimuth sine/cosine features, and amplitudes use the training-only global RMS. Validation and
test amplitudes are used only to validate the existing split contract; they do not enter
training, model selection, or metrics. The split contract fixes a 0.20 random trace holdout and
assigns 0.25 of that holdout to validation.

One seed-42 6-input SIREN has width 256, four sine layers, and `omega_0: 300.0`. It is freshly
initialized and trained with pure MSE, Adam, and learning rate `1.0e-3`; no correlation loss is
used. Every update uniformly selects eight distinct training traces without replacement within
that update, then uses all 625 samples from each selected trace. Trace selection starts again
from the full 435-trace pool on the next update, so traces may recur across updates. The batch
size is 5,000, the update count is 50,000, and the total point-evaluation budget is 250,000,000.

Every 500 updates, chunked prediction in batches of 65,536 over all 435 training traces and all
625 samples reports training fit on `cuda:0`. The reported metrics are median training-trace
S/N, global training S/N, median training-trace correlation, prediction/target RMS ratio, and
mean training loss since the prior report. The best point maximizes median training-trace S/N.

## Acceptance and interpretation

- `strong_fit`: best median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, best median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

The corresponding summary decisions are `full_ffid_strong_fit`,
`full_ffid_escaped_zero_predictor`, and `full_ffid_near_zero`.

## Reproduction

```bash
python scripts/run_study_008_full_ffid_trace_batches.py \
  --config studies/study_008_full_ffid_trace_batches/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_008_full_ffid_trace_batches \
  --device cuda:0
```

## Expected outputs

One immutable run directory contains `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
and `run.json`. One sibling summary JSON records the single run and summary decision. No
checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only. It does not evaluate interpolation,
validation/test performance, temporal patches, correlation loss, or alternative optimizers and
batches. It changes the within-update sampling structure relative to Study 007 but does not
isolate every possible effect of trace-complete batching and is not a complete reproduction of
the paper's 5D setup.

## Current conclusion

Random complete-trace batches did not escape the near-zero predictor on all 435 training traces.
The best median training-trace S/N was -0.00116 dB with a 0.00548 prediction/target RMS ratio,
yielding `full_ffid_near_zero`; this one-seed result does not test temporal patching.

Historical rationale belongs in [`decisions.md`](decisions.md).
