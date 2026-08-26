# study_007_full_ffid_large_batch

## Status

`active`

## Research question

Can one SIREN fit all 435 FFID 2348 training traces under 5,000-point uniform
random-replacement batches and a fixed 50,000-update budget?

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split and all 625 time
samples. Coordinates use training-min/max linear scaling plus azimuth sine/cosine features, and
amplitudes use the training-only global RMS. Validation and test amplitudes are used only to
validate the existing split contract; they do not enter training, model selection, or metrics.
The split contract fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to
validation.

One seed-42 6-input SIREN has width 256, four sine layers, and `omega_0: 300.0`. It is freshly
initialized and trained with pure MSE, Adam, learning rate `1.0e-3`, and uniform sampling with
replacement from the 271,875 training points. The batch size is 5,000, the update count is
50,000, and the total point-evaluation budget is 250,000,000. Every 500 updates, chunked
prediction in batches of 65,536 over all training points reports training fit on `cuda:0`; the
best point maximizes median training-trace S/N.

## Acceptance and interpretation

- `strong_fit`: best median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, best median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

The corresponding summary decisions are `full_ffid_strong_fit`,
`full_ffid_escaped_zero_predictor`, and `full_ffid_near_zero`.

## Reproduction

```bash
python scripts/run_study_007_full_ffid_large_batch.py \
  --config studies/study_007_full_ffid_large_batch/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_007_full_ffid_large_batch \
  --device cuda:0
```

## Expected outputs

One immutable run directory contains `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
and `run.json`. One sibling summary JSON records the single run and summary decision. No
checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only. It does not evaluate interpolation,
validation/test performance, temporal patches, or alternative optimizers and batches. Because
both batch size and total point budget differ from the earlier 1,024-point experiment, this run
cannot isolate their causal effects and is not a complete reproduction of the paper's 5D setup.

Historical rationale belongs in [`decisions.md`](decisions.md).
