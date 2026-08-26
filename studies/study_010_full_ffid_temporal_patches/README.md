# study_010_full_ffid_temporal_patches

## Status

`active`

## Research question

Can one SIREN fit all 435 FFID 2348 training traces when each update uses a shared 64-sample
temporal patch across 78 randomly selected traces?

## Fixed conditions

The experiment uses every `train` row in the existing FFID 2348 trace split as the sampling
pool and retains all 625 time samples for full-training evaluation. Coordinates use
training-min/max linear scaling plus azimuth sine/cosine features, and amplitudes use the
training-only global RMS. Validation and test amplitudes are used only to validate the existing
split contract; they do not enter training, model selection, or metrics. The split contract
fixes a 0.20 random trace holdout and assigns 0.25 of that holdout to validation.

One seed-42 6-input SIREN has width 256, four sine layers, and `omega_0: 300.0`. It is freshly
initialized and trained with pure point-wise MSE, Adam, and learning rate `1.0e-3`. Every update
uniformly selects 78 distinct training traces without replacement within that update and one
temporal patch start. The same patch is shared by all 78 traces and contributes 64 consecutive
samples from each trace. Trace and patch selection restart on the next update, so both may recur
across updates.

The fixed patch starts are `[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416,
448, 480, 512, 544, 561]`. They use a nominal 50% overlap for 64-sample patches, with the final
start shifted to 561 so the last patch ends at sample 624. The batch size is 4,992
(`78 * 64`), the update count is 50,000, and the total point-evaluation budget is 249,600,000.

Every 500 updates, chunked prediction in batches of 65,536 over all 435 training traces and all
625 samples reports median training-trace S/N, global training S/N, median training-trace
correlation, prediction/target RMS ratio, and mean training loss since the prior report on
`cuda:0`. The best point maximizes median training-trace S/N.

## Acceptance and interpretation

- `strong_fit`: best median training-trace S/N is at least 20 dB.
- `escaped_zero_predictor`: otherwise, best median training-trace S/N exceeds 1 dB and the
  prediction/target RMS ratio at that same report exceeds 0.1.
- `near_zero`: otherwise.

The corresponding summary decisions are `full_ffid_strong_fit`,
`full_ffid_escaped_zero_predictor`, and `full_ffid_near_zero`.

## Reproduction

```bash
python scripts/run_study_010_full_ffid_temporal_patches.py \
  --config studies/study_010_full_ffid_temporal_patches/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_010_full_ffid_temporal_patches \
  --device cuda:0
```

## Expected outputs

One immutable run directory contains `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
and `run.json`. One sibling summary JSON records the single run and summary decision. No
checkpoint, plot, table, or notebook is produced.

## Limitations

This one-seed diagnostic measures training fit only. It does not evaluate interpolation,
validation/test performance, correlation loss, or alternative patch sizes and overlaps. The
condition changes temporal coverage per trace and the number of traces per update relative to
Studies 008 and 009, while also reducing the point budget slightly, so any observed difference
cannot be attributed to one of those factors alone. The fixed patching contract remains a POC
condition rather than a complete reproduction of the paper's 5D experiment.

Historical rationale belongs in [`decisions.md`](decisions.md).
