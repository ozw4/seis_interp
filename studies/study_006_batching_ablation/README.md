# study_006_batching_ablation

## Status

`completed`

## Research question

Is a 5,000-point random-replacement batch sufficient to fit the fixed eight-trace subset, or is
exact coverage of all 5,000 trace-time points required on every update?

## Fixed conditions

Both conditions use the same seed-42 nested eight-trace subset from the existing FFID 2348
training split, all 625 time samples, and training-only global-RMS normalization. Each condition
starts from an identically initialized 6-input SIREN with width 256, four sine layers,
`omega_0: 300.0`, Adam, L2 loss, learning rate `1.0e-3`, and 50,000 updates on `cuda:0`.
Each update evaluates 5,000 training points, and full-subset training fit is reported every 500
updates. Validation and test amplitudes are not used.

## Compared conditions

- `exact_full_batch`: every update uses each of the 5,000 points exactly once.
- `random_replacement_5000`: every update draws 5,000 points uniformly with replacement using
  `RandomPointSampler`.

## Acceptance and interpretation

A condition is `strong_fit` when its best median training-trace S/N is at least 20 dB. Otherwise
it is `escaped_zero_predictor` when that S/N exceeds 1 dB and its prediction/target RMS ratio at
the same report exceeds 0.1; all other outcomes are `near_zero`.

The summary decision distinguishes whether random replacement succeeds, partially succeeds, or
leaves exact coverage necessary. If the exact control does not reach `strong_fit`, the result is
`control_failed_unexpected`. This diagnostic does not select a production model or batching
strategy.

## Current conclusion

Random replacement at 5,000 points per update also achieved a strong fit, so exact point coverage
is not required for this eight-trace setup. The earlier 1,024-point batch was likely too small or
had excessive gradient variance under the fixed training conditions.

## Reproduction

```bash
python scripts/run_study_006_batching_ablation.py \
  --config studies/study_006_batching_ablation/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_006_batching_ablation \
  --device cuda:0
```

## Expected outputs

Two immutable condition directories contain resolved configuration, locked inputs, metrics, and
run metadata. One immutable summary JSON records both best points, classifications, and the
batching decision. No checkpoints or plots are produced.

## Limitations

The comparison covers one seed, one eight-trace subset, and training fit only. It does not test
435-trace scaling, interpolation quality, validation/test performance, or other optimizers and
batch sizes.

Historical rationale and completed-run conclusions belong in [`decisions.md`](decisions.md).
