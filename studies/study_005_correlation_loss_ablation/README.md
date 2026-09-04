# study_005_correlation_loss_ablation

## Status

`completed`

## Research question

Does a trace-wise correlation auxiliary loss help the first failing Study 004 subset—eight
training traces over all 625 samples—escape the near-zero predictor?

## Fixed conditions

Both conditions use the same seed-42 nested eight-trace prefix, the existing split and global-RMS
normalization, and a fresh 6-input SIREN with width 256, four sine layers, `omega_0: 300.0`, Adam,
learning rate `1.0e-3`, and 50,000 full-batch updates on `cuda:0`. Every update uses all 5,000
training points, with reports every 500 updates. No validation or test amplitudes are evaluated.

## Conditions

- `mse_control`: MSE only.
- `mse_corr_0p1`: MSE plus `0.1` times mean trace-wise `1 - correlation`.

## Acceptance and interpretation

A condition escapes the zero predictor only when its best-report median trace S/N exceeds 1 dB,
median trace correlation exceeds 0.1, and prediction/target RMS ratio exceeds 0.1. A successful
control attributes no causal benefit to correlation loss. If only the correlation condition
passes, it is promising; RMS-only growth is classified as amplitude inflation without alignment.
These are POC-specific thresholds, not criteria reported by the source paper.
The trace-correlation auxiliary loss itself is also outside the paper-reported L1, L2, and Huber
losses.

## Reproduction

```bash
python scripts/run_study_005_correlation_loss_ablation.py \
  --config studies/study_005_correlation_loss_ablation/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output-root runs/study_005_correlation_loss_ablation \
  --device cuda:0
```

## Current conclusion

Both `mse_control` and `mse_corr_0p1` completed 50,000 full-batch updates and escaped the
near-zero predictor, giving the summary decision `full_batch_control_succeeds`. Best median trace
S/N and correlation were 32.48 dB / 0.9997 for the control and 33.15 dB / 0.9998 with correlation
loss. Because the MSE control alone succeeded, no causal benefit is attributed to the auxiliary
loss and the correlation-loss path is closed.

Historical rationale belongs in [`decisions.md`](decisions.md).
