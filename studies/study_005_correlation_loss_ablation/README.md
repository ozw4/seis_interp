# Study 005: correlation-loss ablation

Status: `completed`

## Purpose

- Test whether a trace-wise correlation auxiliary loss is required for the first failing Study 004 subset.

## Conditions

- dataset: seed-42 nested eight-trace FFID 2348 subset, all 625 samples
- normalization: training-only global RMS; validation and test amplitudes unused
- model: six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: Adam, learning rate `1e-3`, full 5,000-point batches, 50,000 updates, `cuda:0`
- conditions: MSE; MSE + `0.1 * mean(1 - trace_correlation)`
- config: [`config.yaml`](config.yaml)

## Results

| Condition | Best median trace S/N | Best median correlation |
|---|---:|---:|
| `mse_control` | 32.48 dB | 0.9997 |
| `mse_corr_0p1` | 33.15 dB | 0.9998 |

- summary decision: `full_batch_control_succeeds`
- MSE control run: `20260826T020640Z_b550db8_mse_control`

## Decision

- The MSE control succeeded, so no causal benefit is assigned to the correlation term.
- The correlation-loss path is closed and no production loss setting is selected here.
