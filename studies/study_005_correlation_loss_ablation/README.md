# Study 005: correlation-loss ablation

Status: `completed` — correlation-loss path closed

## Purpose

- Test whether a trace-wise correlation auxiliary loss is required for the first failing Study 004 subset.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- all 625 samples per trace; full 5,000-point batches (8 traces x 625); 50,000 updates

## Results

| Condition | Best median trace S/N | Best median correlation |
|---|---:|---:|
| `mse_control` | 32.48 dB | 0.9997 |
| `mse_corr_0p1` | 33.15 dB | 0.9998 |

- summary decision: `full_batch_control_succeeds`
- MSE control run: `20260826T020640Z_b550db8_mse_control`
