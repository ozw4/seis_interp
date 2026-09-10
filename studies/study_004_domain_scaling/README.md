# Study 004: domain scaling

Status: `active`

## Purpose

- Measure training fit as the number of FFID 2348 training traces increases.

## Conditions

- dataset: FFID 2348, nested seed-42 prefixes of `1, 8, 32, 128, 435` training traces, all 625 samples
- split / normalization: existing trace split and training-only global RMS; validation and test amplitudes unused
- model: six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: Adam, L2, learning rate `1e-3`, batch 1,024, 50,000 updates, report every 500 updates, `cuda:0`
- classification: `strong_fit` at median training-trace S/N >=20 dB; otherwise `escaped_zero_predictor` above 1 dB with RMS ratio above 0.1; otherwise `near_zero`
- config: [`config.yaml`](config.yaml)

## Results

| Training traces | Classification | Best median training-trace S/N |
|---:|---|---:|
| 1 | `strong_fit` | 37.36 dB |
| 8 | `near_zero` | not recorded here |
| 32 | `near_zero` | not recorded here |
| 128 | `near_zero` | not recorded here |
| 435 | `near_zero` | not recorded here |

## Decision

- The observed scaling boundary under this condition is between 1 and 8 traces.
- This study reports training fit only; it does not select a production model or report interpolation performance.
