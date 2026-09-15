# Study 004: domain scaling

Status: `active`

## Purpose

- Measure training fit as the number of FFID 2348 training traces increases.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- all 625 samples per trace; 50,000 updates (500 steps x 100 epochs)
- classification: `strong_fit` at median training-trace S/N >=20 dB; `escaped_zero_predictor` above 1 dB with RMS ratio above 0.1; otherwise `near_zero`

## Results

| Training traces | Classification | Best median training-trace S/N |
|---:|---|---:|
| 1 | `strong_fit` | 37.36 dB |
| 8 | `near_zero` | not recorded here |
| 32 | `near_zero` | not recorded here |
| 128 | `near_zero` | not recorded here |
| 435 | `near_zero` | not recorded here |
