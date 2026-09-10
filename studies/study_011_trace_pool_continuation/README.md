# Study 011: trace-pool continuation

Status: `completed`

## Purpose

- Test whether one fitted SIREN remains fitted while its FFID 2348 training pool expands from 8 to 435 traces.

## Conditions

- dataset: seed-42 nested pools `[8, 16, 32, 64, 128, 256, 435]`, all 625 samples
- normalization: training-only global RMS; validation and test amplitudes unused
- model: one seed-42 six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: L2, Adam, learning rate `1e-3`, 5,000 random-replacement points, 50,000 updates per stage
- continuation: model and Adam state carried forward; sampler seeds `[42,43,44,45,46,47,48]`
- total: 350,000 updates and 1,750,000,000 point evaluations
- config: [`config.yaml`](config.yaml)

## Results

| Pool | Final median training-trace S/N | Classification |
|---:|---:|---|
| 8 | 26.8248 dB | `strong_fit` |
| 16 | 30.0633 dB | `strong_fit` |
| 32 | 21.1843 dB | `strong_fit` |
| 64 | 20.9149 dB | `strong_fit` |
| 128 | -0.03638 dB | `near_zero` |
| 256 | not recorded here | `near_zero` |
| 435 | -0.06302 dB | `near_zero` |

- 128-trace entry S/N: 6.1350 dB; best post-update S/N: -0.02026 dB
- final-pool best step: cumulative 307,000; median S/N -0.02189 dB; global S/N -0.00235 dB; median correlation -0.00234; RMS ratio 0.02230
- decision: `full_ffid_near_zero`
- run: `20260826T232352Z_1aa9755_continuation8to435_random5000`
- summary: `20260826T232352Z_1aa9755_summary.json`

## Decision

- This continuation retained `strong_fit` through 64 traces and lost it at the 128-trace stage.
- It did not fit the final 435-trace pool.
