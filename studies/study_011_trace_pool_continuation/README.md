# Study 011: trace-pool continuation

Status: `completed`

## Purpose

- Test whether one fitted SIREN remains fitted while its FFID 2348 training pool expands from 8 to 435 traces.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- all 625 samples per trace
- totals across the seven stages: 350,000 updates and 1,750,000,000 point evaluations

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

- 128-trace entry S/N 6.1350 dB; best post-update S/N -0.02026 dB
- final-pool best step cumulative 307,000: median S/N -0.02189 dB, global S/N -0.00235 dB, median correlation -0.00234, RMS ratio 0.02230
- decision: `full_ffid_near_zero`
- run / summary: `20260826T232352Z_1aa9755_continuation8to435_random5000`, `20260826T232352Z_1aa9755_summary.json`
