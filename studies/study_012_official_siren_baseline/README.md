# Study 012: official SIREN parameterization

Status: `completed`

## Purpose

- Compare the official 30/30 frequency and initialization package with the legacy 300/1 package on all 435 FFID 2348 training traces.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- dataset / split / normalization: Study 007 condition

## Results

| Condition | Classification | Best step | Median S/N | Global S/N | Median correlation | RMS ratio | Run |
|---|---|---:|---:|---:|---:|---:|---|
| `legacy_control` | `near_zero` | 19,000 | -0.018082 dB | -0.001365 dB | 0.001756 | 0.022250 | `20260827T004311Z_fcfeec9_legacy_control` |
| `official_siren_30` | `near_zero` | 19,000 | -0.020534 dB | -0.002343 dB | 0.000955 | 0.023597 | `20260827T004311Z_fcfeec9_official_siren_30` |

- final median S/N: -0.039974 dB (legacy), -0.040617 dB (official)
- summary / decision: `20260827T004311Z_fcfeec9_summary.json`; `official_siren_near_zero`
