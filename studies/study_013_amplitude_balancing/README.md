# Study 013: amplitude balancing

Status: `completed`

## Purpose

- Test per-trace RMS balancing and Huber loss on the full 435-trace FFID 2348 training pool.

## Conditions

- dataset / split: Study 007 condition, 435 traces x 625 samples; held-out amplitudes unused
- model: legacy six-input SIREN, width 256, four sine layers, `omega_0=300`, `hidden_omega=1`
- training: seed 42, Adam, learning rate `1e-3`, random-replacement batch 5,000, 50,000 updates
- conditions: global-RMS MSE control; per-trace-unit-RMS MSE; global-RMS Huber with delta 1.0
- config: [`config.yaml`](config.yaml)

## Results

| Condition | Classification | Best step | Median S/N | Median correlation | RMS ratio |
|---|---|---:|---:|---:|---:|
| `global_rms_control` | `near_zero` | 19,000 | -0.0181 dB | 0.0018 | 0.0222 |
| `per_trace_rms` | `near_zero` | 4,000 | -0.0029 dB | 0.0003 | 0.0265 |
| `huber_global_rms` | `near_zero` | 36,500 | -0.0013 dB | 0.0075 | 0.0096 |

- recorded per-trace RMS range: 0.152 to 15.62; median 0.264
- each condition completed 250,000,000 point evaluations and 100 finite reports
- runs: `20260827T021745Z_9835849_global_rms_control`, `20260827T021745Z_9835849_per_trace_rms`, `20260827T021745Z_9835849_huber_global_rms`
- summary / decision: `20260827T021745Z_9835849_summary.json`; `per_trace_rms_near_zero`

## Decision

- Per-trace RMS balancing alone and Huber loss at delta 1.0 did not escape the near-zero predictor.
