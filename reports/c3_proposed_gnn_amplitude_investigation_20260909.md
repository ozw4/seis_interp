# C3 proposed GNN amplitude diagnostics — 2026-09-09

## Conditions

- fixed four-query CPU training diagnostic, seed 20260908, width 32, two rounds, two neighbors per relation, AdamW `1e-3`, 1,000 updates
- `observed_trace_rms`: normalize visible traces by their RMS and reconstruct query gain from deduplicated direct observed senders using D0 inverse-distance-squared weighting

## Results

| Amplitude mode | Four-query S/N | RMSE |
|---|---:|---:|
| train-global RMS | 20.0908 dB | 1.1356 |
| observed-trace RMS | 21.1003 dB | 1.0110 |

| Input multiplier | Global-RMS output multiplier | Observed-RMS output multiplier | Width-64 step-1000 global-RMS multiplier |
|---:|---:|---:|---:|
| 0.1 | 0.4902 | 0.1000 | 0.2452 |
| 0.3 | 0.6357 | 0.3000 | 0.3724 |
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.1507 | 3.0000 | 2.6278 |
| 10 | 1.1698 | 10.0000 | 6.2350 |

- observed-RMS maximum proportionality relative L2: `5.9814e-7`
- all-target gain-only diagnostic: RMS RMSE 0.3605, relative L2 0.0363, absolute relative error median/p95/p99 1.6007%/8.9315%/15.0471%, zero missing gain contexts
- oracle unit-waveform x estimated-gain score: 28.8088 dB; this uses target waveform shape and is not a model score

## Decision

- Carry observed-trace RMS into a full model condition; do not treat the gain-only oracle as model performance.
