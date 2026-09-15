# Study 014: full-trace-batch ablation

Status: `completed`

## Purpose

- Isolate complete full-pool batches, correlation loss, and per-trace RMS balancing after an informal combined condition escaped the near-zero predictor.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- all 435 training traces x 625 samples; the full batch is 271,875 points

## Results

Run family: `20260827T041050Z_925e8e4_*`; decision: `full_trace_batch_escaped_zero_predictor`.

| Condition | Classification | Best step | Median S/N | Median correlation | RMS ratio |
|---|---|---:|---:|---:|---:|
| `small_batch_control` | `near_zero` | 14,000 | 0.0039 dB | 0.0395 | 0.0177 |
| `full_trace_batch` | `escaped_zero_predictor` | 45,500 | 8.91 dB | 0.9349 | 0.9226 |
| `full_trace_batch_correlation` | `escaped_zero_predictor` | 2,000 | 8.19 dB | 0.9272 | 0.6074 |
| `full_trace_batch_per_trace_rms` | `escaped_zero_predictor` | 44,500 | 16.14 dB | 0.9878 | 0.9857 |
| `full_trace_batch_correlation_per_trace_rms` | `escaped_zero_predictor` | 47,500 | 16.39 dB | 0.9885 | 0.9874 |

- combined condition reproduced the informal result at 16.3882 dB; no condition reached the 20 dB `strong_fit` threshold within 50,000 updates

## Decision

- Retained recipe: full complete-trace batches plus per-trace RMS balancing, without correlation loss.
