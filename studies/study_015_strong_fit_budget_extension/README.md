# Study 015: strong-fit budget extension

Status: `completed`

## Purpose

- Determine whether the retained Study 014 recipe reaches the 20 dB `strong_fit` threshold with a longer budget.

## Conditions

- dataset / split / model: Study 014 `full_trace_batch_per_trace_rms`
- batch: all 435 complete FFID 2348 training traces (271,875 points)
- normalization / loss: training-only global RMS followed by per-trace unit RMS; L2 without correlation
- training: seed 42, Adam, learning rate `1e-4`, 400,000 updates, report every 500
- reproduction gate: first-50,000-update best must match 16.1377 dB within 0.05 dB
- config: [`config.yaml`](config.yaml)

## Results

- decision: `extended_budget_strong_fit`
- reproduction gate: passed exactly at full float precision; 16.1377 dB when rounded
- first `strong_fit`: step 199,000, 20.03 dB median trace S/N, 0.9950 median correlation
- best: step 356,500, 21.29 dB median trace S/N, 20.95 dB global S/N, 0.9963 median correlation, 0.9934 RMS ratio
- final step 400,000: 19.40 dB median trace S/N; mean training loss 0.0184
- run: `20260827T054748Z_94b479e_full_trace_batch_per_trace_rms`

## Decision

- The Study 014 recipe reaches `strong_fit` under the 400,000-update budget.
- This is a training-fit result; validation and test interpolation were not evaluated.
