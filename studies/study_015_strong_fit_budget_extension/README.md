# Study 015: strong-fit budget extension

Status: `completed`

## Purpose

- Determine whether the retained Study 014 recipe reaches the 20 dB `strong_fit` threshold with a longer budget.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- dataset / split / model: Study 014 `full_trace_batch_per_trace_rms`

## Results

- decision: `extended_budget_strong_fit`; reproduction gate passed exactly at full float precision (16.1377 dB rounded)
- first `strong_fit`: step 199,000, 20.03 dB median trace S/N, 0.9950 median correlation
- best: step 356,500, 21.29 dB median trace S/N, 20.95 dB global S/N, 0.9963 median correlation, 0.9934 RMS ratio
- final step 400,000: 19.40 dB median trace S/N; mean training loss 0.0184
- run: `20260827T054748Z_94b479e_full_trace_batch_per_trace_rms`
