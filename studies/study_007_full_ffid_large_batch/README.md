# Study 007: full-FFID large random batch

Status: `completed`

## Purpose

- Test whether the Study 006 random-replacement batch fits all 435 FFID 2348 training traces.

## Conditions

- dataset: FFID 2348, 435 training traces x 625 samples
- split / normalization: 20% trace holdout, 25% of holdout assigned to validation; training-only global RMS
- model: seed-42 six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: MSE, Adam, learning rate `1e-3`, random-replacement batch 5,000, 50,000 updates, report every 500, `cuda:0`
- config: [`config.yaml`](config.yaml)

## Results

- classification / decision: `full_ffid_near_zero`
- best step: 19,000
- best median training-trace S/N: -0.01808 dB
- best global S/N: -0.001365 dB
- best median trace correlation: 0.001756
- best prediction/target RMS ratio: 0.02225
- final median training-trace S/N: -0.03997 dB
- run: `20260826T061901Z_b0af699_random5000_trace435`
- summary: `20260826T061901Z_b0af699_summary.json`

## Decision

- The 5,000-point batch did not fit the 435-trace pool under this condition.
- The result is training-fit only and does not isolate batch size from total point budget.
