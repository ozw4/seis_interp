# Study 007: full-FFID large random batch

Status: `completed`

## Purpose

- Test whether the Study 006 random-replacement batch fits all 435 FFID 2348 training traces.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- 435 training traces x 625 samples

## Results

- classification / decision: `full_ffid_near_zero`
- best step 19,000: median training-trace S/N -0.01808 dB, global S/N -0.001365 dB, median trace correlation 0.001756, prediction/target RMS ratio 0.02225
- final median training-trace S/N: -0.03997 dB
- run / summary: `20260826T061901Z_b0af699_random5000_trace435`, `20260826T061901Z_b0af699_summary.json`
