# Study 008: full-FFID complete-trace batches

Status: `completed`

## Purpose

- Test complete-trace sampling on all 435 FFID 2348 training traces.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- dataset / split / normalization: Study 007 condition
- eight complete traces per update = 5,000 points

## Results

- classification / decision: `full_ffid_near_zero`
- best step 3,500: median training-trace S/N -0.001160 dB, global S/N approximately 0 dB, median trace correlation 0.002370, prediction/target RMS ratio 0.005476
- final median training-trace S/N: -0.04134 dB
- run / summary: `20260826T065417Z_fa548ba_tracebatch8_trace435`, `20260826T065417Z_fa548ba_summary.json`
