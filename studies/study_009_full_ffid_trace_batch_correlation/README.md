# Study 009: complete-trace batches with correlation loss

Status: `completed`

## Purpose

- Test the Study 005 correlation term with Study 008 complete-trace batches on the full FFID 2348 training pool.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- dataset / split / normalization / model: Study 008 condition
- 50,000 updates = 250,000,000 point evaluations

## Results

- classification / decision: `full_ffid_near_zero`
- best step 3,500: median training-trace S/N -0.017734 dB, global S/N -0.001588 dB, median trace correlation 0.000583, prediction/target RMS ratio 0.018018
- final median training-trace S/N: -0.098969 dB
- run / summary: `20260826T071528Z_3013774_tracebatch8_corr0p1_trace435`, `20260826T071528Z_3013774_summary.json`
