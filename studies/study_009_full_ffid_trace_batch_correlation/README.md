# Study 009: complete-trace batches with correlation loss

Status: `completed`

## Purpose

- Test the Study 005 correlation term with Study 008 complete-trace batches on the full FFID 2348 training pool.

## Conditions

- dataset / split / normalization / model: Study 008 condition
- training: Adam, learning rate `1e-3`, eight complete traces per update, 50,000 updates, 250,000,000 point evaluations
- loss: MSE + `0.1 * mean(1 - trace_correlation)`, epsilon `1e-4`
- config: [`config.yaml`](config.yaml)

## Results

- classification / decision: `full_ffid_near_zero`
- best step: 3,500
- best median training-trace S/N: -0.017734 dB
- best global S/N: -0.001588 dB
- best median trace correlation: 0.000583
- best prediction/target RMS ratio: 0.018018
- final median training-trace S/N: -0.098969 dB
- run: `20260826T071528Z_3013774_tracebatch8_corr0p1_trace435`
- summary: `20260826T071528Z_3013774_summary.json`

## Decision

- The correlation term did not make this full-pool condition escape the near-zero predictor.
