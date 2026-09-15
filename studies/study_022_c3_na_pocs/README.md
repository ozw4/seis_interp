# Study 022: C3 NA POCS

Status: `draft` — smoke complete (execution-contract evidence only), formal volume not run

## Purpose

- Evaluate fixed hard-threshold Fourier POCS-5D on the shared validation benchmark.

## Conditions

Executable condition: [`config.yaml`](config.yaml), [`config_smoke.yaml`](config_smoke.yaml).

- formal volume 81,920 traces; smoke volume 4,096 traces
- method: orthonormal time rFFT and four-dimensional spatial FFT; hard threshold; observed samples reinserted each iteration; no amplitude normalization

## Results

- smoke run: `20260907T025323Z_df93b6b_smoke`; target domain 3,275 traces / 209,600 samples
- target S/N 22.8842 dB; RMSE 0.2875; relative L2 0.0717; zero-fill RMSE 4.0074
- exact observed-sample consistency; zero uncovered samples; no warnings
- commit `df93b6b7b4c527d687ae9ebfb9556a1b20c1b556`; clean worktree
