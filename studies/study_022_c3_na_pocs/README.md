# Study 022: C3 NA POCS

Status: `draft`; smoke complete, formal volume not run

## Purpose

- Evaluate fixed hard-threshold Fourier POCS-5D on the shared validation benchmark.

## Conditions

- partition: source lines train `[0,25)`, validation `[25,35)`, test `[35,50)`; lowest-`array_row` physical-cell canonicalization
- mask: validation `random_trace`, requested missing fraction 0.8, seed 42
- formal volume: time `[0,384)`, source lines `[25,35)`, shots `[27,59)`, receiver x `[0,8)`, receiver y `[18,50)`; 81,920 traces
- smoke volume: `[64,128) x [25,29) x [27,35) x [0,8) x [18,34)`; 4,096 traces
- method: orthonormal time rFFT and four-dimensional spatial FFT; hard threshold; observed samples reinserted each iteration; no amplitude normalization
- formal settings: 100 iterations, geometric threshold ratio 1.0 to 0.01, windows `[128,8,16,8,32]`, overlaps `[64,4,8,4,16]`
- smoke settings: 20 iterations and one block
- metric: physical-amplitude global S/N on `evaluation_target` only
- configs: [`config.yaml`](config.yaml), [`config_smoke.yaml`](config_smoke.yaml)

## Results

- smoke run: `20260907T025323Z_df93b6b_smoke`
- target domain: 3,275 traces / 209,600 samples
- target S/N: 22.8842 dB; RMSE: 0.2875; relative L2: 0.0717
- zero-fill RMSE: 4.0074
- exact observed-sample consistency; zero uncovered samples; no warnings
- commit: `df93b6b7b4c527d687ae9ebfb9556a1b20c1b556`; clean worktree

## Decision

- Treat the smoke as execution-contract evidence only; no formal-volume or test result is recorded.
