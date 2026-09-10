# Study 023: C3 NA damped rank reduction

Status: `draft`; smoke complete, formal volume not run

## Purpose

- Evaluate reconstruction-only damped rank-reduction 5D on the same benchmark volumes as Study 022.

## Conditions

- partition, case, mask, formal/smoke volumes, and target metric: Study 022
- method: reconstruction-only hard observed-data consistency; orthonormal time rFFT padded to the next power of two; exact NumPy SVD of four-dimensional spatial Hankel matrices
- formal settings: rank 4, damping power 3, 10 iterations, 5 Hz through Nyquist, windows `[5,8,4,8]`
- smoke settings: rank 4, damping power 3, 3 iterations, windows `[4,4,4,8]`
- normalization: none
- configs: [`config.yaml`](config.yaml), [`config_smoke.yaml`](config_smoke.yaml)

## Results

- smoke run: `20260907T035913Z_ca7bc8b_smoke`
- target domain: 3,275 traces / 209,600 samples
- target S/N: 2.9109 dB; RMSE: 2.8662; relative L2: 0.7152
- zero-fill: 0.0000 dB / RMSE 4.0074
- prediction: float32 shape `[64,4,8,8,16]`; eight nonempty blocks, Hankel shape `[135,32]`; 30 FFT bins from 5.8594 to 62.5000 Hz
- timing: load/hash 3.9607 s; reconstruction 0.5783 s; evaluation 0.0051 s; maximum RSS 256.8164 MiB
- observed maximum error 0; no uncovered traces or warnings
- commit: `ca7bc8bd7bcac694703093376fe274b23179b0af`; clean worktree

## Decision

- Treat the smoke as execution-contract evidence only; no formal-volume, tuned validation, method-ranking, or test result is recorded.
