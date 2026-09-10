# C3 NeRSI implementation audit — 2026-09-10

## Scope

- repository NeRSI reimplementation, verified-volume pipeline and CLI
- Study 034 fixed validation plan, checkpoint restoration, saved-prediction rescoring, optional result collection
- suite SHA-256 `f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`
- case `c3_benchmark_validation_random_trace_80_seed142`; 14,729 observed traces and 58,999 targets x 384 samples

## Verified contracts

- Training receives only the zero-filled observed volume.
- Global RMS is accumulated in float64 from observed samples only; observed zeros remain included.
- Loss selects observed samples before subtraction, so masked NaNs do not enter the loss.
- Adam performs exactly the configured updates; there is no evaluation-driven stopping, retry, target statistic, or best-checkpoint selection.
- Full physical prediction and hard observed reinsertion finish before evaluation.
- The checkpoint binds case, volume, hashes, constructor, preprocessing, and final state; it contains no target waveform or evaluation result.
- CPU restoration validates bindings and matches saved predictions at `rtol=1e-6, atol=1e-6`.
- Independent scoring checks shape, dtype, finiteness, observed consistency, target count, S/N, RMSE, and energies.
- Test execution is disabled.

## Checks recorded at audit time

- focused pytest: 321 passed in 61.79 s
- `ruff check .`: passed
- `ruff format --check .`: 514 files already formatted
- `git diff --check`: passed

The subsequent real-data candidate results are recorded in
[Study 034](../studies/study_034_c3_nersi_baseline/README.md).
