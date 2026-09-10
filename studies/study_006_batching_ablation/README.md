# Study 006: batching ablation

Status: `completed`

## Purpose

- Determine whether exact coverage of the eight-trace subset is required when batch size and update count are fixed.

## Conditions

- dataset: Study 005 seed-42 eight-trace subset, 5,000 points
- normalization: training-only global RMS; validation and test amplitudes unused
- model: six-input SIREN, width 256, four sine layers, `omega_0=300`
- training: Adam, L2, learning rate `1e-3`, 50,000 updates, 250,000,000 point evaluations, `cuda:0`
- conditions: exact full batch; 5,000 uniform samples with replacement
- config: [`config.yaml`](config.yaml)

## Results

| Condition | Classification | Best median training-trace S/N | Run |
|---|---|---:|---|
| `exact_full_batch` | `strong_fit` | 32.48 dB | `20260826T045538Z_55b4b9d_exact_full_batch` |
| `random_replacement_5000` | `strong_fit` | 30.14 dB | `20260826T045538Z_55b4b9d_random_replacement_5000` |

- summary: `20260826T045538Z_55b4b9d_summary.json`

## Decision

- Exact point coverage is not required for this eight-trace condition when both batches contain 5,000 points.
- No production batching strategy is selected by this training-fit-only study.
