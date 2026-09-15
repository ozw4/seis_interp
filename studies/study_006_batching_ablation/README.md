# Study 006: batching ablation

Status: `completed`

## Purpose

- Determine whether exact coverage of the eight-trace subset is required when batch size and update count are fixed.

## Conditions

Executable condition: [`config.yaml`](config.yaml).

- 5,000 points x 50,000 updates = 250,000,000 point evaluations

## Results

| Condition | Classification | Best median training-trace S/N | Run |
|---|---|---:|---|
| `exact_full_batch` | `strong_fit` | 32.48 dB | `20260826T045538Z_55b4b9d_exact_full_batch` |
| `random_replacement_5000` | `strong_fit` | 30.14 dB | `20260826T045538Z_55b4b9d_random_replacement_5000` |

- summary: `20260826T045538Z_55b4b9d_summary.json`
