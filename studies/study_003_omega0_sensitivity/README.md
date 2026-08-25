# study_003_omega0_sensitivity

## Status

`active`

## Research question

With the model, split, normalization, loss, batch size, and training budget held fixed, how do `model.omega_0` and `training.learning_rate` affect validation convergence for FFID 2348 of SEG C3 Narrow-Azimuth?

## Hypothesis

An `omega_0` larger than 10 may accelerate early learning of the high frequencies of the full 625-sample time axis, because the first sine layer scales its initialized weights by `omega_0`. An excessive `omega_0` or learning rate may instead make convergence unstable. The winning condition is chosen by the best validation S/N, not by the smallest training loss.

## Inputs

`inputs.yaml` locks the same formal input as `study_001_c3_na_baseline`: FFID 2348 of `SEG_C3NA_ffid_1201-2400.sgy`, 544 traces of 625 samples at 8 ms. The existing prepared datasets are reused:

```text
data/interim/c3_na/ffid_2348
data/processed/c3_na/ffid_2348_random_split
```

The split and normalization conditions are unchanged, so the processed dataset is not regenerated. Training still verifies that the resolved configuration matches the recorded preparation contract.

## Method

Seven conditions are compared: the control in `config.yaml` and the six configurations in `variants/`.

| Condition | Config | `omega_0` | `learning_rate` |
|---|---|---|---|
| control | `config.yaml` | 10.0 | 1.0e-4 |
| variant | `variants/omega_100_lr_3e-4.yaml` | 100.0 | 3.0e-4 |
| variant | `variants/omega_100_lr_1e-3.yaml` | 100.0 | 1.0e-3 |
| variant | `variants/omega_300_lr_3e-4.yaml` | 300.0 | 3.0e-4 |
| variant | `variants/omega_300_lr_1e-3.yaml` | 300.0 | 1.0e-3 |
| variant | `variants/omega_600_lr_3e-4.yaml` | 600.0 | 3.0e-4 |
| variant | `variants/omega_600_lr_1e-3.yaml` | 600.0 | 1.0e-3 |

Every condition keeps the L2 loss, Adam, batch size 1024, 500 steps per epoch, and 100 maximum epochs. `early_stopping_patience` is 100, so a condition is not stopped before it can leave the near-zero plateau. Saving only the best checkpoint is unchanged. The test split is not used; selection uses validation S/N alone. Source code, model architecture, sampler, and checkpoint format are not changed by this study.

## Reproduction

Each run uses the existing CLI directly; this study adds no sweep runner. A run directory is rejected if it already exists, so every execution needs a new run ID.

Control:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_omega10_lr1e4"

python -m seis_interp.cli train siren \
  --config studies/study_003_omega0_sensitivity/config.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output "runs/study_003_omega0_sensitivity/$RUN_ID" \
  --device cuda:0
```

Variant:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_omega300_lr1e3"

python -m seis_interp.cli train siren \
  --config studies/study_003_omega0_sensitivity/variants/omega_300_lr_1e-3.yaml \
  --interim data/interim/c3_na/ffid_2348 \
  --processed data/processed/c3_na/ffid_2348_random_split \
  --output "runs/study_003_omega0_sensitivity/$RUN_ID" \
  --device cuda:0
```

## Acceptance criteria

- The control and the six variants each finish successfully in their own immutable run directory.
- Each run records its effective `omega_0`, learning rate, and device in `config.resolved.yaml`.
- Each run locks the same interim and processed inputs in `inputs.lock.json`.
- Either at least one condition clearly exceeds 0 dB best validation S/N, or all conditions reproducibly stay near 0 dB.
- The adopted condition is the one with the highest best validation S/N. If the differences are marginal, no optimum is claimed.

This study compares sensitivity; it does not fix a new success threshold such as 1 dB.

## Limitations

The comparison is limited to a single shot of narrow-azimuth, clean synthetic data. `omega_0` and the learning rate vary together, so this is not a design that isolates their main effects. Temporal patching, learning-rate schedules, weighted sampling, and capacity changes are out of scope. Because validation selects the condition, the final performance of the adopted condition must be evaluated against the test split in a separate study.

## Decision log

- 2026-08-25: Facts. The first full-FFID run with L1 at `learning_rate: 1.0e-5` (`runs/study_001_c3_na_baseline/20260825T053620Z_09d01f2_baseline`) stopped early after 54 epochs with a best validation S/N of -0.0044 dB. The full-FFID run with L2 at `1.0e-4` (`runs/study_001_c3_na_baseline/20260825T070226Z_73694b8_l2_baseline`) stopped early after 31 epochs with a best validation S/N of -0.00061 dB and a final training loss of 0.986 on RMS-normalized amplitudes. Interpretation. Changing the loss from L1 to L2 did not resolve the near-zero convergence, so the loss was not the limiting condition.
- 2026-08-25: Facts. In one-trace diagnostics outside `runs/`, a larger `omega_0` converged faster than `omega_0: 10.0` under a short fixed step budget, while `omega_0: 10.0` still fit one trace at high S/N when the budget was long enough. Interpretation. `omega_0: 10.0` is treated as a slow condition rather than an unrepresentable one, so this study compares `omega_0`, learning rate, and a fixed training budget instead of declaring the current value invalid.
- 2026-08-25: This study is separate from `study_001_c3_na_baseline` because it changes training conditions rather than the interpolation question, and its runs live under `runs/study_003_omega0_sensitivity/`. Existing runs and the study_001 configuration are left unchanged as immutable execution history.
