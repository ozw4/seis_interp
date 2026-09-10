# Study 003: omega0 sensitivity

Status: `active`

## Purpose

- Measure validation convergence as `model.omega_0` and learning rate vary on FFID 2348.

## Conditions

- dataset / split / normalization: Study 001 FFID 2348 prepared dataset, unchanged
- model: SIREN; control `omega_0=10`, learning rate `1e-4`
- variants: `omega_0` in `{100, 300, 600}` crossed with learning rate in `{3e-4, 1e-3}`
- fixed training: L2, Adam, batch 1,024, 500 steps per epoch, at most 100 epochs, early-stopping patience 100
- selection: highest validation S/N; test is unused
- config: [`config.yaml`](config.yaml) and [`variants/`](variants/)

## Results

- No completed comparison is recorded.

## Decision

- Keep the fixed-budget matrix in this separate optimization study; `omega_0=10` is a slow control, not an unrepresentable condition.
