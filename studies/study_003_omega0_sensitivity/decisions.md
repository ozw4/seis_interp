# Decisions

This file records rationale that may matter when changing study conditions.
The current study contract is defined by `README.md`, `config.yaml`, and
`inputs.yaml`. Current code and tests define implementation behavior.

## 2026-08-25 — Separate optimization study

**Status:** active

**Decision:**
Track `omega_0` and learning-rate sensitivity in a separate study rather than changing the baseline study in place.

**Reason:**
The interpolation question remains unchanged, while this study changes optimization conditions and has independent runs.

## 2026-08-25 — Fixed-budget sensitivity matrix

**Status:** active

**Decision:**
Compare the declared `omega_0` / learning-rate matrix with L2 and a fixed training budget; treat `omega_0 = 10` as a slow control rather than an unrepresentable condition.

**Reason:**
Full-FFID L1 and L2 runs remained near the zero predictor, while one-trace diagnostics showed that larger `omega_0` values accelerated short-budget convergence and `omega_0 = 10` remained representationally capable with a longer budget.
