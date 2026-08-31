# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-31 — Interpret 75% as a within-FFID trace ratio

**Status:** active

**Decision:** Assign 75% of amplitude-eligible traces independently within every
FFID, using the established seed-42 whole-trace permutation. Keep 25% of the
holdout for validation and 75% for test.

**Reason:** The interpolation question concerns observed-trace density within
each FFID. Holding out complete FFIDs would instead measure shot extrapolation.
This interpretation changes only the requested train density from Study 018 and
keeps every eligible FFID represented in all three splits.

## 2026-08-31 — Start from the accepted Study 018 architecture

**Status:** active

**Decision:** The first diagnostic changes only the per-FFID split density and
the strict success threshold. It otherwise uses Study 018's accepted K274,
width-384, FiLM, coordinate-gated, FIR-aligned architecture and training recipe.

**Reason:** A direct transfer is required to distinguish the effect of 75% train
density from any later architecture or optimization improvement.

## 2026-08-31 — Require a frozen fresh full-scope acceptance run

**Status:** active

**Decision:** Staged validation results guide the investigation, but acceptance
requires a newly started full-scope run of the final frozen condition. The saved
checkpoint must reproduce a raw oracle validation metric strictly above 25 dB,
and every formal leakage and scope check must pass.

**Reason:** Short diagnostics and learning-curve extrapolation are useful for
selection but are not sufficient evidence of the requested threshold.
