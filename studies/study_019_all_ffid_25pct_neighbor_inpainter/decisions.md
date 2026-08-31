# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-31 — Interpret 25% as a within-FFID trace ratio

**Status:** active

**Decision:** Assign 25% of amplitude-eligible traces independently inside every
FFID using the seed-42 whole-trace permutation. Keep validation at 25% of the
75% holdout and test at the remaining 75% of holdout.

**Reason:** This preserves the interpolation question and the predecessor
studies' split meaning while changing only observed-trace density. Holding out
whole FFIDs would instead test shot extrapolation.

## 2026-08-31 — Start from the accepted Study 018 architecture

**Status:** active

**Decision:** Stage 01 reuses the Study 018 formal architecture and optimization
condition for 2,500 updates, changing only the split and success threshold.

**Reason:** A direct density-controlled baseline is necessary before attributing
later gains to neighborhood, model, objective, or budget changes.
