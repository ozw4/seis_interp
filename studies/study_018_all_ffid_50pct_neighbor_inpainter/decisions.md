# Decisions

The current contract is defined by `README.md`, `config.yaml`, `inputs.yaml`,
code, tests, and immutable runs. This file records rationale relevant to changing
that contract.

## 2026-08-29 — Interpret 50% as a within-FFID trace ratio

**Status:** active

**Decision:** Assign 50% of amplitude-eligible traces independently within every
FFID to train, using the established seed-42 whole-trace permutation. Keep the
existing 25/75 validation/test allocation inside the holdout.

**Reason:** The predecessor's `split_scope: per_ffid` and the interpolation use
case define train ratio as the observed-trace density within each FFID. This keeps
every eligible FFID represented while changing only the requested train ratio.
Holding out entire FFIDs would instead test shot extrapolation and require a
different research question.

## 2026-08-29 — Make formal scope requirements study-configurable

**Status:** active

**Decision:** Keep strict runtime comparison between observed scope and the
study's declared required counts, but do not hard-code Study 017's counts or
15 dB threshold in the reusable pipeline.

**Reason:** Immutable Study 017 remains locked by its own config and regression
test. Reusable training code must also execute the new 50% split and strict 20 dB
contract without falsely accepting mismatched artifacts.

## 2026-08-29 — Begin with the unchanged Study 017 model

**Status:** active

**Decision:** The first fresh run changes only the split and success threshold.

**Reason:** A direct baseline is required to distinguish degradation caused by
lower observed-trace density from gains due to later model or training changes.
