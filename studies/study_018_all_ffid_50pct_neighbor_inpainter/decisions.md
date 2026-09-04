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

## 2026-08-29 — Freeze the positive model changes for the formal run

**Status:** active

**Decision:** Use the Stage 15 model condition for the fresh formal run: K274 on
the same source-x line, four target coordinates, width 384, full-trace receptive
field, FiLM, target-coordinate masked-softmax neighbor gating, and an
identity-initialized 31-tap depthwise neighbor-alignment FIR. Train for 50,000
updates with the established replacement sampler.

**Reason:** At 2,500 updates, width 384 improved the width-256 condition by
0.2357 dB, the gate by 0.0507 dB, and the lightweight FIR
by 0.1309 dB. Their combined condition preserved the gains and reached
16.8037 dB, 0.4371 dB above Stage 07. In contrast, the
parameter-heavy kernel-31 stem lost 0.2003 dB, crossline neighbors lost
0.2458 dB, and epoch-without-replacement sampling changed the 10,000-
step result by only -0.0019 dB. Training budget was the largest
positive factor: extending the width-256 condition from 2,500 to 10,000 updates
added 1.8819 dB and was still improving at the final checkpoint.

## 2026-08-29 — Accept the 50,000-update formal result

**Status:** accepted

**Decision:** Accept run
`20260829T075432Z_ee3d9e5_formal_50000_steps` as satisfying this study's formal
success contract. Its best and final checkpoint reached
`oracle_per_trace_unit_rms_global_snr_db = 20.4604`, strictly above
20 dB, with `metric_success`, `scope_success`, and overall `success` all true.

**Reason:** The saved checkpoint reproduced the selected raw validation metric
exactly within the declared `1e-8` tolerance. All formal scope checks passed for
4,780 eligible FFIDs, 625 samples, effective train/validation/test counts of
1,151,731 / 287,933 / 863,801, and fully excluded FFID 1746. The audits found no
remaining duplicate physical cells, train coordinate collisions,
train-validation coordinate overlaps, or center neighbor offsets; test and
excluded amplitude rows were not materialized. This acceptance applies to the
predeclared oracle validation metric, not to unseen test performance or physical
amplitude recovery.
