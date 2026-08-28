# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Isolate exact coverage from batch size

**Status:** active

**Decision:** Compare exact full-batch coverage with random replacement at the same 5,000 points
per update and 50,000 updates on the identical seed-42 eight-trace subset.

**Reason:** Study 005 showed that full-batch MSE fits the subset strongly, but that change also
increased batch size, reduced gradient variance, and guaranteed complete point coverage. Holding
batch size and point budget fixed isolates whether exact coverage is necessary.

**Evidence:** Study 005 run `20260826T020640Z_b550db8_mse_control` reached 32.48 dB best
median training-trace S/N.

**Impact:** Production training remains unchanged while the batching cause is diagnosed.

## 2026-08-26 — Random replacement succeeds at matched batch size

**Status:** active

**Decision:** Record `random_replacement_succeeds`; exact point coverage is not required for the
fixed eight-trace fit when both conditions use 5,000 points per update.

**Reason:** Both conditions reached `strong_fit` with the same 250,000,000 training point
evaluations. This points to the former 1,024-point batch size or its gradient variance, rather
than exact coverage, as the likely batching cause.

**Evidence:** Summary `20260826T045538Z_55b4b9d_summary.json`; best median training-trace S/N was
32.48 dB for `20260826T045538Z_55b4b9d_exact_full_batch` and 30.14 dB for
`20260826T045538Z_55b4b9d_random_replacement_5000`.

**Impact:** The next study candidate is 435-trace training with random-replacement batch size
5,000. No production setting or model artifact is selected here.
