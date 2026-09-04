# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-26 — Vary only the training-trace count in Experiment A

**Status:** active

**Decision:** Keep the complete 625-sample time domain and vary only the deterministic nested
training-trace counts.

**Reason:** This probe measures training fit alone, so temporal patching and interpolation
evaluation stay outside Experiment A and cannot confound the trace-count axis.

**Impact:** Validation and test amplitudes are unused, and the study reports no interpolation
result.

## 2026-08-26 — The scaling boundary lies between 1 and 8 traces

**Status:** active

**Decision:** Record the empirical boundary without attributing it to parameter count or to an
`omega_0` optimum.

**Reason:** Only the 1-trace condition reached `strong_fit`; 8, 32, 128, and 435 traces all
remained `near_zero` under one fixed setup, which locates the boundary but does not explain it.

**Evidence:** The 1-trace condition reached 37.36 dB best median training-trace S/N.

**Impact:** The first failing subset, eight traces, becomes the probe used by the follow-up loss
and batching studies.
