# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-27 — Target the amplitude distribution after structural probes failed

**Context:** Studies 007-010 remained near zero under random-point, complete-trace, and shared
temporal-patch batches; Study 011's continuation collapsed when the pool reached 128 traces;
Study 012 showed the official SIREN parameterization changes nothing on this probe. A
2026-08-27 analysis of the prepared FFID 2348 dataset found that the failures line up with
extreme amplitude concentration instead: after global-RMS normalization, per-trace RMS spans
0.15-15.6, the top ten traces hold 79% of training energy, and the top 1% of points hold 94%.
The highest-RMS traces are the near-offset traces (20-82 m; corr(log RMS, log offset) = -0.946),
with 90% of the strongest trace's energy in its first 16 samples. The Study 011 nested pools
kept maximum trace RMS at or below 1.12 through 64 traces (all strong fits); the 128-trace stage
introduced an RMS-5.5 trace holding 48% of pool energy, exactly where continuation collapsed. In
Studies 007/012, mean training loss never left the data variance near 1.0, and the legacy and
official loss histories matched to about 1e-4, meaning the model output contributed almost
nothing to the loss.

**Decision:** Run a paired diagnostic under the unchanged Study 007/012 training budget:
`global_rms_control` (exact re-execution gate), `per_trace_rms` (each training trace scaled to
unit RMS), and `huber_global_rms` (unchanged target, Huber delta 1.0).

**Evidence:** Study 007 run `20260826T061901Z_b0af699_random5000_trace435`, Study 011 run
`20260826T232352Z_1aa9755_continuation8to435_random5000`, Study 012 summary
`20260827T004311Z_fcfeec9_summary.json`, and the amplitude statistics computed from
`data/interim/c3_na/ffid_2348` with the recorded split and normalization.

**Impact:** One run either confirms amplitude imbalance as the operative failure condition or
rejects this direction. Production training is unchanged.

## 2026-08-27 — Keep the legacy SIREN parameterization for all conditions

**Status:** active

**Decision:** Use `omega_0: 300.0` and `hidden_omega: 1.0` in every condition.

**Reason:** Study 012's paired run showed the legacy and official packages behave identically on
this probe, and the legacy parameterization is the lineage in which 8-64-trace pools reached
strong fits, so a per-trace success is directly comparable to those anchors.

## 2026-08-27 — Classify the per-trace condition with the unchanged thresholds

**Status:** active

**Decision:** Evaluate each condition against its own training target and keep the fixed
20 dB / 1 dB / 0.1 classification thresholds.

**Reason:** Median trace S/N and median trace correlation are invariant to scaling one trace's
target and prediction by the same positive factor, so per-trace scaling preserves the meaning of
the classification metric. Global S/N and the RMS ratio reweight traces and are recorded as
diagnostics only.

## 2026-08-27 — Record Huber as a secondary observation outside the decision

**Status:** active

**Decision:** Run `huber_global_rms` with PyTorch's default delta 1.0, record its
classification, and exclude it from the summary decision.

**Reason:** Huber is one of the paper-reported losses and bounds extreme-point gradients without
changing the target, so it separates "robust loss alone" from "balanced target" at one fixed
delta. Making it decision-relevant would require a delta sweep outside this study's scope.
