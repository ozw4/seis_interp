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

## 2026-08-27 — Amplitude balancing does not escape the near-zero predictor

**Context:** The paired full-training run completed all three 50,000-update conditions on all
435 training traces. The control reproduced Study 007/012's near-zero result (best report at
step 19,000 with -0.0181 dB median trace S/N and a 0.0222 RMS ratio), so the comparison
validity gate passed. The recorded per-trace scales spanned 0.152 to 15.62 with median 0.264,
confirming the amplitude-concentration analysis on the executed data.

**Decision:** Record `per_trace_rms_near_zero`. Scaling every training trace to unit RMS did
not change the optimization outcome: the per-trace condition's best report was step 4,000 with
-0.0029 dB median trace S/N, 0.0003 median correlation, and a 0.0265 RMS ratio, and its mean
training loss stayed at the unit data variance near 1.0 through all 50,000 updates. The Huber
condition (delta 1.0, unchanged target) was also `near_zero`, best at step 36,500 with
-0.0013 dB and a flat loss near 0.070.

**Evidence:** Runs `20260827T021745Z_9835849_global_rms_control`,
`20260827T021745Z_9835849_per_trace_rms`, and `20260827T021745Z_9835849_huber_global_rms`;
summary `20260827T021745Z_9835849_summary.json`. Each condition completed 250,000,000 point
evaluations and 100 finite reports.

**Impact:** The hypothesis that amplitude imbalance is sufficient to explain the fresh
435-trace failure is refuted under this budget; robust loss at one delta is also insufficient.
The failure mechanism must involve something these conditions share. The strongest remaining
discriminator from Studies 004-006 is per-trace sample density per update (the successful
eight-trace fits saw about 625 points per trace per 5,000-point batch, the failing 435-trace
runs about 11.5), with full-batch training over all 271,875 points as the direct next probe.
This one-seed result does not test whether amplitude imbalance interacts with pool expansion in
the Study 011 continuation collapse.
