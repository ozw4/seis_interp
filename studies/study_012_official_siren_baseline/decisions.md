# Decisions

This file records historical rationale and run-backed conclusions. The current contract is
defined by `README.md`, `config.yaml`, and `inputs.yaml`; code and tests define implementation
behavior.

## 2026-08-27 — Test the official parameterization as one package

**Context:** Study 007 remained near zero on all 435 training traces, and Study 011's nested
continuation collapsed when the trace pool reached 128. The proposed SIREN parameterization
changes both activation frequencies and the corresponding initialization scales.

**Decision:** Test `omega_0: 30.0`, `hidden_omega: 30.0`, and the associated weight
initialization together as `official_siren_30`, without a decomposition sweep.

**Evidence:** The fixed contract couples each non-first layer's activation frequency to its
weight scale. Separating those settings would answer additional research questions outside this
two-condition diagnostic.

**Impact:** A difference from the control can be attributed only to the parameterization
package, not to any individual frequency or initialization choice.

## 2026-08-27 — Pair the official condition with a direct legacy control

**Context:** Interpreting the official condition requires confirming that the earlier near-zero
behavior is reproduced by the same execution path and training budget.

**Decision:** Run `legacy_control` first with `omega_0: 300.0` and `hidden_omega: 1.0`, followed
by `official_siren_30`; freshly initialize and reseed every model, optimizer, and sampler.

**Evidence:** Study 007 used the legacy 300/1 parameterization with 5,000-point
random-replacement batches for 50,000 updates and classified the result as `near_zero`.

**Impact:** If the paired legacy condition is not `near_zero`, the summary records
`legacy_control_not_reproduced` instead of interpreting the official condition as a valid
comparison.

## 2026-08-27 — Apply hidden-frequency scaling to the final linear

**Context:** The final linear has no sine activation, but it is part of the network to which the
general hidden-layer initialization is applied before the first layer receives its special
override.

**Decision:** Initialize the final linear, like every non-first sine-layer linear, uniformly in
`[-sqrt(6 / fan_in) / hidden_omega, sqrt(6 / fan_in) / hidden_omega]`. This divides its official
condition scale by 30.

**Evidence:** Applying the general sine initialization to the network and then overriding only
the first layer yields this explicit final-linear contract.

**Impact:** The paired conditions differ in final-linear weight scale as part of the complete
official parameterization; the study does not isolate that choice.

## 2026-08-27 — Retain PyTorch's default bias initialization

**Context:** The experiment requires explicit weight initialization but does not prescribe a
separate bias initialization.

**Decision:** Leave biases at their PyTorch defaults in both conditions.

**Evidence:** This is the smallest change from the existing SIREN and avoids introducing an
additional initialization variable not present in the fixed contract.

**Impact:** Any paired difference comes from the specified frequency and weight-initialization
package rather than a new bias rule.

## 2026-08-27 — Exclude held-out amplitudes from evaluation

**Context:** The research question concerns optimization on the full training split, not
interpolation performance.

**Decision:** Use validation and test data only to verify the existing trace-level split
contract. Do not include their amplitudes in training, model selection, or metrics.

**Evidence:** The Study 012 acceptance thresholds classify training fit from all 435 training
traces and their 625 samples.

**Impact:** The result cannot support claims about validation/test interpolation, but directly
answers the scoped training-optimization question without held-out leakage.

## 2026-08-27 — Official SIREN package remains near zero

**Context:** The paired full-training run completed both 50,000-update conditions on all 435
training traces. The legacy control reproduced Study 007's near-zero result, so the comparison
validity gate passed.

**Decision:** Record `official_siren_near_zero`. The official 30/30 frequency and
initialization package did not escape the near-zero predictor under the fixed Study 007
training conditions.

**Evidence:** Runs `20260827T004311Z_fcfeec9_legacy_control` and
`20260827T004311Z_fcfeec9_official_siren_30`; summary
`20260827T004311Z_fcfeec9_summary.json`. The legacy control's best report was step 19,000 with
-0.018082 dB median trace S/N, -0.001365 dB global S/N, 0.001756 median correlation, and a
0.022250 RMS ratio; its final median trace S/N was -0.039974 dB. The official condition's best
report was also step 19,000 with -0.020534 dB median trace S/N, -0.002343 dB global S/N,
0.000955 median correlation, and a 0.023597 RMS ratio; its final median trace S/N was -0.040617
dB. Both conditions were `near_zero`; each completed 250,000,000 point evaluations and 100
finite reports.

**Impact:** Switching to the complete official parameterization package was not sufficient to
fit the full training set in this paired seed-42 run. This result does not identify the effect
of any individual frequency or initialization component and does not establish behavior for
other seeds, batching schemes, or training budgets.
