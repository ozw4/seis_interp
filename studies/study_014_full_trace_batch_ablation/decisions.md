# Decisions

## 2026-08-27: Ablate the informal escape instead of adopting it wholesale

An informal scratch run (uncommitted `study_temp` harness, same data preparation, training
engine, and classification as the numbered studies) produced the first 435-trace escape:
`random_complete_traces` with all 435 traces per update, correlation weight 0.3, per-trace RMS
balancing, `omega_0: 30.0`, `hidden_omega: 30.0`, learning rate `1.0e-4`, 50,000 updates →
best median trace S/N 16.39 dB at step 47,500, median trace correlation 0.985, RMS ratio
0.984, still improving at the end. Because that run changed three ingredients at once relative
to every failing baseline, it cannot attribute the escape. This study fixes the model,
learning rate, seed, and budget at the escape run's values and varies only the three
ingredients.

## 2026-08-27: Condition set is a gated 2x2 plus control, not a full factorial

A full factorial over batch structure, correlation, and amplitude scaling would be 8
conditions, but the small-batch cells are already known: Study 013 showed per-trace RMS alone
under 5,000-point random-replacement batches stays near zero, Study 009 showed a correlation
loss on small trace batches stays near zero, and an informal run of this study's exact model
and learning rate under 5,000-point random-replacement batches stayed near zero (best
+0.0039 dB at step 14,000). Therefore the study re-runs only the small-batch control as a gate
and covers the 2x2 over correlation and amplitude scaling under the full complete-trace batch
(four conditions), whose fourth cell doubles as the formal reproduction of the informal
escape.

## 2026-08-27: Decision keys on `full_trace_batch`

The leading surviving hypothesis (per-trace sample density per update) predicts that the full
complete-trace batch alone escapes. The decision therefore keys on the `full_trace_batch`
classification after two gates: the control must reproduce `near_zero`, and the combined
condition must reproduce the escape. The single-ingredient conditions are recorded for
attribution but do not select the decision.
