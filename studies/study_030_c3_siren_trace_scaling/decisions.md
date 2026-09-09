# Decisions

## 2026-09-09 — Estimate missing-trace physical RMS from observed traces

**Evidence:** [Study 013](../study_013_amplitude_balancing/README.md) found that
per-trace RMS alone under small random-point batches remained near zero.
[Study 014](../study_014_full_trace_batch_ablation/README.md) reached training-fit
median trace S/N of 8.9100 dB with full complete-trace batches and global RMS,
and 16.1400 dB with per-trace RMS. These were training-fit measurements, not
physical held-out interpolation scores. Studies 017–021 evaluated waveform
shape using oracle target RMS, which does not provide an inference-time gain.

**Decision:** Follow the user's requested per-trace normalization with a new
observed-only physical RMS interpolation stage. Retain the source QC suite,
zero traces, fixed crop, mask, seed and 2,000-update SIREN budget. This isolates
the requested scaling change. It does not assume that normalization alone
will reproduce the full-trace-batch improvement of Study 014.

**Decision:** Use a fixed eight-neighbor inverse-distance-squared arithmetic
interpolator in source/relative-receiver coordinates. Normalize distances by
the measured acquisition spacings `[160,80,40,40] m`. The alternating source
line stagger remains in the actual physical coordinates. Only observed scales
are eligible; target geometry is known, while target waveforms are reserved
for evaluation. Do not tune the metric, neighbor count or power against held-out
amplitudes or use oracle gains to reconstruct the primary prediction.

**Reason:** A positive weighted average keeps gains within the selected
observed RMS range and preserves valid zero gains. An explicit row-bound gain
field and interpolation parameters in the checkpoint make physical restoration
reviewable. This study supplies the missing-location scaling mechanism that
prevented physical per-trace normalization in the prior SIREN contract.
