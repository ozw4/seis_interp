# Decisions

## 2026-09-08 — Restore explicit physical trace QC in a separate input contract

**Evidence:** The raw IBM float words in FFID 1746 reproduce the large finite
amplitudes in the interim array. Its source-file SHA256 agrees with the download
lock and interim metadata. The 544 original traces comprise 437 traces with
peak amplitude above 10000 and 107 exact-zero traces. Every nonzero peak is at
least 115210. This agrees with the previously recorded Study 016 investigation.
The origin or intention of this anomalous block remains undetermined.

**Decision:** Reuse the previously declared absolute bound 10000. Exclude the
whole offending trace, preserve its original array-row identifier in the
partition audit, and retain zero traces. Apply the fixed QC over all original
samples before masks, independently of the subsequent model training time.
Generate a new suite under Study 029; preserve the old suite, interim amplitudes,
SEG-Y files, checkpoints and first-results records.

**Reason:** Study 027 required an unfiltered source-line partition, so the
previous amplitude filter was not applied. Finite-only checks and float64 RMS
accumulation accepted values near the float32 limit. A robust scale alone would
leave those physical targets in the training data and could create a different
overflow. QC addresses their eligibility before fitting the unchanged RMS.

## 2026-09-08 — Keep zeros and use a bounded train-only numerical probe

**Decision:** Do not inherit Study 016's removal of all-zero traces or switch to
per-trace normalization. Keep the fixed evaluation geometry, mask recipes and
target IDs. Reject an exclusion that silently promotes an alias or removes a
fixed crop row. Recompute exclusions independently during full verification.

**Reason:** Zero amplitudes are valid in the benchmark contract. A per-query
normalization scale derived from a missing waveform is unavailable at inference.
The repair must preserve a scale fitted on eligible training rows and avoid
using held-out labels for normalization or threshold selection.

**Decision:** Add an explicit physical amplitude guard before graph RMS fitting
and require numerical-retention checks for this study. Use one predetermined
CPU probe: four hidden training queries, 100 updates, one seed and no quality
retries. It tests restored signal representation and local learning; it cannot
establish full-volume interpolation quality.

## 2026-09-09 — Rerun SIREN on the QC suite at the original budget

**Decision:** Follow the requested SIREN-5D rerun by binding the new suite to the
original validation case and unchanged SIREN fragment. Start from a fresh model
with seed 20260908 and retain 2,000 updates. Keep the original run and checkpoint.

**Reason:** This isolates the effect of the 437 source-trace exclusions. SIREN
fits only the selected volume's observed samples and their local RMS; the
excluded train-partition traces do not enter this model. Its unchanged score
therefore does not test whether repaired train normalization improves a model
that learns across the full training partition. The near-zero validation SNR
and near-unit training loss do not justify a claim of successful interpolation.
