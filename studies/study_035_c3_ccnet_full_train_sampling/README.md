# Study 035: CCNet-5D fixed patches from the QC training dataset

Status: `completed_degraded`.

This study isolates one change from the adopted Study 033 control: the 512
fixed fit patches are sampled across the complete QC training dataset instead
of its limited dense fit region. The model, patch shape, patch count, artificial
mask, batch size, epoch and update budget, optimizer, learning-rate schedule,
loss, seeds, numerical settings, prediction tiling, checkpoint rule, fixed
validation case, and normalization value remain unchanged.

## Fixed data and normalization

The training dataset is the Study 029 suite's 1,146,366 QC-authorized canonical
train traces and time samples `[0,384)`. Validation and test traces and the 437
QC-excluded traces cannot enter a valid fit patch. Legitimate zero-valued traces
remain eligible. Candidate starts depend only on geometry and authorized row
membership; amplitudes are read from the existing memmap after a descriptor has
been accepted.

The amplitude RMS is loaded from the adopted Study 033 final checkpoint and is
not recomputed. Its checkpoint SHA-256 is
`b8fb3a07e7a9d4a5b3c5540667abbfff8d87f4e24a6358f8cc9cd5e6cfa48697`.
The stored RMS value and its provenance are recorded in each training run.

## Method and primary result rule

Seed 20260908 deterministically rejection-samples 512 unique valid starts for
patches shaped `(64,8,16,8,16)`. Each accepted descriptor has an independent
fixed whole-trace mask seed at 80% missingness. The 32 Study 033 selection
descriptors retain the same selection region and RNG stream.

Training uses the Study 033 width-32/intermediate-32 kernel-3 linear CCNet,
Adam, complete-patch MSE, batch two, eight epochs, 0.001 learning rate through
epoch six and 0.0001 thereafter. The declared-budget final checkpoint at 2,048
updates is the primary checkpoint. Selection is an auxiliary diagnostic; when
fit traces overlap selection traces it is labeled
`diagnostic_only_not_independent_holdout` and cannot replace the final
checkpoint.

Frozen evaluation uses
`c3_benchmark_validation_random_trace_80_seed142`, reinserts observed traces,
and scores all 58,999 targets over all 384 samples in physical amplitudes. The
Study 033 control is 11.660714997498403 dB SNR and 2.595911597220349 RMSE.

No full-training RMS fit, epoch-wise patch resampling, seed retry, larger model,
larger patch budget, loss change, or follow-on experiment is part of this
study.

## Result

The declared final checkpoint completed all 2,048 updates and scored **6.7996
dB** SNR and **4.5430** RMSE on all 58,999 validation targets and 384 samples.
Saved-output rescoring matched the native metric exactly. Relative to Study 033,
the SNR change is **-4.8612 dB**. Under this one seed and validation case,
expanding the fixed-512 sampling domain therefore degraded interpolation
performance.

The 512 unique patches cover 976,504 of 1,146,366 authorized train traces
(85.1826%) and every train source line. Sampling accepted 512 of 941 candidates;
429 candidates contained an absent or unauthorized cell and no duplicate start
was accepted. The fit patches overlap 31,944 unique selection traces, so the
final 4.6045 dB selection score is diagnostic only and is not an independent
holdout.

The fixed Study 033 RMS is 8.351643078934. The patch-presentation RMS of the
fixed Candidate B descriptors is 12.221864869477, 1.4634 times the fixed scale;
this is diagnostic evidence of a normalization-distribution mismatch, not a
full-training RMS fit. Candidate epoch-eight reported loss averaged 0.9701,
compared with 0.0671 for the Study 033 control. These observations may explain
the degradation but do not isolate causality.

[The saved comparison](../../results/study_035_c3_ccnet_full_train_sampling/comparison.csv)
and [machine-readable summary](../../results/study_035_c3_ccnet_full_train_sampling/summary.json)
record exact paths, hashes, metrics, coverage, and resources. No follow-on RMS
or fresh-patch experiment was started.
