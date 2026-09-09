# Decisions

## 2026-09-09 — Transfer complete-trace Cartesian SIREN to the fixed physical benchmark

**Evidence:** Study 030 per-trace RMS plus observed-scale IDW gives physical
SNR -0.0002 dB after 2,000 random-point updates. Scale-estimation relative L2
is5.3229%, while the model waveform remains near zero. Studies014–015 show
that complete-trace batches can escape this failure on a small training pool.
The best prior held-out coordinate-only SIREN is scratch Stage12 at
`runs/study_all_ffid_temp/20260828T175604Z_f7c0ea2_stage12_trace_batch_ffid2348_2363_cartesian_omega90_w256_cosine`:
10.6805 dB withCartesian5 inputs, width256, four layers, omega90/30,435 complete
traces/update and50,000 cosine-decayed Adam steps. It evaluated oracle unit-RMS
on80% observed16FFIDs, so it does not establish physical performance at the
current20% observed density.

**Decision:** Adopt that model/batch/optimizer recipe as the first measured
transfer. Keep the current physical scale estimator and full fixed target domain.
Use one seeded complete-trace selection per update and bounded, sample-weighted
gradient accumulation when necessary. Do not add correlation loss, supervised
neighbor waveforms, target scale oracle, or a different method under the SIREN name.

**Authorization:** The user's explicit request to improve past10 dB permits
recorded new variants and training-budget revisions beyond the earlier2,000-step
single-run diagnostic. It does not change the target-data boundary. All runs
remain immutable and all attempted conditions will be reported.

## 2026-09-09 — Set the first full-run time cap from the measured preflight

Preflight `20260909T005221115308Z_1819109f28e1_siren` completed ten updates
in 0.8187 seconds, giving a conservative linear 50,000-update estimate of
4,093.5388 seconds. Peak allocated GPU memory was 1,794,512,896 bytes, within
the available memory on CUDA device 1. Before the full run, increase its
process time cap from 3,600 to 7,200 seconds to cover that estimate and input,
prediction, and evaluation work. The declared 50,000 updates and all model,
data, scoring, and seed settings remain fixed. The preflight model is discarded.

## 2026-09-09 — Test temporal coordinate bandwidth after the first global transfer

Run `20260909T005658230199Z_1819109f28e1_siren` finished all 50,000 updates
in 345.6280 seconds overall. Full-target physical SNR was -0.7700 dB;
observed-model RMSE before reinsertion was 9.0161 and final normalized batch
MSE was 0.8218. It fails the target and is retained as a measured unsuccessful
transfer. Partial observed fitting has not produced useful missing-trace
interpolation.

The next predeclared variant changes only `model.time_coordinate_scale` to
4.0, retaining Cartesian5, width256, omega90/30, seeds, all input/scaling
contracts, 435 complete traces/update, and 50,000 cosine-decayed updates.
Past held-out SIREN trials included this time factor, and scaling time
independently changes the temporal frequencies represented at initialization
without increasing the spatial frequencies. The final checkpoint and full
physical target remain the selection boundary. No previous checkpoint is reused.

## 2026-09-09 — Prioritize the user-requested larger-batch investigation

The user explicitly requested investigation of increasing batch size while
the time-scale-4 transfer was running. Complete that declared run, then
prioritize the number of observed traces per optimizer update and the
physical forward microbatch separately. The current435 traces are only
2.9534% of the14,729 observed pool; in Study014,435 complete traces were
the entire training pool. Its strong train-only full-batch result therefore
provides a more specific motivation than merely increasing random points.

Predeclare an observed-only resource pilot with435,1,740,6,960,and14,729
complete traces per update, each at262,144 and524,288 maximum forward points.
Use fresh identical seeded Cartesian5/time-scale4 models and ten Adam
updates per condition, retaining the first two as startup measurements and
reporting subsequent timing separately. Keep the same observed RMS input;
read no target waveforms. Run sequentially on CUDA1 after the active run
ends, with a120-second cap per condition. Preserve OOM/failure records.
A resource pilot is not an interpolation-quality result. Choose the later
learning comparison budget from measured time/memory, and report optimizer
updates, total presented points, and elapsed time alongside observed fit.

## 2026-09-09 — Retain the unsuccessful time-scale-4 transfer

Run `20260909T010327892987Z_1819109f28e1_siren` completed 50,000 updates
in344.8479 seconds overall. Physical target SNR is-2.5383 dB and RMSE
13.3119. Observed-model RMSE improved to5.0725, with final normalized
batch loss0.2676. The improved observed fit does not generalize to the
missing traces, so this run is not adopted. Keep it intact while prioritizing
the requested batch-size investigation.

The historical full-batch evidence is Study014
`runs/study_014_full_trace_batch_ablation/20260827T041050Z_925e8e4_summary.json`.
At the same50,000 updates,5,000 random points yielded median training SNR
0.0039 dB, all435 complete traces yielded8.9140 dB, and per-trace RMS
yielded16.1377 dB. Full-batch exposures were54.375 times larger; these
are training-only results, not a matched-compute interpolation comparison.

## 2026-09-09 — Compare stochastic and full observed batches at 5,000 updates

Resource pilot `20260909T011126736333Z_1819109f28e1_large_batch_resource`
completed all eight conditions without OOM. For all14,729 traces, maximum
forward262,144 points used2,763,041,792 allocated bytes and the median
warm step took0.2378 seconds. Doubling the forward bound used5,455,784,960
bytes and0.2883 seconds; retain262,144 for the learning comparison.

Before execution, declare two fresh matched-update conditions:435 and
all14,729 observed traces/update, each5,000 Adam updates at constant
1e-4, report interval100, Cartesian5/time-scale4/omega90-30/width256x4.
Use the existing seeds, observed per-trace RMS, IDW scales, fixed target
mask/crop, and final-checkpoint scoring. Constant learning rate avoids
confounding the comparison with a shortened cosine decay. Prediction
still uses65,536 points per forward. These are5,000 updates each but
835,200,000 versus28,279,680,000 presented points, a33.8598-fold compute
exposure difference. Compare physical target SNR as well as observed fit;
the full-batch estimate is about1,189 seconds of warm training. The existing
7,200-second action cap covers each condition. Preserve both results and
decide later budgets from them; no intermediate checkpoint is selected by
target SNR.

## 2026-09-09 — Full observed batch fits observations better but fails interpolation

The matched5,000-update run with435 traces
`20260909T011656120058Z_1819109f28e1_siren` has observed-model RMSE6.7069,
target RMSE12.0943 and physical target SNR-1.7051 dB. The full14,729-trace
run `20260909T011832015888Z_1819109f28e1_siren` has observed-model RMSE
5.0412, target RMSE13.9549 and target SNR-2.9480 dB, completing in
1,387.3571 seconds overall. Thus improved observed fit does not imply improved
interpolation. The global SNR and RMSE are consistent within each fixed
domain; the apparently opposing changes concern different domains.

## 2026-09-09 — Predeclare the observed-derived temporal shear at the same full batch

Observed-only audit `20260909T010135591280Z_1819109f28e1_observed_bandwidth_audit`
found a24ms relative shift for every one of2,048 adjacent receiver-y pairs
separated by40m, with median correlation improving from-0.7132 to0.9343
after alignment. The observed temporal energy median frequency is21.1589Hz.
The older Study019 Stage05 independently used3 samples per receiver-y cell
for neighbor waveform alignment, but did not improve its CNN result; it is
a precedent for the observed alignment mechanism, not proof of SIREN quality.

Keep the previous full14,729-trace batch,5,000 constant Adam1e-4 updates,
Cartesian5, time factor4, omega90/30, width256x4, seed, all data/scales and
scoring settings. Change only the fixed input transformation to
`tau = time_s + 0.0006 * relative_receiver_y_m`, equivalently
`tau = time_s - 0.0012 * half_offset_y_m` because half-offset is
(source minus receiver)/2. Transform coordinates during observed training
and at every prediction location, with no waveform resampling, time-window
change, or target-derived coefficient. The original physical samples are
still predicted and evaluated over the entire fixed target domain.

The invertible five-coordinate transformation changes optimization geometry
without adding input features or restricting the representable function class.
It is a measured hypothesis about the phase relationship, not a declaration
that overfitting or any single cause has been established. The final checkpoint
after the declared budget remains the comparison boundary.

## 2026-09-09 — Predeclare lower spatial coefficients with matched initial temporal coefficients

While the full-batch shear run is still training, declare the next candidate
conditionally on its final physical target SNR not exceeding10 dB. Retain the
same full14,729-trace batch,5,000 constant Adam1e-4 updates, hidden omega30,
width256x4, shear0.0006, seeds, per-trace normalization, observed-only gain
interpolation, and full final-checkpoint evaluation. Change first-layer omega
from90 to30 and time coordinate scale from4 to12. Store this executable
condition in `config_omega30_time12_shear_batch14729_5k.yaml`.

The product of first omega and time scale remains360. For the same raw
initial weights this preserves the initial temporal and shear-derived terms
inside the first sine, while reducing the four independent spatial terms
by a factor of three. First-layer bias phase and parameter sensitivities
also change, so this is not a perfectly isolated spatial-frequency ablation
or a guarantee of a hard bandwidth limit. The existing initialization gives
the same raw weights and biases with the same seed and null omega schedule.

The motivation is the completed no-shear batch comparison: stronger target
predictions remained poorly aligned with truth. The older oracle held-out
omega30/time4 run reached10.4732 dB, but no past comparison held omega times
time scale fixed. The new physical-target outcome is therefore unproven.
Do not run this conditional candidate if the current model already exceeds
the declared target and passes independent validation. Never reuse its weights.

## 2026-09-09 — Temporal shear improves the full physical target but remains below the goal

Run `20260909T015011064447Z_1819109f28e1_siren` completed all5,000 updates
in1,298.3319 seconds overall. Full-target physical SNR is7.8831 dB, target
RMSE4.0102, and observed-model RMSE3.3105. Compared with the matched
no-shear full-batch run, both observed fitting and missing-trace prediction
improved substantially. Thus the observed-derived time/receiver-y relation
is a useful model-coordinate change; the evidence does not establish a unique
cause for the earlier generalization failure.

Retain and independently audit the completed output. Since it has not exceeded
10 dB, execute the previously declared omega30/time12 candidate from fresh
seeded weights. Its final full-target score, not intermediate observed loss,
determines whether it meets the goal. Preserve the preceding executable plan
as `config_time4_shear_batch14729_5k.yaml`.

## 2026-09-09 — Adopt the independently verified 11.3422 dB configuration

The predeclared omega30/time12 run
`20260909T021337632365Z_1819109f28e1_siren` completed5,000 fresh Adam
updates in1,216.6931 seconds overall. Physical target SNR is11.3422 dB,
target RMSE2.6929, observed-model RMSE2.3861, and target uncentered cosine
0.9626. The full58,999-target,22,655,616-sample CPU float64 rescore in
`20260909T022105534877Z_1819109f28e1_omega30_time12_saved_output_audit`
exactly reconciles with the native result. Prediction samples are all finite,
observed reinsertion is exact, and row IDs, gains and input hashes are unchanged.

The independently prepared CPU checkpoint restoration audit
`20260909T021544855410Z_1819109f28e1_checkpoint_cpu_restoration_audit`
also passed its fixed engineering tolerances on16 target traces selected by
geometry before completion. It reconstructs physical predictions using saved
gains and independently checks the transformed time coordinate. This is a
sampled GPU-to-CPU numerical comparison, not a claim of bitwise equality or
an all-target checkpoint-forward audit.

Adopt this final configuration and its immutable checkpoint as the model that
meets the declared physical validation target. Retain all unsuccessful runs
and comparisons. Keep large predictions and model artifacts in the source run;
publish compact comparison artifacts and the source paths and hashes. This
selection uses the fixed validation case and does not establish test performance.

## 2026-09-09 — Isolate fixed shear at the adopted omega30/time12 setting

The user requested the missing matched comparison: keep the adopted omega30
and time scale12 and set only the fixed shear to0. The earlier no-shear
result used omega90/time4, so it does not answer whether shear is required
at the adopted frequency settings.

Before execution, declare one fresh5,000-update run with all14,729 observed
traces per update, constant Adam1e-4, microbatch262,144, seed20260908,
width256x4, hidden omega30, Cartesian5, the same observed per-trace RMS and
observed-only IDW gains, and the identical fixed QC case/crop/mask. Change
only the native model field `relative_receiver_y_time_shear_s_per_m` from
0.0006 to0.0. The executable plan is
`config_omega30_time12_shear0_batch14729_5k.yaml`; the adopted `config.yaml`
remains the reference configuration. Discard the ten-step preflight model,
then train from the same fresh seed with no checkpoint reuse. Preserve the
7,200-second action cap and do not run automatic quality retries.

Compare its declared final checkpoint with adopted run
`20260909T021337632365Z_1819109f28e1_siren` on every58,999 target trace
and384 original time sample. Independently re-score physical RMSE/SNR,
prediction RMS and uncentered cosine; verify identical input hashes, row IDs
and gain vectors. Treat observed fitting separately from target quality.
The requested comparison is complete regardless of whether shear0 exceeds
10 dB; it does not authorize a new parameter search or test-partition evaluation.

## 2026-09-09 — Complete the requested matched shear ablation

Run `20260909T030400486453Z_1819109f28e1_siren` completed all5,000 fresh
updates in1,140.3644 seconds overall. With omega30/time12 and shear0,
physical target SNR is-3.1486 dB, target RMSE14.2809, and observed-model
RMSE5.2413. The fixed-shear reference has11.3422 dB,2.6929 and2.3861,
respectively. Thus the requested one-field removal substantially worsened
both observed fitting and missing-trace accuracy under the same budget.

Independent CPU audit
`20260909T025839182592Z_1819109f28e1_shear0_ablation_saved_output_audit`
exactly reconciles both full-target scores. All58,999 targets and22,655,616
samples are covered; predictions are finite and observed reinsertion is exact.
Inputs, row IDs and gain vectors are identical. Target uncentered cosine
changes from0.9626 to-0.0540, while prediction RMS remains similar at9.6059
and9.7326. The degradation is therefore accompanied by loss of waveform
alignment rather than a changed gain estimator.

Keep the existing fixed-shear model as the adopted result and publish this
ablation as a diagnostic comparison. The current configuration and5,000-step
budget benefit strongly from the observed-derived fixed transform. This does
not establish that SIREN cannot represent the relation or learn it with other
initializations, optimization settings or budgets. The requested comparison
is complete; do not expand it into an automatic new search.

## 2026-09-09 — Exclude SIREN-only fixed shear from the common-method comparison

The user rejected supplying observed-derived fixed shear only to SIREN:
that preprocessing would also have to be made available to the other methods
for the intended comparison. Observed-only fitting avoids target-waveform
leakage, but does not satisfy this separate requirement for common conditions.
The primary comparison therefore uses no externally estimated fixed shear;
the time/space relationship remains part of what SIREN must learn.

This decision supersedes the adoption in “Adopt the independently verified
11.3422 dB configuration” and the instruction to retain that adoption in
“Complete the requested matched shear ablation.” The measured 11.3422 dB
score remains valid for its shear-assisted configuration, but is diagnostic
and no longer counts as a common-condition 10 dB success. The earlier
7.8831 dB fixed-shear result has the same diagnostic classification.

Point current `config.yaml` at the already completed omega30/time12/shear0
condition, retaining every other native setting. Its matched final result is
-3.1486 dB with target RMSE14.2809; this is the current comparison condition,
not a claim that it is the best shear-free recipe. The common-condition
10 dB goal remains unmet. Keep automatic quality retries disabled; this
correction does not launch a new search or modify the other methods.

Preserve the original runs, scores and published manifests as historical
evidence. Record the superseding classification and source hashes in
[the comparison classification](../../results/study_031_c3_siren_10db/20260909T033540186900Z_1819109f28e1_comparison_contract_revision/comparison_classification.json),
including the old manifest whose `model_adopted` and `goal_achieved` flags
describe the withdrawn adoption. Current study and reports use this decision.

## 2026-09-09 — Predeclare three shear-free initialization and envelope-loss conditions

The user explicitly authorized implementing and executing three additional
conditions without fixed shear. This is a bounded follow-up to the completed
shear ablation, not an automatic quality retry. The reference remains the
fresh5,000-step run `20260909T030400486453Z_1819109f28e1_siren`: physical
target SNR -3.1486 dB, target RMSE14.2809, observed-model RMSE5.2413.
Keep current `config.yaml` and the historical runs unchanged until a new
common-condition result exceeds10 dB and passes independent audits.

Declare all three conditions before their preflights or full runs:

| Condition | Plan | Native training changes from the reference |
|---|---|---|
| A: time_init3 | [config_time_init3.yaml](config_time_init3.yaml) | `initial_time_weight_scale: 3.0` only |
| B: envelope | [config_envelope.yaml](config_envelope.yaml) | `envelope_loss: {weight: 1.0, sigma_samples: [4.0, 8.0], decay_steps: 2500}` only |
| AB: time_init3_envelope | [config_time_init3_envelope.yaml](config_time_init3_envelope.yaml) | Both changes above |

Keep Cartesian5, width256x4, omega30/30, time scale12, fixed shear0,
per-trace observed RMS normalization and the same observed-only IDW gains.
The initialization option scales only the first layer's time-input weights
once after seeded initialization; it does not alter the time coordinate,
insert a time/space shear, or resume a previous checkpoint. The envelope
term uses observed traces only and supplements the retained `training.loss: l2`.
Its coefficient is1.0 at optimizer step1, decreases linearly to0.0 at
step2500, and remains0.0 thereafter. Record interval means of `train_loss`,
`waveform_mse`, and `weighted_envelope_mse`, together with `envelope_weight`
at the interval's final step. Unweighted envelope MSE is not persisted
separately. A reporting interval ending at step2500 can have a positive mean
weighted contribution even though its final weight is zero. Do not label the
combined objective as plain MSE or compare it directly with physical RMSE.

All conditions use all14,729 observed traces and all384 original samples per
optimizer update, fresh seed20260908, Adam at constant1e-4,5,000 updates,
maximum262,144 forward points, and65,536 prediction points per forward.
When the envelope term is active, forward microbatches contain complete
traces:682x384=261,888 points. Retain the legacy262,144-point bound for the
ordinary MSE path. Every condition has the same logical28,279,680,000-point
exposure, but the altered microbatch boundaries and reductions preclude a
claim of bitwise equality with the legacy reference trajectory.

Run A, B, then AB sequentially on CUDA1. Each condition gets a fresh ten-step
preflight whose model is discarded before a separately initialized full run.
Retain the600-second preflight cap and7,200-second full-action cap. Keep
automatic quality retries, GPU action parallelism, CPU training fallback,
and test-partition execution disabled. Preserve every outcome and do not
extend an individual training budget or select an intermediate checkpoint.

After completion, independently rescore each saved final physical prediction
on all58,999 targets and22,655,616 samples; verify finite values, exact observed
reinsertion, unchanged input hashes, row IDs and stored gain vectors. Compare
observed-model RMSE separately from target RMSE/SNR and record prediction
RMS, uncentered cosine and the full energy identity. Check the declared
initialization and loss metadata, fresh final5,000-step role, and all loss
components without claiming identical checkpoint training metadata across
variants. A candidate is eligible for adoption only after strictly exceeding
10 dB under the shear-free common condition and passing the full-target and
checkpoint-restoration audits. No target waveform enters initialization,
envelope-loss fitting, gain fitting or training; validation scoring may guide
the later choice, while test amplitudes remain unused.

## 2026-09-09 — Complete the three declared conditions without adopting a model

The bounded A, B, and AB runs completed all 5,000 updates from fresh
seed20260908 initializations. Each used the same 14,729 observed traces,
384 samples per trace, fixed validation targets, physical gain vectors,
Cartesian5/omega30/time12 coordinates, and zero fixed shear. Retain the
reference run without retraining it. Final physical metrics are:

| Condition | Target SNR (dB) | Target RMSE | Observed-model RMSE before reinsertion |
|---|---:|---:|---:|
| Reference | -3.1486 | 14.2809 | 5.2413 |
| A: time initialization ×3 | -3.0870 | 14.1799 | 5.2205 |
| B: envelope auxiliary loss | -1.6301 | 11.9903 | 5.2220 |
| AB: both additions | -2.7047 | 13.5693 | 5.5394 |

Independent rescoring of all three new saved predictions covered all 58,999
targets and 22,655,616 samples and agreed exactly with their native scoring
records. B was best among these four declared conditions, but its SNR remained
negative and did not exceed 10 dB. Set `model_adopted=false`, retain current
`config.yaml`, and keep the common-condition goal unmet. This completes the
authorized three-condition comparison; do not extend the budgets or launch
quality retries.

Keep the full-target saved-prediction checks separate from checkpoint
restoration. The predeclared CPU restoration checks failed tolerance for all
three additions: A failed all three numerical criteria; B and AB exceeded
only the normalized maximum-error limit. Preserve those failures and their
original thresholds. A separate 16-target diagnostic reproduced A's saved
predictions exactly using the original native GPU precision and batch context;
GPU results with the highest float32 matmul setting agreed with CPU within
the fixed tolerances. A compact GPU batch produced large differences under
the native high setting; under highest, compact and native batch predictions
were identical. This is evidence about A's tested subset, not proof of
the cause of B or AB's differences or of full-target checkpoint equivalence.

These results describe one seed on one fixed validation case. The envelope
conditions also change microbatch boundaries while the auxiliary weight is
positive, so the comparison does not isolate the mathematical loss from all
floating-point effects. No target waveform entered training, initialization,
or gain fitting, and the test partition remained unused. Preserve all runs,
failed restoration checks, and the earlier shear-assisted diagnostic results.
See the [final comparison report](../../reports/c3_siren_time_learning_20260909.md)
and [published comparison records](../../results/study_031_c3_siren_10db/20260909T044151000000Z_1819109f28e1_time_learning_comparison).
