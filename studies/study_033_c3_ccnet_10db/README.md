# Study 033: C3 CCNet-5D above 10 dB

Status: `final_2048_adopted_10db_achieved`.

The adopted final 2,048-update CCNet reconstructs all 58,999 target traces and
384 time samples with physical global SNR **11.6607 dB**, RMSE **2.5959** and
relative L2 **0.2612**. Independent saved-output rescoring agrees exactly with
the native evaluation. Fixed CPU restoration passes all declared tolerances.
There is no fixed shear or fitted time alignment.

The [final report](../../reports/c3_ccnet_context_2048_20260909.md) contains the
comparison, learning curve, fixed-location waveforms and adoption evidence.
The [baseline investigation](../../reports/c3_ccnet_investigation_20260909.md)
records the audited -0.0027 dB baseline and train-region diagnostics.

## Fixed input and information use

The Study 029 suite excludes the 437 anomalous traces before the authorized
train pool is formed. Its manifest hash is
`f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`.
The case remains `c3_benchmark_validation_random_trace_80_seed142`, with shape
`(384,9,32,8,32)`, 14,729 observed traces and 22,655,616 target samples.
No additional clipping, trace exclusion or target-derived normalization is used.
Validation truth is reserved for scoring and visualization; test data are not
numerically evaluated.

[Inputs](inputs_context.yaml) specify complete labels from disjoint train-only
shot halves: fit `[32,48)`, internal selection `[48,64)`, with common time
`[0,384)`, source lines `[0,8)`, receiver x `[0,8)` and receiver y `[17,49)`.
Each contains 32,768 dense traces, all in the allowed QC train pool. Geometry
and row membership were checked before reading their amplitudes. Only the
complete fit region determines the fixed global RMS, **8.3516**. Internal
selection labels do not fit this scale.

These training labels differ from per-volume SIREN supervision and from the
GNN's entire allowed train pool. The observed-only scale interpolation used by
those models is not a CCNet input in this condition. Matched-information
architecture superiority is not claimed.

## Adopted condition

[Current configuration](config.yaml) identifies the adoption. The immutable
[executed condition](config_context_width32_batch2_2k.yaml) and
[native fragment](methods/context_width32_batch2_2k.yaml) preserve its
pre-execution declaration.

- Four Conv3D/Conv2D modules, width/intermediate width 32, kernel 3, signed
  linear output: 111,969 parameters. The receptive field is nine samples per
  axis, with halo four.
- Patches `(64,8,16,8,16)`, 512 fixed fit descriptors and 32 internal-selection
  descriptors, whole-trace 80% artificial masks, seed 20260908.
- Adam, batch two, eight epochs and 2,048 updates. Learning rate 0.001 for six
  epochs and 0.0001 for two. Complete-patch MSE includes observed and missing
  samples. No learned input normalization or extra loss term is introduced.
- Float32, both TF32 environment overrides zero, cuDNN benchmark false,
  physical CUDA device zero and one CPU thread. Native numerical flags are
  recorded separately from environment overrides.
- Frozen prediction uses cores `(64,9,16,8,16)`, 24 tiles and clipped halos;
  maximum input is `(72,9,20,8,20)`. Physical observed traces are reinserted
  exactly after prediction.

The final checkpoint is also the best by internal-selection score. Its internal
patch-instance SNR is **9.2691 dB**, a different domain from the **11.6607 dB**
full validation score. There was no validation-based stopping or checkpoint
substitution. Complete sample presentations total 4,294,967,296 with overlapping
patches; this is not a unique-sample count. The expanded condition changes
context, width, batch size, learning rate, teacher-region split and budget
jointly, so its improvement is not attributed to a single change.

## Evidence and limits

Training plus internal selection took **580.4847 s**; frozen prediction took
**2.3190 s**. These native stage times exclude loading, verification and other
outer-action work. Training peak allocated/reserved memory was
**4.4728 / 5.5512 GB**. The two-update resource preflight is not a memory or
runtime upper bound for another experiment.

CPU restoration uses predeclared native tiles 0, 12 and 23 with the same
core/halo and only observed inputs. It covers 2,838,720 target samples. Normalized
difference RMSE is **1.5475e-7**, maximum difference **2.0554e-6**, and physical
relative L2 **4.3232e-7**, below fixed limits 1e-4, 1e-3 and 1e-3. It complements
full saved-output scoring and does not claim full-volume CPU bitwise identity.

The 183 Python files frozen before training match publication source commit
`f2dccac6d1e33294156710dc688fc21667aff89a` byte for byte. Native records preserve
the actual earlier training commit and dirty-worktree warning. Adoption here
is an audited Study 033 reference, not the Study 025 clean-formal designation
or a complete reproduction of the original kernel-5, width-64 paper candidate.
Only one seed and one fixed validation case have been evaluated.

The training native run is
`runs/study_033_c3_ccnet_10db/20260909T130443857538Z_5df55f375e66_ccnet-train/native`;
its final checkpoint SHA-256 is
`b8fb3a07e7a9d4a5b3c5540667abbfff8d87f4e24a6358f8cc9cd5e6cfa48697`.
Frozen prediction is
`runs/study_033_c3_ccnet_10db/20260909T131457466479Z_f2dccac6d1e3_ccnet-predict/native`.
The completion handoff at
`runs/study_033_c3_ccnet_10db/20260909T130443471844Z_5df55f375e66_context_2048_completion_handoff`
records exact commands, independent scoring, CPU restoration and unchanged
source/config hashes. Large predictions and checkpoints remain local.

Execution uses `scripts/run_c3_first_results.py` with explicit config, inputs,
action and `--execute`; prediction requires the corresponding `final.pt`.
Use a new immutable run for any new condition. Current decision rationale is
in [decisions.md](decisions.md).
