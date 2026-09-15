# NeRSI at the GNN width128 parameter budget

Status: running. Target: full-T physical-amplitude mean trace SNR >15 dB.

GNN width128 run `20260914T150810Z_684c4e4c659a_mask10_fourier16_width128_20k`
has 363,269 parameters. Each candidate has exactly 363,269 trainable parameters,
without dummy parameters. GNN trained for 20,000 updates; these NeRSI runs use
50,000 updates as requested. This is not an equal-compute comparison.

| Config | Fourier components | Encoder width | Latent channels | Decoder channels |
|---|---:|---:|---:|---|
| fourier160.yaml | 160 | 96 | 13 | [13, 7, 4] |
| fourier40.yaml | 40 | 40 | 28 | [29, 14, 8] |
| fourier16.yaml | 16 | 33 | 29 | [33, 17, 11] |

Configurations inherit the fixed v3 no-shear NeRSI settings. Only the four
architecture fields listed above may differ. Global RMS is O-only; loss is MSE;
Adam uses constant LR 0.001, EMA decay 0.999, and 50,000 updates from initialization.
Seeds, sampling, prediction, all input hashes, coverage and physical evaluator stay fixed.
No cosine schedule, time shear, target exclusions or per-trace normalization.
Integer widths were chosen to match the parameter budget before evaluating T.
This tests both Fourier bandwidth and parameter allocation, not a single-factor
Fourier ablation. T is used for exploratory selection, not independent evaluation.

The first batch runs sequentially on GPU 1, retaining every success or failure;
there are no automatic retries or training extensions. Inspect completed metrics
before deciding subsequent candidates. Existing baselines remain unchanged.

## Completed first batch and next allocation comparison

The first batch completed successfully: Fourier160 mean trace SNR 11.8699 dB,
Fourier40 13.7965 dB, Fourier16 13.5338 dB. Full input locks, artifact hashes and
all-target coverage were verified. The >15 dB target remains unmet.

The allocation batch also completed: encoder_heavy 13.7505 dB,
decoder_heavy 13.4981 dB, kernel3 13.3931 dB mean trace SNR.
The best remains fourier40 at 13.7965 dB.

The second batch fixes Fourier40 and LR=0.001. Each model still has exactly
363,269 parameters and trains for 50,000 updates. Only the widths listed below
change, plus kernel size for the kernel3 candidate. Seeds, EMA, preprocessing,
sampling and evaluation remain identical to fourier40.yaml. No learning-rate search.

| Candidate | Encoder | Latent | Decoder | Kernel |
|---|---:|---:|---|---:|
| encoder_heavy | 91 | 15 | [26, 12, 5] | 5 |
| decoder_heavy | 19 | 35 | [36, 20, 13] | 5 |
| kernel3 | 51 | 28 | [32, 24, 13] | 3 |

```bash
bash scripts/run_c3_v3_nersi_budget.sh encoder_heavy decoder_heavy kernel3
```

```bash
bash scripts/run_c3_v3_nersi_budget.sh fourier160 fourier40 fourier16
```

## Frequency-base comparison

`base105.yaml`, `base110.yaml`, `base115.yaml` change only
`model.frequency_base` to 1.05, 1.10, 1.15 respectively, relative to fourier40
(base 1.25). All have exactly 363,269 parameters and 40 Fourier components.
Learning rate stays 0.001, with 50,000 updates, EMA 0.999, Global RMS and no shear
or cosine schedule. These are new runs from the same seed, not continuations.
All three candidates run sequentially; preserve every result regardless of SNR.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base105 base110 base115
```

## Approximately 64 supervised traces per update

Frequency-base runs completed: base105 13.6919 dB, base110 13.7397 dB,
base115 13.8147 dB mean trace SNR. The best is base115.
`base115_profiles10.yaml` changes only `training.profiles_per_step` from 16 to 10.
It completed at 13.5967 dB mean trace SNR, with 3,216,942 supervised trace
presentations (64.3388 per update). Ten profiles/update is adopted by user choice
to approximately match supervised traces per update, not because it maximizes SNR.
All other conditions, including LR=0.001 and
50,000 updates, are unchanged. No gradient accumulation is added.

Ten profiles contain about 64 observed traces on average, not exactly 64 at every
update. The existing sampler draws only profiles with at least one O trace;
the expectation is `10 * O_count / eligible_profile_count`. The full grid has
4,096 profiles and 26,322 O traces. No O labels are dropped to force a count of 64.
Only O traces contribute to the loss; unobserved receiver-y cells do not.
The cumulative actual count is recorded in `compute.supervised_trace_presentations`.
This approximately matches supervised trace count, not FLOPs or input context size.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10
```

## AdamW candidate

`base115_profiles10_adamw.yaml` changes only the adopted configuration's optimizer
to AdamW. Weight decay is explicitly zero in the trainer and recorded in run
metadata; switching optimizer does not silently add PyTorch's AdamW default decay.
With zero decay Adam and AdamW are mathematically equivalent under the same
hyperparameters, so a substantive SNR improvement is not expected from this change
alone. No learning rate, EMA, sampling or update-budget change is introduced.
The final checkpoint remains an inference snapshot without optimizer state.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10_adamw
```

The AdamW candidate completed at 13.6217 dB mean trace SNR. It retains ten
profiles/update and 3,216,942 supervised trace presentations. The zero-decay
AdamW change improved the Adam result by 0.0250 dB, which is too small to treat
as evidence of an optimizer advantage.

## Shot-axis profile candidate

`base115_profiles10_adamw_shot_profile.yaml` changes only the generated profile
axis from `relative_receiver_y` to `shot_in_line`. Both axes contain 32 traces,
so model shape and the exact 363,269 parameter count remain unchanged. Profiles
are keyed by `(source_line, relative_receiver_x, relative_receiver_y)` in stable
C order. Ten sampled profiles still present approximately 64 O traces per
optimizer update; the exact cumulative count is recorded in metadata.

This is motivated by the O-only time-shift diagnostic: receiver-y has a strong
approximately 3 sample/cell shift, while the other spatial axes selected zero
integer shift. Since this study forbids time shear, decoding the shot axis tests
a profile orientation with less uncorrected moveout. Global RMS, MSE, AdamW with
zero weight decay, constant LR 0.001, EMA 0.999, 50,000 updates, seeds, inputs,
full-T physical mean-trace-SNR evaluation, and exact O reinsertion stay fixed.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10_adamw_shot_profile
```

The shot-axis candidate completed at 10.9229 dB mean trace SNR and is rejected.
It preserved all input, compute and artifact contracts, but its final training
MSE (0.05994) and target error were substantially worse than the receiver-y
profile baseline. The result remains stored; the profile axis is returned to
receiver-y for subsequent candidates.

## AdamW regularization candidates

`base115_profiles10_adamw_wd0001.yaml` and
`base115_profiles10_adamw_wd001.yaml` return to the adopted receiver-y profile
and change only AdamW weight decay from zero to 0.0001 or 0.001. This directly
tests whether the 363,269-parameter model's observed-to-target generalization
gap is caused by overfitting. Parameter count, ten profiles/update, LR 0.001,
50,000 updates, EMA 0.999, Global RMS, no shear, seeds and evaluator stay fixed.
Both candidates start from initialization and are retained regardless of T.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_wd0001 base115_profiles10_adamw_wd001
```

## Parameter-efficient decoder candidate

`base115_profiles10_adamw_depthwise.yaml` uses depthwise-separable convolution
inside each existing PixelShuffle decoder block. It reallocates the saved
convolution parameters to encoder width 54, latent channels 32 and decoder
channels `[30, 27, 24]`, retaining exactly 363,269 trainable parameters. The
three-block profile decoder, output resolution and signed linear output remain
unchanged. Relative to the adopted standard-convolution model, this shifts the
budget from 230,056 encoder / 133,213 decoder-output parameters to 350,934 /
12,335, closer to the encoder-dominant allocation of the successful large
NeRSI while respecting the GNN width128 parameter budget.

This is an architecture candidate, not a compute-equivalent convolution
ablation. Input lock, receiver-y profile orientation, ten profiles/update,
AdamW with zero weight decay, LR 0.001, EMA 0.999, 50,000 updates, seeds, Global
RMS, no shear and full-T evaluator remain fixed.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10_adamw_depthwise
```

The depthwise candidate completed at 11.0780 dB mean trace SNR and is rejected.
Its final train MSE was 0.07998, confirming that depthwise spatial mixing made
the fixed-budget model underfit despite the wider encoder.

## Axis-specific Fourier bases

The profile coordinate axes have 16 source lines, 32 shots and 8 receiver-x
positions. `base115_profiles10_adamw_axis_rx105.yaml` keeps base 1.15 on the
first two axes and lowers only the sparse receiver-x axis to 1.05.
`base115_profiles10_adamw_axis_nyquist.yaml` uses `[1.07, 1.09, 1.05]`, close
to the maximum exponential bases whose 40th frequencies remain below each
regular index grid's Nyquist frequency. Unlike the earlier hard bandlimit,
these mappings keep all 240 Fourier features active. They add no parameters.

Both candidates restore the standard decoder and zero AdamW weight decay.
All fixed input, parameter, update, trace-presentation, seed, LR, EMA, Global
RMS, no-shear and evaluation conditions are unchanged.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_rx105 base115_profiles10_adamw_axis_nyquist
```

The receiver-x-only candidate scored 13.4476 dB and was rejected. The
all-axis Nyquist candidate scored 13.7633 dB, improving the adopted ten-profile
baseline by 0.1416 dB but remaining below 15 dB. Because lowering receiver-x
alone was harmful, `base115_profiles10_adamw_axis_source107_shot109.yaml` keeps
receiver-x at 1.15 and lowers only source/shot to `[1.07, 1.09]`. This isolates
the combined source/shot contribution while retaining the stronger sparse-axis
encoding. Every other condition stays fixed.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10_adamw_axis_source107_shot109
```

If the combined candidate remains below 15 dB, `axis_source107` and
`axis_shot109` isolate the two changes. They are prepared in advance but must
start only after the combined run completes. Each changes one axis from the
scalar-base configuration and retains every fixed comparison condition.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_source107 base115_profiles10_adamw_axis_shot109
```

The combined source/shot candidate completed at 13.6648 dB. The source-only
and shot-only candidates completed at 13.6129 and 13.4934 dB respectively.
All three passed the fixed-v3 input-lock, artifact-hash and full-coverage checks,
but none improved the 13.7633 dB all-axis Nyquist result.

## Independent-seed diagnostic; ensemble abandoned

A target-informed diagnostic over the nine completed ten-profile NeRSI
predictions found complementary errors: the best individual scored 13.7633 dB,
the equal-weight best five scored 14.7869 dB, and a nonnegative sum-to-one
weight fit on full T scored 14.8615 dB. These values are exploratory leakage,
not independent generalization evidence, and no ensemble artifact was produced.

`base115_profiles10_adamw_axis_nyquist_seed102.yaml` and `seed103.yaml` repeat
the current best architecture with independent model-initialization and sampling
streams `(102, 202)` and `(103, 203)`. Every member individually retains exactly
363,269 parameters, 50,000 updates, ten profiles/update, AdamW with zero weight
decay, LR 0.001, EMA 0.999, Global RMS, no time shear and the complete fixed-v3
evaluation population. An ensemble has multiple times the total stored parameters,
optimizer updates and supervised presentations of one member; those totals must be
reported explicitly rather than described as equal-compute with the width128 GNN.
After these two already-started runs, the user rejected the ensemble as an unfair
comparison. No ensemble prediction or adopted ensemble result will be produced;
all subsequent candidates must be a single 363,269-parameter NeRSI at inference.
Both independent runs completed and passed the fixed-v3 lock, hash and coverage
checks. Seed 102 scored 13.3183 dB and seed 103 scored 13.6092 dB; neither replaces
the seed-101 all-axis Nyquist single-model best of 13.7633 dB.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_nyquist_seed102 \
  base115_profiles10_adamw_axis_nyquist_seed103
```

The next single-model candidate,
`base115_profiles10_adamw_nyquist_bandlimit.yaml`, keeps the scalar base 1.15
mapping but masks components above each fixed index lattice's Nyquist limit.
Unlike changing the exponential bases, this removes aliased columns. It adds no
parameters and retains every training, normalization and evaluation condition.

```bash
bash scripts/run_c3_v3_nersi_budget.sh base115_profiles10_adamw_nyquist_bandlimit
```

The hard-bandlimit run completed at 13.3145 dB and passed the fixed-v3 lock,
artifact-hash and full-coverage checks. It is rejected because it is 0.4488 dB
below the current single-model best.

## Single-model profile-embedding candidate

`base115_profiles10_adamw_axis_nyquist_profile_embedding.yaml` augments the
fixed Fourier coordinate mapping with one learned 32-channel embedding per
fixed profile ID. The embedding is indexed only by the three integer profile
coordinates; it does not read T amplitudes or change the O mask. This targets
the main information gap relative to the GNN: small profile-local deviations
need not all pass through a 40-wide global coordinate bottleneck.

The parameter budget is reallocated to encoder width 41, latent channels 11
and decoder channels `[40, 16, 15]`. The complete model, including all 4,096
embedding rows, has exactly 363,269 trainable parameters; there is no dummy or
frozen padding. It remains one model at inference. The fixed ten profiles/update,
50,000 updates, AdamW, LR 0.001, EMA 0.999, Global RMS, no shear, seeds and
full-T evaluator are unchanged. One profile has no O trace in this mask, so its
embedding receives no supervised update; it is retained in full-target coverage.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_nyquist_profile_embedding
```

The profile-embedding run completed at 13.0708 dB and passed the fixed-v3
lock, artifact-hash and full-coverage checks. It is rejected. The extra local
degrees of freedom reduced generalization from each profile's O traces to its
missing T traces.

## Rank-2 separable coarse-latent candidate

`base115_profiles10_adamw_axis_nyquist_rank2_latent.yaml` generates each coarse
`48×4` latent channel as a sum of two learned time-by-receiver outer products.
This uses the expected coherence of seismic profiles to replace a dense coarse
map with a low-rank representation. The saved parameters are reallocated to
encoder width 80, latent channels 20 and decoder channels `[40, 17, 16]`.
The resulting single model has exactly 363,269 trainable parameters. All data,
normalization, optimizer, LR, update, trace-presentation, seed, EMA, no-shear
and evaluation conditions remain unchanged.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_nyquist_rank2_latent
```

The rank-2 run completed at 13.7494 dB and passed the fixed-v3 lock, artifact-hash
and full-coverage checks. Its fit improved, but target SNR remained 0.0139 dB
below the single-model best, so it is not adopted.

## Learned 128-component temporal-basis candidate

`base115_profiles10_adamw_axis_nyquist_temporal_basis128.yaml` makes the decoder
predict 128 temporal coefficients per receiver-y trace and reconstructs 384
samples with one shared trainable temporal basis. The basis starts from the
first 128 orthonormal DCT components and is then optimized only from O. This
matches the observed low-pass diagnostic while letting the basis adapt instead
of applying a fixed post-filter.

The budget is reallocated to encoder width 78, latent channels 22 and decoder
channels `[39, 20, 10]`. Including the 49,152 basis parameters, the single model
has exactly 363,269 trainable parameters. Ten profiles/update, 50,000 updates,
AdamW, LR 0.001, EMA 0.999, Global RMS, no shear, seeds and evaluation remain fixed.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_nyquist_temporal_basis128
```

The temporal-basis run completed at 11.8820 dB and passed the fixed-v3 lock,
artifact-hash and full-coverage checks. It is rejected because the 128-component
coefficient representation underfit the observed waveforms.

## Fourier48 axis-Nyquist candidate

`fourier48_profiles10_adamw_axis_nyquist.yaml` restores the standard time-domain
decoder and increases each coordinate axis from 40 to 48 Fourier components.
Its three exponential bases are chosen so component 48 equals the corresponding
16-, 32- or 8-point index lattice Nyquist frequency. The budget is reallocated
to encoder width 40, latent channels 24 and decoder channels `[33, 17, 16]`.
The resulting single model has exactly 363,269 trainable parameters; all fixed
training, normalization, seed, no-shear and evaluation conditions are retained.

```bash
bash scripts/run_c3_v3_nersi_budget.sh fourier48_profiles10_adamw_axis_nyquist
```

The Fourier48 run completed at 13.5749 dB mean trace SNR and passed the fixed-v3
input-lock, artifact-hash and full-target coverage checks. It is rejected because
the additional Fourier components reduced accuracy relative to the 40-component
single-model best.

## Cartesian CMP and half-offset coordinate candidate

`base115_profiles10_adamw_axis_nyquist_cartesian.yaml` changes only the three
profile coordinates of the current 13.7633 dB single-model best. The regular
source-line, shot and receiver-x indices are replaced by normalized physical
CMP-x, CMP-y at the first receiver-y position, and half-offset-x. Receiver-y
remains the generated decoder axis, with its physical half-offset-y grid stored
in the checkpoint metadata. The input lock, model parameter count, Fourier
bases, ten profiles/update, 50,000 updates, AdamW, LR 0.001, EMA 0.999, Global
RMS, seeds, no-shear condition and evaluator are unchanged.

```bash
bash scripts/run_c3_v3_nersi_budget.sh \
  base115_profiles10_adamw_axis_nyquist_cartesian
```

The run completed at 13.6524 dB mean trace SNR, 0.1110 dB below the index-space
baseline. It passed the fixed-v3 input-lock, artifact-hash and full-target
coverage checks. Cartesian CMP and half-offset coordinates alone are therefore
not adopted.
