# study_017_all_ffid_neighbor_inpainter

## Status

`draft`

## Research question

Can one geometry-conditioned temporal network reconstruct every amplitude-eligible SEG C3
Narrow-Azimuth validation trace with an oracle per-trace unit-RMS global S/N strictly greater
than 15 dB when only training-trace amplitudes are available as neighbors?

This is the leakage-safe successor to the coordinate-only `study_016_all_ffid_siren` condition.
It preserves that study's survey scope and trace split, but changes the model after staged SIREN,
Fourier, profile-wise, low-rank, moveout, and non-local retrieval diagnostics did not reach 15 dB.

## Inputs and split

The study uses all four locked SEG C3 NA sources and all 4,780 amplitude-eligible FFIDs, including
incomplete line-end FFIDs. FFID 1746 remains wholly excluded by the prepared amplitude-quality
rule. The prepared seed-42 per-FFID split contains 1,842,102 train, 114,492 validation, and 346,886
test traces. Before model use, the 15 repeated physical cells are canonicalized across all splits
by retaining only the lowest `array_row`. The effective split is therefore 1,842,090 train,
114,490 validation, and 346,885 test traces, with all 4,780 FFIDs still represented. Time samples
from one canonical trace never cross split boundaries.

Validation and test coordinates are known query metadata. Only train amplitudes may populate a
neighbor channel. Validation targets are used solely for checkpoint selection and metrics; test
targets are neither optimized nor evaluated.

## Model condition

For each target coordinate, the input contains 104 ordered positions on the natural acquisition
lattice. The neighborhood spans relative receiver-x index +/-1 (40 m), source-shot index +/-2
(80 m per shot), and relative receiver-y index +/-3 (40 m), excluding the target position itself.
All source-shot offsets stay on the same source-x line. A missing or non-train neighbor has zero
amplitude and a false availability channel. Incomplete FFIDs therefore need no imputation before
the model.

Each available train trace is divided by its own RMS. The network receives 104 unit-RMS amplitude
channels, 104 availability channels, three target coordinates fitted only from train geometry
`[relative_receiver_x, source_y, relative_receiver_y]`, and one normalized time channel. A
kernel-15 temporal stem of width 128 feeds eleven gated depthwise residual blocks with dilations
`[1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]`, followed by a scalar trace head. The fixed condition has
983,041 parameters.

The `normalization` mapping in `config.yaml` locks compatibility with the reused Study 016
prepared artifact; it is not the model-input transform. The inpainter's three target coordinates
use their own train-only min/max transform declared under `model.target_coordinate_scaling`.

## Training and validation

AdamW runs 2,500 updates with seed 42, batch size 96, learning rate `5e-4`, weight decay `1e-5`,
and cosine decay to `1.5e-5`. Five percent of otherwise available neighbors are randomly dropped.
The loss is trace MSE plus `0.1` times first-time-difference MSE, with gradient norm clipped at 1.
CUDA training uses bfloat16 autocast while predictions and energy accumulation remain float32 and
float64 respectively.

Validation runs after the first update, every 500 updates, and at the final update. The checkpoint
is selected by raw prediction S/N; normalizing a prediction by its own RMS is diagnostic only and
cannot select the checkpoint. A deterministic seed-44 sample of 114,492 train traces supplies a
leave-one-out training audit after selection and cannot affect the checkpoint. The primary metric is
`oracle_per_trace_unit_rms_global_snr_db`, computed as
`10 log10(sum(target_unit^2) / sum((target_unit - prediction_raw)^2))`.

The success rule was fixed before the formal run:

```text
oracle_per_trace_unit_rms_global_snr_db > 15 dB
```

## Reproduction

From the repository root, prepare the locked survey and split if they are not already present:

```bash
python -m seis_interp.cli data prepare-c3-survey \
  --manifest data/external/seg_c3_na/manifest.yaml \
  --data-root "$SEIS_INTERP_DATA_ROOT" \
  --output data/interim/c3_na/all_ffids

python -m seis_interp.cli data prepare-baseline \
  --config studies/study_017_all_ffid_neighbor_inpainter/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc
```

Then run the formal training condition:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)_all_ffids"

python -m seis_interp.cli train neighbor-inpainter \
  --config studies/study_017_all_ffid_neighbor_inpainter/config.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc \
  --output "runs/study_017_all_ffid_neighbor_inpainter/$RUN_ID" \
  --device cuda:0
```

## Expected outputs

The immutable run contains the resolved configuration, locked input hashes and acquisition
geometry, seed and environment metadata, complete validation history and energy totals, leakage
audit, neighbor-availability statistics, and a loadable best checkpoint.

## Acceptance criteria

- All 4,780 eligible FFIDs contribute train and validation traces; incomplete FFIDs are retained.
- Duplicate physical cells are canonicalized before routing targets or neighbor amplitudes, using
  only the deterministic lowest-`array_row` rule.
- The target offset is absent, source-x lines never mix, and every supplied amplitude comes from
  the train split.
- Test and excluded targets do not affect optimization, model selection, or reported validation.
- Model selection uses only raw oracle per-trace unit-RMS validation global S/N.
- The best validation metric is strictly greater than 15 dB.
- Run provenance is complete and the repository quality gates pass.

## Limitations

This is not a coordinate-only SIREN: it conditions on neighboring train waveforms and therefore
answers a different, more operational interpolation question than Study 016. Per-trace target
normalization is an oracle waveform diagnostic; a held-out trace's physical RMS still needs a
separate gain model. The study uses one seed and one survey. Canonicalizing duplicate physical
cells removes 15 raw rows from the model condition, so its effective trace counts differ slightly
from the reusable prepared split artifact.

## Current result

The reusable implementation is being validated against the successful staged proxy. No immutable
Study 017 run is recorded yet.

Historical rationale belongs in [`decisions.md`](decisions.md).
