# study_all_ffid_temp

## Purpose

This is a scratch workspace for repeated survey-wide SIREN experiments using the prepared
`all_ffids_per_ffid_random_split_amplitude_qc` dataset. It is not a numbered study and its latest
output is not an immutable research record.

Edit [`config.yaml`](config.yaml) directly between runs. The initial values match
`study_016_all_ffid_siren` except for the deliberately enabled per-trace RMS training target.
Inherited random-point settings are written explicitly so that the available knobs are visible
in one file.

## Fixed prepared-data contract

The existing processed dataset was prepared with the following values. Keep them unchanged when
reusing it:

- `project.random_seed`
- `sampling.random_trace_holdout_fraction`
- `sampling.validation_fraction_of_holdout`
- `sampling.split_scope`
- `sampling.trace_amplitude_filter`
- `normalization.coordinates`
- `normalization.amplitude`

Changing one of those values requires preparing a different processed dataset. Model and training
values can be changed without rebuilding the split or normalization.

## Trace-amplitude eligibility

Amplitude eligibility is determined from raw interim traces before split assignment and before
normalization is fitted. This rule does not depend on `training.amplitude_scaling`:

- exclude a trace when all 625 samples are exactly zero;
- exclude a trace when any sample has absolute amplitude greater than `1.0e4`;
- never clip amplitudes or remove individual time samples.

For the locked SEG C3 NA input this excludes exactly 544 traces: 107 all-zero traces and 437
excessive-amplitude traces, all belonging to FFID 1746. The remaining 2,303,480 traces across
4,780 FFIDs are split and normalized. Excluded rows remain recorded with the `excluded` split
label but are used by neither training nor validation.

## Training-target amplitude scaling

`training.amplitude_scaling` is independent of the fixed prepared-data normalization above and
accepts two values:

- `train_global_rms` divides every target by the RMS fitted from all training samples. This is
  the physical-amplitude interpolation contract used by Study 016.
- `per_trace_rms` divides each complete trace by that trace's own RMS at training time. The
  scratch config selects this value initially.

For `per_trace_rms`, validation traces are also divided by their own target RMS so that early
stopping measures waveform fit in the unit-RMS target domain. This is an oracle-normalized
diagnostic: the validation metric is not directly comparable with Study 016's global-RMS S/N,
and a held-out trace's physical amplitude cannot be reconstructed because its RMS is unavailable
at inference. The checkpoint records the scaling name but cannot supply that unknown scale. Do
not promote this condition as a physical-amplitude interpolation result without a separate gain
model fitted only from training data. Generated metrics, run metadata, and the checkpoint label
this metric domain as `oracle_per_trace_unit_rms`.

## Run

Prepare the amplitude-filtered split once, or regenerate it after changing a fixed prepared-data
condition:

```bash
python -m seis_interp.cli data prepare-baseline \
  --config studies/study_all_ffid_temp/config.yaml \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_per_ffid_random_split_amplitude_qc \
  --overwrite
```

From the repository root:

```bash
python scripts/run_study_all_ffid_temp.py
```

To override only the execution device:

```bash
python scripts/run_study_all_ffid_temp.py --device cuda:1
```

The runner trains into a staging directory. After a successful run it replaces
`runs/study_all_ffid_temp/current`; if training fails, the previous successful output remains.
The output contains the normal `config.resolved.yaml`, `inputs.lock.json`, `metrics.json`,
`run.json`, and `artifacts/best.pt` files.

Do not use this overwriteable directory as formal evidence. Copy an accepted condition into a
numbered study and run it with an immutable run ID when the result needs to be retained.
