# study_all_ffid_temp

## Purpose

This is a scratch workspace for repeated survey-wide SIREN experiments using the prepared
`all_ffids_per_ffid_random_split` dataset. It is not a numbered study and its latest output is
not an immutable research record.

Edit [`config.yaml`](config.yaml) directly between runs. The initial values match
`study_016_all_ffid_siren`, with inherited random-point settings written explicitly so that the
available knobs are visible in one file.

## Fixed prepared-data contract

The existing processed dataset was prepared with the following values. Keep them unchanged when
reusing it:

- `project.random_seed`
- `sampling.random_trace_holdout_fraction`
- `sampling.validation_fraction_of_holdout`
- `sampling.split_scope`
- `normalization.coordinates`
- `normalization.amplitude`

Changing one of those values requires preparing a different processed dataset. Model and training
values can be changed without rebuilding the split or normalization.

## Run

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
