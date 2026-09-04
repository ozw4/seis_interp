# Data management

Data are separated by origin, processing stage, and artifact responsibility.

- `external/`: externally obtained datasets and their manifests. SEG C3 NA belongs here.
- `interim/`: source-derived trace data and reproducible intermediate conversions.
- `processed/`: dataset partitions, normalization, and separately generated interpolation mask artifacts.

A prepared dataset partition and its masks use this layout:

```text
data/processed/<dataset>/<partition-id>/
  trace_split.parquet
  normalization.json
  preparation.json
  masks/<mask-id>/
    observation_mask.parquet
    interpolation_mask.json
```

`trace_split.parquet` assigns dataset partitions. Each mask directory independently assigns `observed` and `evaluation_target` roles within one partition; it does not replace or modify the split table.

In the Dev Container, `SEIS_INTERP_DATA_ROOT` is `/workspace/data`. SEG C3 NA is therefore stored at `/workspace/data/external/seg_c3_na/`.

The manifest and documentation are tracked. Raw SEG-Y files, generated download locks, interim data, processed partitions, mask artifacts, large arrays, and other local data products are ignored by Git and must not be committed.
