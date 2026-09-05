# Data management

Data are separated by origin, processing stage, and artifact responsibility.

- `external/`: externally obtained datasets and their manifests. SEG C3 NA belongs here.
- `interim/`: source-derived trace data and reproducible intermediate conversions.
- `processed/`: dataset partitions, normalization, separately generated interpolation masks, model-independent benchmark cases, and case-bound volume indices.

A prepared dataset partition, its masks, benchmark cases, and volume indices use this layout:

```text
data/processed/<dataset>/<partition-id>/
  trace_split.parquet
  normalization.json
  preparation.json
  masks/<mask-id>/
    observation_mask.parquet
    interpolation_mask.json
  cases/<case-id>/
    benchmark_case.json
  volumes/<volume-id>/
    volume_index.parquet
    volume.json
```

`trace_split.parquet` assigns dataset partitions. Each mask directory independently assigns `observed` and `evaluation_target` roles within one partition; it does not replace or modify the split table.

A benchmark case does not copy its interim, partition, or mask artifacts. `benchmark_case.json` records their exact file hashes without absolute paths, and callers provide the current directories for hash verification before use. Generated case artifacts, like other processed data, must not be committed to Git.

`volume_index.parquet` stores only the selected trace-to-5D-cell mapping and contains no amplitudes. `volume.json` stores the zero-based half-open crop contract and binds the volume to one `benchmark_case.json` hash. The current dense-only contract rejects a crop containing an incomplete shot or any other missing cell. Volume artifacts are generated data and must not be committed to Git.

In the Dev Container, `SEIS_INTERP_DATA_ROOT` is `/workspace/data`. SEG C3 NA is therefore stored at `/workspace/data/external/seg_c3_na/`.

The manifest and documentation are tracked. Raw SEG-Y files, generated download locks, interim data, processed partitions, mask, benchmark-case, and volume artifacts, large arrays, and other local data products are ignored by Git and must not be committed.
