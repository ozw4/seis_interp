# Data management

Data are separated by origin and processing stage.

- `external/`: externally obtained datasets and their manifests. SEG C3 NA belongs here.
- `interim/`: reproducible caches and intermediate conversions. This directory is generated locally and ignored by Git.
- `processed/`: QC-approved inputs for training and evaluation. This directory is generated locally and ignored by Git.

The tracked `data/` tree contains manifests and documentation only. Real data live outside the repository beneath `SEIS_INTERP_DATA_ROOT` using the same processing-stage names. For example, SEG C3 NA is stored at `${SEIS_INTERP_DATA_ROOT}/external/seg_c3_na/`.

Do not commit raw SEG-Y files, large arrays, generated download locks, or machine-specific paths. Configure the host root through `.devcontainer/.env` or `SEIS_INTERP_DATA_ROOT`.
