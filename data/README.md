# Data management

Data are separated by origin and processing stage.

- `external/`: externally obtained datasets and their manifests. SEG C3 NA belongs here.
- `interim/`: reproducible caches and intermediate conversions. This directory is generated locally and ignored by Git.
- `processed/`: QC-approved inputs for training and evaluation. This directory is generated locally and ignored by Git.

Do not commit raw SEG-Y files, large arrays, or machine-specific paths. Store the external data outside the repository and configure its host path through `.devcontainer/.env` or `SEIS_INTERP_DATA_ROOT`.
