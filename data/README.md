# Data management

Data are separated by origin and processing stage.

- `external/`: externally obtained datasets and their manifests. SEG C3 NA belongs here.
- `interim/`: reproducible caches and intermediate conversions. This directory is generated locally and ignored by Git.
- `processed/`: QC-approved inputs for training and evaluation. This directory is generated locally and ignored by Git.

In the Dev Container, `SEIS_INTERP_DATA_ROOT` is `/workspace/data`. SEG C3 NA is therefore stored at `/workspace/data/external/seg_c3_na/`.

The manifest and documentation are tracked. Raw SEG-Y files, generated download locks, large arrays, and other local data products are ignored by Git and must not be committed.
