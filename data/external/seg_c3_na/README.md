# SEG C3 Narrow-Azimuth

This directory contains only provenance and manifest information. The SEG-Y data files are not redistributed by this repository.

Set the host data location in `.devcontainer/.env`:

```dotenv
SEISMIC_DATA_ROOT=/absolute/path/to/seg_c3_na
```

Inside the Dev Container the dataset is available read-only at `/home/dcuser/data`, and the application sees that path through `SEIS_INTERP_DATA_ROOT`.

Before a formal experiment, update `manifest.yaml` with the exact file names, retrieval time, version information, license terms, and SHA-256 checksums.
