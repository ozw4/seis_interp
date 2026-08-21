# SEG C3 Narrow-Azimuth

This tracked directory contains provenance and source-manifest information only. The four SEG-Y files are downloaded to an external data root and are never committed to this repository.

## Storage layout

Set `SEIS_INTERP_DATA_ROOT` to a directory outside the repository. The downloader creates the following layout:

```text
${SEIS_INTERP_DATA_ROOT}/
└── external/
    └── seg_c3_na/
        ├── SEG_C3NA_ffid_2-1200.sgy
        ├── SEG_C3NA_ffid_1201-2400.sgy
        ├── SEG_C3NA_ffid_2401-3600.sgy
        ├── SEG_C3NA_ffid_3601-4781.sgy
        └── download.lock.yaml
```

`download.lock.yaml` is generated outside Git. Its checksums are trust-on-first-use values calculated from the downloaded bytes, not checksums published by SEG. It records the exact byte count and SHA-256 checksum observed for each local file, plus the checksum of this tracked source manifest. It contains no absolute host path.

## Download on the host

Install the package once, configure the external root, and run the thin script from the repository root:

```bash
python -m pip install -e .
export SEIS_INTERP_DATA_ROOT=/absolute/path/to/seis_interp_data
./scripts/download_seg_c3_na.sh
```

Interrupted `.part` files are resumed with an HTTP Range request. To discard existing complete and partial files and start again:

```bash
./scripts/download_seg_c3_na.sh --force
```

Verify the four files against the generated lock:

```bash
./scripts/verify_seg_c3_na.sh
```

The equivalent CLI commands are:

```bash
seis-interp data download seg_c3_na
seis-interp data verify seg_c3_na
```

The Dev Container mounts the configured external root read-only. Download on the host before opening or restarting the container.

## Source and use conditions

The tracked manifest records the SEG Wiki landing page and the four public S3 object URLs. The source page describes the files as SEG-Y Rev. 1. Confirm the applicable source terms before a formal experiment or redistribution; this repository does not redistribute the data.
