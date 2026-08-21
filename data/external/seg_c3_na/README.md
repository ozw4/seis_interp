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

## Download inside the Dev Container

On the host, create the external data root and print its absolute path:

```bash
mkdir -p "$HOME/seis_interp_data"
realpath "$HOME/seis_interp_data"
```

Copy `.devcontainer/.env.example` to `.devcontainer/.env` and assign the printed path to `SEIS_INTERP_DATA_ROOT`. Rebuild the Dev Container so the directory is mounted writable at `/home/dcuser/data`.

Inside the rebuilt container:

```bash
echo "$SEIS_INTERP_DATA_ROOT"
./scripts/download_seg_c3_na.sh
./scripts/verify_seg_c3_na.sh
```

The first command should print `/home/dcuser/data`.

## Download directly on the host

Install the package once and configure the external root:

```bash
export SEIS_INTERP_DATA_ROOT="$HOME/seis_interp_data"
python -m pip install -e .
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

## Source and use conditions

The tracked manifest records the SEG Wiki landing page and the four public S3 object URLs. The source page describes the files as SEG-Y Rev. 1. Confirm the applicable source terms before a formal experiment or redistribution; this repository does not redistribute the data.
