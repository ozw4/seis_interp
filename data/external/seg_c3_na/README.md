# SEG C3 Narrow-Azimuth

This directory contains the tracked source manifest and the local SEG C3 Narrow-Azimuth files used by the POC. Raw SEG-Y files and the generated download lock are ignored by Git.

## Storage layout

The Dev Container uses:

```text
SEIS_INTERP_DATA_ROOT=/workspace/data
```

The resulting dataset layout is:

```text
/workspace/data/external/seg_c3_na/
├── README.md
├── manifest.yaml
├── SEG_C3NA_ffid_2-1200.sgy
├── SEG_C3NA_ffid_1201-2400.sgy
├── SEG_C3NA_ffid_2401-3600.sgy
├── SEG_C3NA_ffid_3601-4781.sgy
└── download.lock.yaml
```

`download.lock.yaml` records trust-on-first-use byte counts and SHA-256 checksums calculated from the downloaded files. These are local observations, not checksums published by SEG.

## Download and integrity verification

From the repository root:

```bash
export SEIS_INTERP_DATA_ROOT=/workspace/data
./scripts/download_seg_c3_na.sh
./scripts/verify_seg_c3_na.sh
```

Interrupted `.part` files are resumed with an HTTP Range request. To discard existing complete and partial files and start again:

```bash
./scripts/download_seg_c3_na.sh --force
```

## SEG-Y content inspection

Run the checked-in inspection script rather than a temporary Python command:

```bash
./scripts/inspect_seg_c3_na.sh
```

The inspection reports, for every declared SEG-Y file:

- file size, trace count, samples per trace, sample interval, record length, and sample format code
- actual FFID range and traces per FFID, checked against the manifest range
- source, receiver, midpoint, offset, and azimuth ranges
- coordinate units and coordinate scalar values
- delay recording time
- statistics from evenly spaced amplitude samples, including finite-value ratio, zero ratio, range, mean, standard deviation, and RMS

Midpoint, offset, and azimuth come from `seis_interp.processing.geometry` and follow `docs/coordinate_conventions.md`.

The default is 32 sampled traces per file. Increase or reduce it explicitly:

```bash
./scripts/inspect_seg_c3_na.sh --sample-traces 64
```

Machine-readable output is available for a recorded QC result:

```bash
./scripts/inspect_seg_c3_na.sh --json > seg_c3_na_inspection.json
```

The script exits with status 1 when it detects a missing file, manifest FFID mismatch, invalid sample metadata, all-zero geometry, non-finite sampled amplitudes, or all-zero sampled amplitudes.

The equivalent CLI command is:

```bash
python -m seis_interp.cli data inspect seg_c3_na
```

## Source and use conditions

The tracked manifest records the SEG Wiki landing page and the four public S3 object URLs. The source page describes the files as SEG-Y Rev. 1. Confirm the applicable source terms before a formal experiment or redistribution; this repository does not redistribute the data.
