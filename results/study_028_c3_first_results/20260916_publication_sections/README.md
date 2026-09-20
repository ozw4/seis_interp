# Study 028 publication sections

Three fixed central sections re-rendered from the adopted Study 028 native
predictions. Source runs, section geometry, and output checksums are recorded
in `manifest.json`. Original adopted figures remain in their source directory.

PNG: 178 × 240 mm, 600 dpi. PDF: the same physical dimensions, with raster
seismic panels and vector text/axes. Titles are omitted; spatial and time
labels and the amplitude colorbar are shared. All panels use `seismic`,
nearest-neighbor rendering, and the original symmetric amplitude limits
±52.31291854858398. No prediction or residual amplitudes are rescaled.

Arial was requested but is unavailable in the export environment; these files
use DejaVu Sans. All other supplied font sizes and line widths are applied.

Panel identities, in row-major order:

| Row | Left | Right |
| --- | --- | --- |
| 1 | (a) Reference | (b) Masked observed input |
| 2 | (c) POCS | (d) Reference minus POCS |
| 3 | (e) DRR | (f) Reference minus DRR |
| 4 | (g) SIREN-5D | (h) Reference minus SIREN-5D |
| 5 | (i) CCNet-5D | (j) Reference minus CCNet-5D |
| 6 | (k) Relational trace graph | (l) Reference minus relational trace graph |

The dark saturated graph panels retain the original extreme predictions;
the colorbar extensions indicate clipping. Source-line panel (b) is empty
because this fixed section contains no observed traces.

Reproduction from the repository root, using a new output directory:

```bash
.venv/bin/python scripts/export_study028_sections.py --output-dir results/study_028_c3_first_results/<new_export_id>
```

To export with Arial, supply an available Arial TTF using `--font-file /path/to/arial.ttf`.
