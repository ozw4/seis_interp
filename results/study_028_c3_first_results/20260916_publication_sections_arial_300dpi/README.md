# Study 028 publication sections

PNG and PDF exports: 178 × 240 mm, 300 dpi, Arial, `cmap='seismic'`,
`interpolation=None`. Python `None` uses Matplotlib's configured interpolation
mode; the resolved mode is recorded in `manifest.json`.
Titles are omitted and x/y labels and the amplitude colorbar are shared.
The original section geometry, predictions, residuals, and symmetric amplitude
limits are retained. Source runs and output checksums are in `manifest.json`.

| Row | Left | Right |
| --- | --- | --- |
| 1 | (a) Reference | (b) Masked observed input |
| 2 | (c) POCS | (d) Reference minus POCS |
| 3 | (e) DRR | (f) Reference minus DRR |
| 4 | (g) SIREN-5D | (h) Reference minus SIREN-5D |
| 5 | (i) CCNet-5D | (j) Reference minus CCNet-5D |
| 6 | (k) Relational trace graph | (l) Reference minus relational trace graph |

Arial was obtained from the [Microsoft core fonts distribution on SourceForge](https://sourceforge.net/projects/corefonts/files/the%20fonts/final/arial32.exe/download).
Archive SHA-256: `85297a4d146e9c87ac6f74822734bdee5f4b2a722d7eaa584b7f2cbf76f478f6`.
Regular, bold, italic, and bold italic fonts are installed in the user's
`.local/share/fonts/arial/` directory, outside the repository.

Reproduce from the repository root with a new output directory:

```bash
.venv/bin/python scripts/export_study028_sections.py --output-dir results/study_028_c3_first_results/<new_export_id>
```
