# Study 042 adopted-result sections

Current adopted results, as requested, rather than automatic maximization across
all tuning runs. POCS, DRR and CCNet use Study 042's formal comparison;
NeRSI uses Study 042's no-time-shear adoption lock; GNN uses Study 044's
adopted EMA prediction. These are target-informed results, not independent test
estimates. Exact run paths, hashes, metrics and input conditions are in `manifest.json`.

| Method | Mean target-trace SNR (dB) |
| --- | ---: |
| POCS | 13.6988 |
| DRR | 10.5330 |
| NeRSI | 15.5856 |
| CCNet-5D | 5.3555 |
| Proposed GNN | 18.2795 |

The three images vary receiver y, shot in line, and source line respectively.
Other spatial indices are fixed at the center of the v3 volume before inspecting
amplitudes. Reference and masked input occupy the first two columns of row 1;
five predictions occupy row 2; reference-minus-prediction residuals occupy row 3.
All twelve panels have equal dimensions. Unlike Study 028, this comparison uses
NeRSI rather than SIREN.

Each figure is 178 × 142 mm at 1200 dpi, with Arial, 10 pt bold panel letters,
0.5 pt frames, `seismic`, `interpolation='None'`, axis values without tick marks,
and no colorbar. One symmetric contrast limit, derived from the pooled reference
99th percentile over these three v3 sections, applies to every panel. No amplitude
normalization is applied. The Study 028 numerical contrast limit is not reused.

PNG and PDF files are supplied. Reproduce from the repository root, specifying
a new output directory:

```bash
.venv/bin/python scripts/export_study042_sections.py --output-dir results/study_042_c3_v3_five_method_comparison/<new_export_id>
```
