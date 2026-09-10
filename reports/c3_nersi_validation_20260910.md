# C3 NeRSI fixed validation result — 2026-09-10

The authoritative setup and selection are in
[Study 034](../studies/study_034_c3_nersi_baseline/README.md).

| Candidate | Parameters | Target S/N | RMSE | Observed model RMSE | Fit + prediction | Peak GPU allocated |
|---|---:|---:|---:|---:|---:|---:|
| A | 3,459,793 | 15.6996 dB | 1.6306 | 1.2170 | 160.4926 s | 2.338 GB |
| B | 3,459,793 | 13.8712 dB | 2.0126 | 1.5980 | 166.5564 s | 2.338 GB |
| **C** | **7,728,697** | **15.9189 dB** | **1.5899** | **1.0407** | **170.0764 s** | **4.307 GB** |
| D | 7,728,697 | 14.6172 dB | 1.8470 | 1.3131 | 160.1146 s | 4.307 GB |

- evaluation domain: 58,999 traces / 22,655,616 samples
- common reference energy: 2,237,830,414.4523
- Candidate C error energy: 57,271,077.1562
- every saved prediction independently re-scored and preserved observed samples exactly
- selected result: Candidate C; +0.2193 dB over A
- test partition unused
