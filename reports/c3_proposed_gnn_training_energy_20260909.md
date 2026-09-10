# C3 proposed GNN training-energy distribution — 2026-09-09

## Conditions

- QC canonical train: 1,146,366 traces x 384 samples, time `[0,384)`
- global RMS: 28.62792245506524; zero traces retained: 1,195; maximum absolute amplitude: 4,298.7656
- eight highest-RMS traces checked against all 625 original SEG-Y samples

## Results

| Offset band | Trace count | Training-energy share |
|---|---:|---:|
| 0-40 m | 4,760 | 69.1804% |
| 40-80 m | 14,279 | 13.8692% |
| 80-160 m | 47,600 | 4.2609% |
| 160-320 m | 85,679 | 2.8888% |
| 320-640 m | 152,016 | 3.4285% |
| 640-1,280 m | 287,805 | 3.1386% |
| 1,280-2,560 m | 498,995 | 2.9982% |
| >=2,560 m | 55,232 | 0.2354% |

- samples 0-64 / 64-128 / 128-192 / 192-256 / 256-320 / 320-384 contain 92.6812% / 3.6546% / 1.9181% / 1.3703% / 0.2847% / 0.0911% of training energy
- top 11,464 RMS-ranked traces contain 79.2234% of total energy
- all 5,000 checked SEG-Y/interim samples matched bitwise; peak 4,298.4102-4,298.7070 at 0.0960 s
- total energy: 360,773,131,061.13715

## Decision

- Retain the data; this diagnostic does not authorize further exclusion.
