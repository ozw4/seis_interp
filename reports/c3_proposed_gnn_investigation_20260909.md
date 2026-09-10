# C3 proposed GNN baseline diagnostics — 2026-09-09

## Conditions

- QC suite; case `c3_benchmark_validation_random_trace_80_seed142`
- train: 1,146,366 canonical traces, time `[0,384)`, global RMS 28.6279
- baseline: width 32, 27,333 parameters, two rounds, two neighbors per relation, query batch 4, AdamW `1e-4`, seed 20260908, 200 updates
- evaluation: 58,999 targets / 22,655,616 samples; test unused

## Results

- independent final prediction: 0.0144 dB, RMSE 9.9222; zero-fill 0.0000 dB / 9.9386
- trainer / independent S/N: 0.014369785017596115 / 0.014369424212006265 dB; fixed energy tolerance passed
- CPU restoration on 1,143 queries / 438,912 samples: normalized RMSE `5.2952e-6`, max difference `8.2391e-5`, physical relative L2 0.0005; all limits passed
- four-query, 1,000-update training fit: 16.5758 dB at learning rate `1e-4`; 20.0908 dB at `1e-3`
- exact-index search matched brute-force arrays where brute force completed; train 32-query time 6.8092 s vs 0.3448 s, validation 512-query time 1.8917 s vs 0.6181 s
- width-32 GPU peak allocated: train query 4/32/128 = 0.2059/1.0815/4.1606 GB; validation 32/128/512 = 0.1038/0.2132/0.5261 GB

## Decision

- Do not adopt the 200-update baseline; retain its audit as valid.
- Use exact-index search for subsequent conditions.
