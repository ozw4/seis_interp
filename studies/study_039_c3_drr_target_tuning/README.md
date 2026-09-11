# DRR target-based parameter tuning

Status: `target_achieved`。全 target の物理振幅 global SNR は **12.1328 dB** で、目標の11.3 dB超えを達成した。
study_036 の凍結 DRR (8.5207 dB) に対して **+3.6121 dB**。
元の formal config、入力 artifact、mask seed 42、全384時間サンプルと解析領域を保持する。
再構成実装は既存の DRR を使用し、変更は rank、damping、iteration、frequency、spatial window の設定のみ。

ユーザーの明示指示により O 内 validation は使用せず、T の真値による SNR を使って調整する。
これは target leakage を含む探索であり、独立した test 性能や HPO なし他手法への優位性とは解釈しない。
振幅を再構成へ入力するのは O のみである。
最初の探索では空間的に分散した非重複4ブロックを固定し、全時間を評価する。
地域限定の pilot SNR は目標達成の証拠とはせず、既存CLIで全領域・全 T を再構成し確認する。
全候補、探索結果、計算時間を runs に記録する。
入力・評価式・全 target coverage・観測値保持の比較条件を維持する。
study_036 とは調整budgetが異なるため、同一計算量・同一探索budgetとは主張しない。

## 実行

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/tune_study_039_drr.py --output runs/study_039_c3_drr_target_tuning/<UTC>_<SHA>_pilot --workers 4
```

初回pilotは `runs/study_038_c3_drr_target_tuning/20260911T102815Z_5829a11_pilot`。
高rank pilotは `runs/study_039_c3_drr_target_tuning/20260911T103035Z_5829a11_pilot_high_rank`。
全12候補を保存し、pilot SNR最大の設定を全領域確認に選択した。

## 探索結果と全領域確認

Pilotは全時間384サンプル、Tのうち4,091 traceを採点する。以下は全Tの値ではない。

| rank | iterations | damping | min Hz | pilot SNR (dB) |
|---:|---:|---:|---:|---:|
| 2 | 30 | 3 | 0.0 | 9.3580 |
| 4 | 30 | 3 | 0.0 | 10.7667 |
| 4 | 30 | 3 | 5.0 | 10.7663 |
| 4 | 30 | 8 | 0.0 | 10.0166 |
| 4 | 60 | 3 | 0.0 | 10.8117 |
| 6 | 30 | 3 | 0.0 | 11.5778 |
| 6 | 60 | 3 | 0.0 | 11.7049 |
| 4 | 10 | 3 | 5.0 | 9.4803 |
| 12 | 60 | 3 | 0.0 | 12.9891 |
| 16 | 60 | 3 | 0.0 | 12.8957 |
| 8 | 30 | 3 | 0.0 | 12.0744 |
| 8 | 60 | 3 | 0.0 | 12.3324 |

選択設定は `rank12_iterations60.yaml`。`bash scripts/run_c3_drr_target_tuning.sh` で全領域確認を実行する。

| 全領域run | Target SNR (dB) | RMSE | 再構成時間 (s) |
|---|---:|---:|---:|
| study_036 reference | 8.5207 | 3.4632 | 1118.2284 |
| rank 12 / 60 iterations | 12.1328 | 2.2849 | 7152.4726 |

[採用run](../../runs/study_039_c3_drr_target_tuning/20260911T103447Z_5829a11_rank12_iterations60/drr/metrics.json) と
[再検証記録](../../runs/study_039_c3_drr_target_tuning/20260911T103447Z_5829a11_rank12_iterations60/verification.json) が正本。
保存済みpredictionを全T (104,710 trace / 40,208,640 samples) で再採点し、記録された全指標と完全一致した。
Oの最大絶対誤差は0、全T coverageは100%、全predictionは有限、prediction SHA-256は一致。
入力lockの完全一致と、study_036の凍結済み全runファイル・formal configのSHA-256保持を確認した。

参照からの変更は rank 4→12、iterations 10→60、frequency_min_hz 5→0 の3項目のみ。
Damping 3、frequency_max_hz=null、window [5,8,4,8]、overlap [0,0,0,0]、
CPU/NumPy再構成とBLAS thread数1は保持する。全領域runは約119.3分。
Tを使った設定選択と追加計算量を含む調整済みの結果であり、独立評価ではない。
