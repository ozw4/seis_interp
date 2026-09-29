指定runのNeRSIを、主指標であるclean_targetの平均トレース相対MSEが0.85以下になることを目標に調整した。
15候補の最良値は **0.9212** で、**目標0.85は未達**だった。
判定に使用した値は `0.9212219976395666`。

| 指標 | 元のNeRSI | 最良参考候補 |
|---|---:|---:|
| clean_target相対MSE（34,150本、主指標） | 1.0000 | 0.9212 |
| 全欠損相対MSE（34,406本） | 1.0000 | 0.9228 |
| clean_target global NMSE | 1.0000 | 1.1520 |
| clean_target相関中央値 | approximately 0 | 0.3764 |
| clean_target定数予測本数 | 32,603 | 0 |

主指標は改善したが、global NMSEは悪化した。
最良候補には、trace版NeRSIへ追加した内部座標表現と時間整列が含まれる。
既存のモデル・学習パラメータだけを変更した7候補では、追加候補の最小値は1.0635であり、元のNeRSIを改善しなかった。
追加表現を含む8候補も合わせて比較したが、合格条件には到達していない。
元の6手法比較と、その固定NeRSI条件は維持する。

最良参考条件は [nersi_real_tuned.yaml](../studies/study_057_forge_2ms_random70_trace_rms/nersi_real_tuned.yaml)、
再実行条件は [nersi_tuned_verification.yaml](../studies/study_057_forge_2ms_random70_trace_rms/nersi_tuned_verification.yaml)。
[評価指標](../runs/study_057_forge_2ms_random70_trace_rms/20260922T102846265039Z_25c5193588b2_nersi_tuning/nersi_moveout_v700_regularized_long/metrics.json)、
[モデル](../runs/study_057_forge_2ms_random70_trace_rms/20260922T102846265039Z_25c5193588b2_nersi_tuning/nersi_moveout_v700_regularized_long/final.pt)、
[予測](../runs/study_057_forge_2ms_random70_trace_rms/20260922T102846265039Z_25c5193588b2_nersi_tuning/nersi_moveout_v700_regularized_long/prediction.npy)、
[全候補と選択記録](../runs/study_057_forge_2ms_random70_trace_rms/20260922T104211285621Z_25c5193588b2_nersi_tuning_summary/summary.json) を保存した。

| 最良候補の項目 | 設定 |
|---|---|
| Fourier成分数 | 80／座標軸 |
| 周波数の底 | 中点x: 1.005、中点y: 1.005、距離: 1.04、方位: 1.00001 |
| エンコーダ幅／潜在チャネル | 512／8 |
| デコーダチャネル／カーネル | [16, 8, 4]／5 |
| バッチサイズ／学習率 | 64／0.0003 |
| 重み減衰／更新数 | 0.0001／24,000 |
| EMA／モデルseed | 0.999／7301 |
| デコーダ内部の時間整列 | 700 m/s相当 |

座標変換は、実震源・実受振点座標から中点、距離、方位を計算する。
座標の範囲と変換係数は幾何情報だけで決め、欠損波形を参照しない。
内部時間整列は距離に比例する遅延をNeRSIのデコーダ出力へ適用する。
拡張した時間キャンバスから全2,001サンプルを出力するため、評価区間の切り詰めはない。
POCS・DRR・ReGSIの予測との合成は行わない。

元の準備済み入力、欠損マスク、2 ms間隔、全時間区間、トレースRMS正規化、物理振幅への復元を維持した。
学習に使用する波形は観測14,745本だけで、欠損34,406本の波形は予測後の評価にだけ使用する。
元実験のoracle RMS条件は共通条件として継承する。
主指標による候補選択は元のtest集合で行っており、独立test性能の推定ではない。

| 候補・指標へのリンク | 座標・時間表現 | 更新数 | 主指標 |
|---|---|---:|---:|
| [ff16_axis_b32_lr3e4_6000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T093032002476Z_51acdb640a29_nersi_tuning/ff16_axis_b32_lr3e4_6000/metrics.json) | 既存パラメータ | 6,000 | 1.1437 |
| [ff16_smooth_b32_lr3e4_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T093349109108Z_51acdb640a29_nersi_tuning/ff16_smooth_b32_lr3e4_12000/metrics.json) | 既存パラメータ | 12,000 | 1.0635 |
| [ff16_mid_b32_lr3e4_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T093428510844Z_51acdb640a29_nersi_tuning/ff16_mid_b32_lr3e4_12000/metrics.json) | 既存パラメータ | 12,000 | 1.1997 |
| [ff16_lines_wide_b64_lr5e4_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T094034698202Z_51acdb640a29_nersi_tuning/ff16_lines_wide_b64_lr5e4_12000/metrics.json) | 既存パラメータ | 12,000 | 1.2539 |
| [ff80_dense_b64_lr3e4_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T094445341021Z_51acdb640a29_nersi_tuning/ff80_dense_b64_lr3e4_12000/metrics.json) | 既存パラメータ | 12,000 | 1.5557 |
| [ff80_original_decoder_b32_lr1e4_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T094924216315Z_51acdb640a29_nersi_tuning/ff80_original_decoder_b32_lr1e4_12000/metrics.json) | 既存パラメータ | 12,000 | 1.4346 |
| [cmp_offset_ff80_b64_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T095505479966Z_51acdb640a29_nersi_tuning/cmp_offset_ff80_b64_12000/metrics.json) | 中点・オフセット | 12,000 | 1.6105 |
| [cmp_polar_ff80_b64_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T095713581539Z_51acdb640a29_nersi_tuning/cmp_polar_ff80_b64_12000/metrics.json) | 中点・距離・方位 | 12,000 | 1.1823 |
| [cmp_polar_ff160_b64_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T100255892388Z_51acdb640a29_nersi_tuning/cmp_polar_ff160_b64_12000/metrics.json) | 中点・距離・方位 | 12,000 | 1.2841 |
| [cmp_polar_ff160_wd1e4_b128_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T100550441857Z_51acdb640a29_nersi_tuning/cmp_polar_ff160_wd1e4_b128_12000/metrics.json) | 中点・距離・方位 | 12,000 | 1.1044 |
| [nersi_moveout_v500_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T101514908320Z_51acdb640a29_nersi_tuning/nersi_moveout_v500_12000/metrics.json) | 距離・方位＋内部時間整列 | 12,000 | 0.9540 |
| [nersi_moveout_v700_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T101514925891Z_51acdb640a29_nersi_tuning/nersi_moveout_v700_12000/metrics.json) | 距離・方位＋内部時間整列 | 12,000 | 0.9365 |
| [original_ff160_b128_lr3e4_wd1e5_12000](../runs/study_057_forge_2ms_random70_trace_rms/20260922T101827018245Z_51acdb640a29_nersi_tuning/original_ff160_b128_lr3e4_wd1e5_12000/metrics.json) | 既存パラメータ | 12,000 | 1.6847 |
| [nersi_moveout_v700_early](../runs/study_057_forge_2ms_random70_trace_rms/20260922T102846034919Z_25c5193588b2_nersi_tuning/nersi_moveout_v700_early/metrics.json) | 距離・方位＋内部時間整列 | 8,000 | 0.9313 |
| [nersi_moveout_v700_regularized_long](../runs/study_057_forge_2ms_random70_trace_rms/20260922T102846265039Z_25c5193588b2_nersi_tuning/nersi_moveout_v700_regularized_long/metrics.json) | 距離・方位＋内部時間整列 | 24,000 | 0.9212 |

広い周波数範囲や大きいモデルでは、観測ミニバッチへの適合が進んでも欠損位置の誤差が増える候補があった。
内部時間整列と正則化による改善は得られたが、今回の探索範囲で0.85以下には到達しなかった。

最良参考候補の再実行:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/forge_nersi_tuning.py \
  --run runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run \
  --config studies/study_057_forge_2ms_random70_trace_rms/nersi_tuned_verification.yaml
```

各探索条件はrun内の`search.yaml`と`resolved_config.yaml`に保存する。
検証コマンドと結果は [verification.json](../runs/study_057_forge_2ms_random70_trace_rms/20260922T101937519199Z_51acdb640a29_nersi_verification/verification.json) を参照する。
