指定runのCCNet5Dを、主指標であるclean_targetの平均トレース相対MSEが0.85以下になることを目標に調整した。
NumPyで最終確認した主指標は **0.8175** で、合格基準0.85以下を満たした。
判定に使用した値は `0.8175340216613434`。

| 指標 | 元のCCNet5D | 採用候補 |
|---|---:|---:|
| clean_target相対MSE（34,150本、主指標） | 1.0000 | 0.8175 |
| 全欠損相対MSE（34,406本） | 1.0000 | 0.8185 |
| clean_target global NMSE | 1.0000 | 1.0074 |
| clean_target相関中央値 | approximately 0 | 0.4240 |
| clean_target定数予測本数 | 33,840 | 0 |

global NMSEは1を上回るため、トレース毎に正規化する主指標と、物理振幅のエネルギーで重み付くglobal NMSEの結果は分けて扱う。

元のCCNet5Dは学習開始直後から学習lossが1.0000に留まり、予測は時間方向に定数のバイアス出力に崩壊していた。
観測トレースの隣接セル間の零ラグ相関は中央値でapproximately 0、最良ラグは29〜73サンプル（58〜146 ms）で、
支配的エネルギーは6.7〜19.7 Hzの低速な線形イベントである。
kernel 3・4段のCCNet5Dの時間受容野は±4サンプル（±8 ms）なので、隣接トレースから欠損トレースを予測できず、零出力が最適解になっていた。

採用候補は [ccnet5d_grid_tuned.yaml](../studies/study_057_forge_2ms_random70_trace_rms/ccnet5d_grid_tuned.yaml)、
再実行条件は [ccnet5d_tuned_verification.yaml](../studies/study_057_forge_2ms_random70_trace_rms/ccnet5d_tuned_verification.yaml)。
[評価指標](../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/align500_k3_p4848_inner10_ema_10k/metrics.json)、
[モデル](../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/align500_k3_p4848_inner10_ema_10k/final.pt)、
[予測](../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/align500_k3_p4848_inner10_ema_10k/prediction.npy)、
[実行記録](../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/manifest.json)、
[全候補と選択記録](../runs/study_057_forge_2ms_random70_trace_rms/20260922T163201806174Z_25c5193588b2_ccnet_tuning_summary/summary.json) を保存した。

| 項目 | 元の設定 | 採用設定 |
|---|---|---|
| 空間カーネル | 3 | 3 |
| 時間カーネル／dilation | 3／1（時間受容野±4サンプル） | 3／1（時間受容野±4サンプル） |
| hidden／intermediate channels／パラメータ数 | 32／32／111,969 | 32／32／111,969 |
| 学習パッチ（時間, source line, source point, receiver line, receiver point） | [2001, 2, 4, 2, 4] | [2001, 4, 8, 4, 8] |
| inner mask fraction | 0.8 | 0.1 |
| optimizer／学習率 | Adam／0.001 | Adam／0.001 |
| 更新数 | 10,000 | 10,000 |
| 最終checkpoint | raw重み | EMA 0.999 |
| 幾何整列 | なし | offset／500 m/s の整数サンプル遅延 |
| 予測core（時間, 空間） | [64, 6, 16, 8, 16] | [256, 6, 32, 8, 32] |

幾何整列は、各セルの震源–受振点距離を500 m/sで割った遅延（最大2,342サンプル）を整数サンプルへ丸め、
全トレースを最大遅延との差だけ遅らせて長さ4,343サンプルの時間キャンバスに置く。
CCNet5Dはこのキャンバス上で学習・予測し、予測後に同じ整数シフトで元の2,001サンプルへ厳密に戻す。
遅延は幾何テーブルのoffsetだけから決め、波形・観測有無・欠損トレースを参照しない。
切り詰めは生じず、観測トレースの再挿入も元の位置で維持される。
これはNeRSI調整で追加した内部時間整列と同種の、幾何だけを使う追加表現である。
参考候補で用いた時間カーネルとdilationは各Conv3Dの時間軸だけを広げるCCNet5D内部の変更で、既定値では元のCCNet5Dと同一の関数になる。
主指標が最小の候補は [align500_tk9d2_p4848_inner10_ema_6k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T143700496117Z_25c5193588b2_ccnet_tuning/align500_tk9d2_p4848_inner10_ema_6k/metrics.json)（0.7198）で、CCNet5D内部の時間軸カーネル拡張を含む。採用は合格基準を満たし、モデル構造を変えない候補とした。

元の準備済み入力、欠損マスク、2 ms間隔、全時間区間、トレースRMS正規化、正解RMSによる物理振幅への復元を維持した。
学習に使用する波形は観測14,745本だけで、欠損34,406本の波形は予測後の評価にだけ使用する。
loss（相対MSE）、Adam、一定学習率、mixed precisionなし、固定最終checkpoint、モデルseed 7301は元の契約どおりで、
学習パッチは全2,001サンプル（整列後は全時間キャンバス）を含む。
POCS・DRR・NeRSI・ReGSIの予測との合成、帯域制限、入力データの前処理変更は行っていない。
主指標による候補選択は元のtest集合で行っており、独立test性能の推定ではない。

既存パラメータだけを変更した候補は元のCCNet5Dを改善しなかった。

| 候補・指標へのリンク | 変更 | 更新数 | 主指標 |
|---|---|---:|---:|
| [inner10_k3_p2424_10k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T134517923809Z_25c5193588b2_ccnet_tuning/inner10_k3_p2424_10k/metrics.json) | inner mask 0.1 | 10,000 | 1.0000 |
| [inner10_k5_p4848_ema_10k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T134517923809Z_25c5193588b2_ccnet_tuning/inner10_k5_p4848_ema_10k/resolved_config.yaml) | inner mask 0.1、kernel 5、パッチ [2001, 4, 8, 4, 8]、EMA 0.999 | 10,000 | 中断、未評価 |
| [inner10_k7_p2424_10k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T134517923809Z_25c5193588b2_ccnet_tuning/inner10_k7_p2424_10k/metrics.json) | inner mask 0.1、kernel 7 | 10,000 | 1.0000 |

幾何整列を含む候補は次のとおり。

| 候補・指標へのリンク | 時間カーネル／dilation | パッチ | バッチ | checkpoint | 更新数 | 主指標 |
|---|---|---|---:|---|---:|---:|
| [align500_tk9d2_b16_inner10_ema_10k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T140328540386Z_25c5193588b2_ccnet_tuning/align500_tk9d2_b16_inner10_ema_10k/metrics.json) | 9／2 | [2001, 2, 4, 2, 4] | 16本 | EMA 0.999 | 10,000 | 0.8321 |
| [align500_k3_p4848_inner10_ema_10k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/align500_k3_p4848_inner10_ema_10k/metrics.json) | 3／1 | [2001, 4, 8, 4, 8] | パッチ1枚 | EMA 0.999 | 10,000 | 0.8175 |
| [align500_tk9d2_p4848_inner10_ema_6k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T143700496117Z_25c5193588b2_ccnet_tuning/align500_tk9d2_p4848_inner10_ema_6k/metrics.json) | 9／2 | [2001, 4, 8, 4, 8] | パッチ1枚 | EMA 0.999 | 6,000 | 0.7198 |
| [align500_tk9d2_p4848_inner10_raw_6k](../runs/study_057_forge_2ms_random70_trace_rms/20260922T150252961757Z_25c5193588b2_ccnet_tuning/align500_tk9d2_p4848_inner10_raw_6k/metrics.json) | 9／2 | [2001, 4, 8, 4, 8] | パッチ1枚 | raw | 6,000 | 0.7316 |

候補の方向づけには、記録しない短時間の予備fitを用いた。正式な比較はrun内に保存した候補だけで行う。
候補は複数のGPUで並行して実行したため、runtimeは他手法と比較しない。

採用候補の再実行:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/forge_ccnet_tuning.py \
  --run runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run \
  --config studies/study_057_forge_2ms_random70_trace_rms/ccnet5d_tuned_verification.yaml
```

各探索条件はrun内の`search.yaml`と`resolved_config.yaml`に保存する。
検証コマンドと結果は [verification.json](../runs/study_057_forge_2ms_random70_trace_rms/20260922T152203151431Z_25c5193588b2_ccnet_verification/verification.json) を参照する。
