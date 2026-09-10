# Study032: QC後の proposed GNN の学習量と振幅処理

固定C3 validationの全58,999 target traces ×384 samplesで、proposed
`RelationalTraceGraphInterpolator`の最終5,000更新モデルは物理振幅global SNR
**11.9354 dB、RMSE 2.5151**となった。保存予測の独立再採点とCPU checkpoint復元の
固定基準を満たし、10 dB超のモデルとして採用する。GNNの固定Shearは0である。
SIRENのShear付き11.3422 dBは[Study031](../study_031_c3_siren_10db/README.md)の
採用参照モデルとして別条件で記録する。

同じ採用設定の振幅処理だけを`train_global_rms`へ変えた比較では、final 5,000更新の
独立予測が**10.3033 dB、RMSE 3.0350**となり、全対象再採点とCPU復元監査に合格した。
観測traceごとの正規化とquery尺度のIDW復元を使わず、訓練データでfitした共通RMSだけで
入出力振幅を変換する。relative MSEの教師RMSによるloss重みは維持した。
この条件のbest 4,000更新は補助予測で11.0417 dBだったが、主比較は両条件ともfinal 5,000更新とする。
採用済み11.9354 dBより低いため、現在の採用モデルと`config.yaml`は維持する。
[global RMSの比較報告](../../reports/c3_proposed_gnn_global_rms_relative_mse_5k_20260910.md)と
[実行条件](config_train_global_rms_relative_mse_fp32_no_benchmark_5k.yaml)を参照する。

入力は[inputs.yaml](inputs.yaml)のQC suite（SHA-256
`f707a2e09dc0c0c2e137c1c2aef3ed57ade7bf8a3f05e9b784f531d70ed3a3ee`）に固定する。
評価caseは`c3_benchmark_validation_random_trace_80_seed142`、欠損80%、観測14,729本、
時間0–3.064 s、dt=0.008 sである。全22,655,616 target samplesを物理振幅で採点し、
contextの有無や振幅を理由に評価対象を除外しない。validation真値の使用は採点・可視化に
限り、test partitionの数値は使用しない。この1 case・1 seedでの達成であり、
未使用testや複数seedでの汎化を確認した結果ではない。

学習と前処理fitは異常437本を除外したcanonical train 1,146,366本の`[0,384)`のみを使う。
各episodeの可視波形だけをGNN入力にし、hidden波形は訓練ラベルとして分離する。
可視トレースをそれぞれのRMSで正規化し、予測点の直接可視近傍を重複除去したD0-IDWで
RMSを内挿する。decoder出力にそのscaleを一度だけ掛け、既存のglobal RMS
28.6279による物理単位への復元を保つ。教師RMSを入力や予測scaleへ渡さない。

訓練lossは`masked_trace_relative_mse`。各hidden訓練教師のRMSをlossの重みにだけ
使い、高振幅トレースへの集中を抑える。正のRMSにfloorやclipは使わず、RMSが0の教師は
固定global正規化単位でdivisor 1とする。評価は引き続き全targetの物理振幅global SNRである。
validation/test教師をこのlossへ混ぜない。SIRENはvalidation観測のみから学習しており、
GNNとは学習情報量が異なる。

採用条件は[config.yaml](config.yaml)、実行前に固定した条件は
[config_observed_trace_rms_relative_mse_fp32_no_benchmark_5k.yaml](config_observed_trace_rms_relative_mse_fp32_no_benchmark_5k.yaml)。
幅64、attention幅32、4関係×近傍2本、2 rounds、time downsample 2、query batch128、
AdamW学習率1e-3、seed20260908、5,000更新、640,000 query exposureである。
`exact_index`とepisode単位の可視近傍cacheで従来の距離・ID順を保った検索を行う。
`max_edge_time_shift_samples=0`で追加の学習lagも使わない。
TF32の環境overrideを両方0に固定し、`training.cudnn_benchmark=false`を使う。
物理MSE基準からはlossと訓練時のbackend設定の両方を変えており、lossだけの因果効果を
分離した比較とは主張しない。

採否は宣言した5,000更新のfinal checkpointで判定する。best checkpointは補助記録である。
最終結果、全対象再採点、事前固定した1,143 queryのCPU復元監査、比較図と採用記録は
[最終報告](../../reports/c3_proposed_gnn_relative_mse_5k_20260909.md)を参照する。
モデル重み・全予測・凍結sourceは生成元runに保持する。採用した評価記録・図表の
生成元とSHA-256は最終報告から参照できる。

`execution.source_snapshot`と`execution.environment`は実行時の記録である。
再実行時もPYTHONPATHと環境変数を明示する。以下は元の凍結sourceがあるworkspaceでの入口。

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=runs/study_032_c3_proposed_gnn_10db/20260909T093547080821Z_e39df16d563c_cudnn_benchmark_implementation_checks/source/src \
python -m seis_interp.pipelines.c3_first_results \
  --config studies/study_032_c3_proposed_gnn_10db/config_observed_trace_rms_relative_mse_fp32_no_benchmark_5k.yaml \
  --inputs studies/study_032_c3_proposed_gnn_10db/inputs.yaml \
  --action gnn-train --execute
```

採用に至る条件変更の理由は[decisions.md](decisions.md)に残す。過去の失敗と監査結果は
上書きせず、[QC基準](../../reports/c3_proposed_gnn_investigation_20260909.md)、
[振幅処理](../../reports/c3_proposed_gnn_amplitude_investigation_20260909.md)、
[global RMS 5k](../../reports/c3_proposed_gnn_global_rms_5k_20260909.md)、
[数値精度と学習lag](../../reports/c3_proposed_gnn_numerical_and_lag_20260909.md)、
[訓練energy](../../reports/c3_proposed_gnn_training_energy_20260909.md)、
[物理MSE・FP32 5k](../../reports/c3_proposed_gnn_observed_rms_fp32_5k_20260909.md)の各報告へまとめる。
