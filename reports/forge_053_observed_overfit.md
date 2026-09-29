Study 053：観測データだけを使った固定バッチ過学習試験

**NeRSIとReGSI real/gridは、固定した少数の教師波形をほぼ再現できた。** CCNet5Dも損失は下がったが、2,000回では完全な過学習には至らなかった。

[実行記録](../runs/study_053_forge_m1_trace_rms_idw/20260921T022520488292Z_51acdb640a29_observed_overfit/manifest.json) / [全指標](../runs/study_053_forge_m1_trace_rms_idw/20260921T022520488292Z_51acdb640a29_observed_overfit/summary.json)。

元のStudy 053と同じ初期重み・モデル構造・学習率・optimizer・loss・4,001サンプルの波形長を使用。各手法の最初の学習バッチを固定し、教師・入力・maskを変更せず2,000回反復した。ReGSI real/gridは同じ6本を教師とし、geometryに応じた近傍を使用した。手法間で教師集合と課題難度が異なるため、この表は手法の優劣比較には使わない。

教師とcontextがすべてouter observedに属し、両者が重ならないことを保存IDから確認した。今回の教師はすべてclean_targetでもあった。test正解波形は読み込んでいない。観測RMSで正規化したモデル空間で採点し、IDWによる振幅復元は行っていない。学習した同じ教師上の指標であり、汎化性能ではない。

| 手法 | 教師本数 | context本数 | 初期相対MSE | 最終raw相対MSE | 最終raw相関中央値 | 最終EMA相対MSE |
|---|---:|---:|---:|---:|---:|---:|
| ccnet5d_grid | 7 | 2 | 1.0018 | 0.3241 | 0.8115 | — |
| nersi_real | 1 | 0 | 1.0088 | 0.0005 | 0.9999 | 0.0492 |
| regsi_real | 6 | 126 | 1.0000 | 0.0029 | 0.9987 | 0.1963 |
| regsi_grid | 6 | 131 | 1.0000 | 0.0022 | 0.9988 | 0.2134 |

NeRSIは座標だけを入力とするためcontextは0本。CCNet5DにはEMAを適用しない。数値は小数4桁に丸めた。rawとEMAはどちらも固定2,000回終了時の値であり、最良stepを選択していない。

![固定バッチの学習損失](forge_053_observed_overfit.png)

今回の結果は、forward・教師・loss・backwardの経路が小規模な課題では機能することを支持する。元のReGSI-realのゼロ出力を「real座標では構造上学習できない」と説明する根拠は弱まった。ただし、固定例の記憶に成功しても、未観測位置の補間能力や実装全体の正しさは保証されない。

元の実験の教師提示回数は、観測1本あたり平均でNeRSIが5.0865回、ReGSIが12.1952回だった。今回は固定した各教師を2,000回提示しており、教師集合とcontext/maskを固定し、各教師の反復量を増やした。どの要因が支配的かは今回だけでは区別できない。次は固定教師の数を段階的に増やす試験と、教師を固定してcontextだけを変える試験を分けるのが適切である。

EMAは短い診断でrawより誤差が大きく、平均化の追従遅れが見られる。ただし元の実験ではEMA前の学習損失自体がほぼ1だったため、元の崩壊をEMAだけに帰すことはできない。

再実行コマンド：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python scripts/forge_observed_overfit.py \
  --preparation runs/study_053_forge_m1_trace_rms_idw/20260920T203738718661Z_3fb8de051ed4 \
  --steps 2000
```

実行条件・実装snapshot・入力hash・初期重みhash・教師/context ID・学習履歴・最終raw重み・raw/EMA予測はrun内に保存した。保存された全artifactのhashと、NumPyによる予測波形からの相対MSE再計算が一致した。
