# C3 SIREN: per-trace RMS and interpolated physical scale

このstudyは、各観測traceをunit RMSへ正規化してSIRENを学習し、欠測位置のRMSを
観測traceから内挿して物理振幅へ戻したときの精度を調べる。
[Study 029](../study_029_c3_amplitude_qc/README.md)の異常437本を除外済みのsuiteを使う。
同studyのglobal RMSによるSIREN再実行が比較対象である。

## 入力と方法

`c3_benchmark_validation_random_trace_80_seed142`の固定crop
`[384,9,32,8,32]`、観測14,729本、評価対象58,999本を使う。
SIRENはこのcropの観測波形だけで学習する。除外437本はtrain partitionにあり、
SIRENの観測波形にもスケール内挿の候補にも含まれない。

観測traceごとに選択時間384 sampleのRMSをfloat64で計算し、そのRMSで割る。
RMSが0の観測traceは正規化波形0、復元尺度0として保持する。
ゼロ除算を避けるための除数1は、物理復元尺度とは区別する。

欠測位置の尺度は、観測traceのみを候補とする8近傍のIDWで求める。
座標順はsource X/Y、relative receiver X/Y、距離は各成分を
`[160,80,40,40] m`で割った4次元ユークリッド距離とする。
この尺度はcropの幾何的な格子間隔から決めたもので、評価波形からfitしない。
重みは距離の逆二乗、補間対象は物理RMSの算術平均である。
距離同値は元の`array_row`で順序を固定し、候補が8本未満なら存在する候補を使う。
観測位置は自身の実測RMSを使い、内挿で置き換えない。

SIRENの出力へ各位置の尺度を1回掛けて物理振幅へ戻し、観測値を厳密に再挿入する。
欠測波形の真値、真値のRMS、予測波形自身のRMSは、この復元尺度の算出に使わない。
全評価対象・全時間sampleの物理SNRとRMSEで評価する。
checkpointには学習済み重み、各位置の尺度、対応する元row IDと内挿設定を保存する。

## 実行と採否

[config.yaml](config.yaml)は実行計画、[inputs.yaml](inputs.yaml)はsuiteのhash固定、
[methods/siren.yaml](methods/siren.yaml)はnative条件である。
モデル・学習率・ランダム点sampler・batch16,384・seed20260908・2,000更新・GPUは
比較対象と同じとし、正規化と物理尺度の復元方法を変更する。

```bash
python scripts/run_c3_first_results.py \
  --config studies/study_030_c3_siren_trace_scaling/config.yaml \
  --inputs studies/study_030_c3_siren_trace_scaling/inputs.yaml \
  --action siren --execute
```

学習・推論・全対象採点の完了、観測値の保持、target波形変更に対する学習入力と
スケールの不変性、checkpointからの物理予測の復元を確認する。
同条件の品質再試行、test partitionの学習・採点は行わない。
現在の状態は`per_trace_scaling_evaluated`。物理SNRは−0.0002 dB、RMSEは9.9389で、
前回から実質的な改善はなかった。尺度内挿の相対絶対誤差中央値は1.9647%、
relative L2は5.3229%だった。学習損失は約1のままで、波形の学習不足が残る。
詳細と比較図は[報告書](../../reports/c3_siren_trace_scaling_20260909.md)に記録する。
