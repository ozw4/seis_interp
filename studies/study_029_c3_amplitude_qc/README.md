# C3 amplitude QC and normalization

このstudyは、初回GNNの極端な出力に結びついた振幅を元SEG-Yまで追跡し、
物理traceのQCと固定RMSによる正規化が数値的に成立することを確認する。
初回の5手法比較は[Study 028](../study_028_c3_first_results/README.md)、
元の固定入力は[Study 027](../study_027_c3_na_benchmark/README.md)に保持する。

[config.yaml](config.yaml)は旧benchmarkの幾何・時間・partition・mask seedを継承し、
既存の`sampling.trace_amplitude_filter`を明示する。元の625 sampleのいずれかが
`abs(amplitude) > 10000`ならtrace全体を除外し、除外行をpartitionへ記録する。
閾値は[Study 016の判断](../study_016_all_ffid_siren/decisions.md)を継承したもので、
今回のvalidation/test指標へ合わせて選んだものではない。

`exclude_all_zero: false`とし、正当なゼロを欠測扱いしない。波形のクリッピング、
補間修復、SEG-Yやinterim配列の書換えは行わない。canonicalなphysical cellの
除外によって別aliasを採用する場合や、固定評価cropの行が除外される場合は準備を失敗させる。

QCは元traceの全625 sampleを対象とするsource integrityの検査である。
GNNの教師・support・正規化fitは、QC後のcanonical trainとtime`[0,384)`だけを使う。
GNNの`training_data.max_abs_amplitude: 10000`はfit直前の再検査であり、
global RMSの式やcheckpoint形式を変えない。fit後にfloat32の二乗エネルギーが消失した
非ゼロtrace、非有限値、二乗のoverflowがないことを別途検査する。

新しいdata artifactだけを生成する。

```bash
python -m seis_interp.cli data prepare-c3-benchmark \
  --config studies/study_029_c3_amplitude_qc/config.yaml \
  --inputs studies/study_029_c3_amplitude_qc/inputs.yaml \
  --execute --json
```

既存directoryへの上書きは拒否する。全20caseは入力契約の生成・照合対象であり、
20caseのモデル訓練やtest予測を行う指定ではない。

数値診断は[probe.yaml](probe.yaml)に固定する。モデル構成は
[gnn_probe.yaml](gnn_probe.yaml)を使い、width32・各relation2近傍・AdamWを維持する。
seed20260908による最初のtrain episodeの最初の4 hidden queryを、CPU上で100更新だけ
繰り返す。正規化尺度は全QC済みtrainから固定し、query自身の波形は入力へ渡さない。
同じ4本に対する学習誤差の変化とencoderの入力保持を診断し、未知queryへの汎化や
全volume精度の証拠とは扱わない。良い結果が出るまでの再実行は行わない。

現在の状態は`qc_and_numerical_validation_complete`。新旧20caseの評価mask・volume index・
held-out splitの一致を確認した。除外はFFID 1746の437本だけで、trainは1,146,366本、
固定RMSは28.6279。非ゼロtraceのfloat32二乗エネルギー消失は0本となった。
CPUの固定4本診断では100更新後の物理RMSEが11.4754から10.5172へ下がった。
詳細なsource追跡、採用成果、検証範囲と限界は[報告書](../../reports/c3_amplitude_qc_20260908.md)に記録する。

SIREN-5Dは[siren_rerun.yaml](siren_rerun.yaml)と
[siren_rerun_inputs.yaml](siren_rerun_inputs.yaml)で新suiteへ接続し、前回と同じseed・
2,000更新で再学習した。評価対象SNRは−0.0003 dB、RMSEは9.9390で改善しなかった。
除外437本はtrainにあり、validation cropの観測点だけで学習するSIRENの入力と
volume内RMSは変わらない。結果と検証は
[SIREN再実行報告](../../reports/c3_siren_qc_rerun_20260909.md)に記録する。
