# FORGE M1 トレースRMS + IDW 実験準備

2026-09-20。study_053として準備と全6手法の限定preflightを完了。本番学習・評価は未実行。外側test正解振幅の読出しは0。

契約は[study README](../studies/study_053_forge_m1_trace_rms_idw/README.md)、採用入力は[lock](../studies/study_053_forge_m1_trace_rms_idw/prepared_inputs.lock.json)。

- [prepare](../runs/study_053_forge_m1_trace_rms_idw/20260920T203738718661Z_3fb8de051ed4/preparation_manifest.json)
- [振幅処理契約](../runs/study_053_forge_m1_trace_rms_idw/20260920T203738718661Z_3fb8de051ed4/input/amplitude_contract.json)
- [スケール表](../runs/study_053_forge_m1_trace_rms_idw/20260920T203738718661Z_3fb8de051ed4/input/trace_scales.parquet)
- [preflight](../runs/study_053_forge_m1_trace_rms_idw/20260920T203809057060Z_3fb8de051ed4_preflight/preflight_manifest.json)

## 検証結果

- commit `3fb8de051ed46a09980ea96dcdfab774b88c92eb` のclean worktreeからprepareと全preflightを生成。
- 元の192 SEG-Yとstudy_051入力をhash検証し、observed 9,830本だけを読み出した。
- 有効49,151、test39,321、clean-target39,035。maskとmappingのhashはstudy_052と同一。
- 各observedを自身のRMSで割った後のtrace RMS範囲: 0.9999999927362226–1.0000000073282216。
- 全testのIDW RMSが正かつ有限。最小0.000011271680046762634、中央値0.0005389328170418409、最大0.15737259056025935（元振幅単位）。これらはobservedからの推定値でありtest正解との比較ではない。
- 共通投影格子、各軸±2セル、距離の逆二乗。4軸距離スケールは[352.89935478865124, 50, 200, 50] m。先頭・中央・末尾test cellのIDWを独立式でも照合。
- 全6手法のresolved configは振幅処理の3項目を除いて旧prepareと一致。学習型4手法の初期weight hashは旧preflightとも一致。
- ReGSI両条件は同一振幅スケール表。inner targetをcontextとして読めないこと、教師が単位RMSであることをテスト。
- モデル出力×IDW RMSを1回適用。予測RMSの強制補正はなし。元Global RMSは診断用のみ。

## 限定preflight

|手法|状態|時間[s]|CPU peak RSS[GiB]|GPU peak allocated[GiB]|
|---|---|---:|---:|---:|
|pocs_grid|passed|8.67|3.786|0.000|
|drr_grid|passed|3.06|1.703|0.000|
|ccnet5d_grid|passed|5.94|3.348|4.737|
|nersi_real|passed|5.87|2.799|2.999|
|regsi_real|passed|5.28|2.471|6.338|
|regsi_grid|passed|2.23|2.488|6.538|

1周波数sliceまたは1学習stepと先頭推論batch/tileの検証値であり、本番全runの時間・メモリ上限ではない。診断重みは採用しない。品質指標は未計算。

## 実行コマンド

```bash
.venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_053_forge_m1_trace_rms_idw/20260920T203738718661Z_3fb8de051ed4 \
  --preflight runs/study_053_forge_m1_trace_rms_idw/20260920T203809057060Z_3fb8de051ed4_preflight
```

6手法の予測検査通過後にだけtest振幅を読み、元振幅単位で評価する。study_052の結果を見た後の明示的な追試であり、旧MVP v0.1と同一条件の結果として扱わない。

## チェック

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest tests/integration/test_forge_mvp.py tests/unit/test_forge_trace_scaling.py tests/unit/test_forge_mvp_contract.py tests/unit/test_forge_mvp_metrics.py tests/unit/test_trace_rms_idw.py -q
```

lint/format成功（668ファイル）、70 passed in 43.68s。旧Global RMSと新trace RMSの両条件で6手法の合成データ実行・評価、失敗時test非読出し、改変検出を確認。
