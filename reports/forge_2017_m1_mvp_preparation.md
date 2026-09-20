# FORGE 2017 M1 MVP 準備結果

2026-09-20。仕様v0.1に従う入力固定と限定preflightを完了。本番6 run、実データtest指標、MVP受入判定は未実施。

採用成果物は[固定lock](../studies/study_052_forge_m1_mvp/prepared_inputs.lock.json)。

- [準備manifest](../runs/study_052_forge_m1_mvp/20260920T053919825212Z_69a85f1aac33/preparation_manifest.json)
- [準備ゲート](../runs/study_052_forge_m1_mvp/20260920T053919825212Z_69a85f1aac33/preparation_gates.json)
- [限定preflight](../runs/study_052_forge_m1_mvp/20260920T053938747257Z_69a85f1aac33_preflight/preflight_manifest.json)
- [実座標／投影格子](../runs/study_052_forge_m1_mvp/20260920T053919825212Z_69a85f1aac33/figures/geometry.png)
- [固定mask占有図](../runs/study_052_forge_m1_mvp/20260920T053919825212Z_69a85f1aac33/figures/outer_mask.png)

## 固定入力

|項目|結果|
|---|---:|
|4Dセル数|49,152|
|有効トレース|49,151|
|自然欠測セル|1|
|eligibleセル衝突|0|
|observed|9,830|
|test（all-eligible）|39,321|
|test（clean-target）|39,035|
|testの要確認union|286|
|test DC要確認|272|
|test 振幅要確認|57|
|test ほぼ定数要確認|0|
|1トレースの時刻数|4,001|
|時間範囲・間隔|0–4 s・1 ms|

共通Global RMSのexact保存値は `0.010077235423311664`。observed 9,830本だけの全サンプルからfloat64で集計した。test振幅は読み出していない。

Mask SHA-256: `8d91941dbf37f20b62dd03aa43e0cc9d1bdec5e844fd6fad90a7632983825f97`。

study_051の5入力を保存hashと照合し、元の192 SEG-Yもheader auditのhashと照合した。trace↔cellの一意性、軸順、行順非依存mask、格子から再計算した投影座標を検証した。元UTM列をNaNに置換してもgrid featureが不変であり、real/grid featureが異なることを確認した。

no-mask対応の検査は全cellの合成signature、実波形の値の往復検査はobserved全本を対象にした。test振幅を準備時に読む必要はない。実testの有限予測・逆対応・指標再計算は本番評価で検査する。

## 固定した比較設定

|手法|学習step|parameter count|checkpoint|
|---|---:|---:|---|
|POCS|—|—|決定的反復|
|DRR|—|—|決定的反復|
|CCNet-5D|5,000|111,969|最終|
|NeRSI実座標|50,000|155,806,513|最終EMA|
|ReGSI実座標|20,000|363,269|最終EMA|
|ReGSI格子|20,000|363,269|最終EMA|

ReGSIの両条件はgeometry以外の設定と初期weight hashが一致する。seed・batch順・inner-mask規則も共通。学習lossは既存の`masked_trace_relative_mse`。ReGSIはC3採用構成を使い、M1上のHPOは行わない。

NeRSIは4連続実座標から1トレースを出力する明示的なshape適応。C3の3座標・断面出力実装と同一ではない。詳細は[固定判断](../studies/study_052_forge_m1_mvp/decisions.md)。本番でこの適応をM1 test品質から選び直さない。

表示断面はsource cell 112（source line 509 / point 171）とreceiver cell 144（receiver line 165 / point 523）、時間0–4 sを固定済み。

## 限定preflight

POCS/DRRは所定の空間windowで1周波数slice、neuralは全4,001サンプルの1 diagnostic updateと先頭query batch/tileを確認した。重み・予測を採用せず、品質指標は計算していない。

|手法|状態|限定処理時間 [s]|CPU peak RSS [GiB]|GPU peak allocated [GiB]|
|---|---|---:|---:|---:|
|pocs_grid|passed|5.6764|3.6388|0.0000|
|drr_grid|passed|1.2326|1.5552|0.0000|
|ccnet5d_grid|passed|3.3539|3.2032|4.7372|
|nersi_real|passed|4.2710|2.6530|2.9986|
|regsi_real|passed|2.6352|2.3376|6.3379|
|regsi_grid|passed|3.1656|2.3386|6.5383|

これらは限定処理の計測値であり、全step・全test推論の時間や最大メモリの上限ではない。GPUはNVIDIA H100 NVL。正式runでは各processのCPU RSS・GPU allocated/reserved peak・実行時間を別途保存する。

## 本番開始コマンド

```bash
.venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_052_forge_m1_mvp/20260920T053919825212Z_69a85f1aac33 \
  --preflight runs/study_052_forge_m1_mvp/20260920T053938747257Z_69a85f1aac33_preflight
```

全6手法のmanifestと予測のhash・shape・cell ID・有限値・全予測の定数崩壊、ReGSI初期状態を検査し、すべて通過した場合だけ評価処理がtest振幅を読む。失敗・未完了・定数崩壊はtestを読まず`incomplete`を保存し、部分成功の指標も計算しない。不正なhash・予測・初期状態も読み出し前に停止する。clean-target/all-eligible、trace別指標、ReGSI paired差・投影距離四分位、事前固定断面を保存する。既存lossと独立式、保存表、元波形からの再計算を照合する。1 mask・1 seedの予備結果として扱う。

prepareと全6手法のpreflightはcommit `69a85f1aac33` のclean worktreeから再生成した。実装snapshotとhashは `src/seis_interp/**/*.py` および `scripts/forge_m1_mvp.py` を対象とし、起動前にファイル集合と内容を照合する。旧準備manifestとpreflightは採用せず、固定lockを更新した。予測保存は既存のNumPy形式であり、Zarr依存は追加していない。
