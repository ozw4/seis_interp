FORGE M1を共通の時間サンプリング・欠損マスク・トレースRMS正規化で比較する。

条件の正本は [config.yaml](config.yaml)、入力の正本は [inputs.yaml](inputs.yaml)。
手法設定は [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、
[CCNet5D](ccnet5d_grid.yaml)、[NeRSI](nersi_real.yaml)、
[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml) に固定する。
採用理由と旧条件との差は [decisions.md](decisions.md) に記録する。

共通の反射padding付き窓付きsincフィルタで時間を間引いた後、
各トレースを自身の全時刻RMSで正規化する。
欠損トレースの正解RMSは振幅復元にだけ使い、波形やRMSをモデルの入力特徴へ渡さない。
正解RMSを使う診断条件であり、未知トレースの振幅を推定する評価ではない。
全欠損トレースを予測し、clean-targetとall-eligibleの両方を同じ物理振幅で評価する。

実行入口は `scripts/forge_m1_mvp.py`。`prepare` は入力・設定・実装のhashと
source snapshotを固定する。`preflight` は各手法の限定された動作確認を行い、
確認用の重みは破棄する。`run` が全手法の本番実行と共通評価を順番に行う。
POCSとDRRのアルゴリズム反復数は各手法設定に従い、学習更新数とは区別する。

```bash
.venv/bin/python scripts/forge_m1_mvp.py prepare \
  --study studies/study_056_forge_2ms_random80_trace_rms
```

採用した準備・事前確認は [prepared_inputs.lock.json](prepared_inputs.lock.json) に固定した。
全手法の事前確認と本番実験・共通評価が完了した。
結果は [指標一覧](../../runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_run/tables/mvp_metrics.csv)、
実行完了の検査結果は [final_mvp_manifest.json](../../runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_run/aggregate/final_mvp_manifest.json) を参照する。
実行先は [本番run](../../runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_run/matrix_manifest.json)、
起動情報は [launch.json](../../runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_launch/launch.json)、
進行状況は [launcher.log](../../runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_launch/launcher.log) に記録する。
再実行する場合は、リポジトリのルートで次を実行すると、固定した全手法を順番に実行して共通評価する。
ニューラル手法はGPU 1、POCSとDRRはCPUを使用する。

```bash
.venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_056_forge_2ms_random80_trace_rms/20260921T155219488331Z_51acdb640a29 \
  --preflight runs/study_056_forge_2ms_random80_trace_rms/20260921T155302873838Z_51acdb640a29_preflight
```

出力先は起動時刻を含む新しいrunディレクトリとして表示される。
各手法のログはその直下の `<method>.log`、共通結果は `aggregate/` と `figures/` に保存される。
コードや固定設定を変更した場合は、新しいprepareとpreflightで入力lockを更新する。

`common_*` の表示は、時間間引き後の正解トレースの全時刻RMSで
正解・入力・各手法の予測・誤差を共通に正規化し、`cmap="seismic"` で表示する。
保存済み予測から再生成した [共通震源断面](../../runs/study_056_forge_2ms_random80_trace_rms/20260922T083301002480Z_51acdb640a29_trace_rms_figures/figures/common_source.png)、
[共通受振点断面](../../runs/study_056_forge_2ms_random80_trace_rms/20260922T083301002480Z_51acdb640a29_trace_rms_figures/figures/common_receiver.png)、
[誤差図と生成記録](../../runs/study_056_forge_2ms_random80_trace_rms/20260922T083301002480Z_51acdb640a29_trace_rms_figures/manifest.json) は、
元の実験runとは別のrunに保存する。再描画は次で実行できる。

```bash
.venv/bin/python scripts/forge_mvp_sections.py \
  --run runs/study_056_forge_2ms_random80_trace_rms/20260921T233952218910Z_51acdb640a29_run
```
