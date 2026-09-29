FORGE M1の欠損率を変えたときの全6手法の補間結果を比較する。

条件の正本は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)。
手法設定は [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、
[CCNet5D](ccnet5d_grid.yaml)、[NeRSI](nersi_real.yaml)、
[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml) に固定する。
条件の採用理由は [decisions.md](decisions.md) を参照する。

比較元は [study_056](../study_056_forge_2ms_random80_trace_rms/README.md)。
同じhash順位を使い、今回の欠損集合を比較元の欠損集合の部分集合とする。
全手法で同一のマスク・時間サンプリング・トレース毎RMS正規化を使う。
欠損トレースの正解RMSは振幅復元に使用し、欠損波形やRMSをモデル入力・教師へ渡さない。
この比較は正解RMSを利用する診断条件である。

実行入口は `scripts/forge_m1_mvp.py`。
`prepare`、`preflight`、`run` の順に、固定入力の準備、限定動作確認、全手法の本番実行と評価を行う。
本番の共通評価は全6手法の予測が検査を通過してから行う。

```bash
.venv/bin/python scripts/forge_m1_mvp.py prepare \
  --study studies/study_057_forge_2ms_random70_trace_rms
```

全手法の本番実験と共通評価が完了した。
結果は [指標一覧](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run/tables/mvp_metrics.csv)、
完了状態は [final_mvp_manifest.json](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run/aggregate/final_mvp_manifest.json) を参照する。
入力と設定は [prepared_inputs.lock.json](prepared_inputs.lock.json) に固定した。
実行先は [本番run](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run/matrix_manifest.json)、
起動情報は [launch.json](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_launch/launch.json)、
進行状況は [launcher.log](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_launch/launcher.log) に記録する。

再実行する場合は、リポジトリのルートで次を実行する。
ニューラル手法はGPU 1、POCSとDRRはCPUを使用する。

```bash
.venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_057_forge_2ms_random70_trace_rms/20260922T032220545149Z_51acdb640a29 \
  --preflight runs/study_057_forge_2ms_random70_trace_rms/20260922T032321898442Z_51acdb640a29_preflight
```

各手法のログはrun直下の `<method>.log`、共通評価は `aggregate/` と `figures/` に保存される。
コードや固定設定を変更した場合は、新しいprepareとpreflightで入力lockを更新する。

POCSの追加調整では、元のtest集合の主指標を **1.8630から0.8381** へ改善し、
ユーザー指定の合格基準0.85以下を満たした。
採用条件は [pocs_grid_tuned.yaml](pocs_grid_tuned.yaml)、
再実行条件は [pocs_tuned_verification.yaml](pocs_tuned_verification.yaml)、
NumPyでの最終評価は [採用runの指標](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T085042853926Z_51acdb640a29_pocs_tuning/pocs_grid_tuned/metrics.json)、
評価上の制約と再実行手順は [調整報告](../../reports/forge_057_pocs_tuning.md) を参照する。
これは元のtest集合を使った調整結果であり、独立したtest評価ではない。

DRRの追加調整でも、元のtest集合の主指標を **0.9610から0.8373** へ改善し、
同じ合格基準0.85以下を満たした。
採用条件は [drr_grid_tuned.yaml](drr_grid_tuned.yaml)、
再実行条件は [drr_tuned_verification.yaml](drr_tuned_verification.yaml)、
NumPy SVDでの最終評価は [採用runの指標](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T091622758001Z_51acdb640a29_drr_tuning/drr_grid_tuned/metrics.json)、
評価上の制約と再実行手順は [DRR調整報告](../../reports/forge_057_drr_tuning.md) を参照する。
この追加条件も元のtest集合で選択した結果であり、独立したtest評価ではない。

NeRSIの追加調整は15候補を評価し、最良の主指標は **0.9212** だった。
目標の0.85以下は未達であり、元の固定NeRSI条件は維持する。
最良参考候補にはNeRSI内部の座標表現・時間整列の追加が含まれる。
条件は [nersi_real_tuned.yaml](nersi_real_tuned.yaml)、
再実行設定は [nersi_tuned_verification.yaml](nersi_tuned_verification.yaml)、
結果と評価上の制約は [NeRSI調整報告](../../reports/forge_057_nersi_tuning.md) を参照する。
この探索も元のtest集合を使用しており、独立したtest評価ではない。

CCNet5Dの追加調整では、元のtest集合の主指標を **1.0000から0.8175** へ改善し、
合格基準0.85以下を満たした。
採用候補には幾何だけから決める整数サンプルの時間整列が含まれる。
CCNet5D内部の時間軸カーネルも拡張した参考候補は主指標 **0.7198** に達したが、モデル構造を変えない採用候補とは分けて記録する。
条件は [ccnet5d_grid_tuned.yaml](ccnet5d_grid_tuned.yaml)、
再実行条件は [ccnet5d_tuned_verification.yaml](ccnet5d_tuned_verification.yaml)、
NumPyでの最終評価は [採用候補の指標](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T141021556552Z_25c5193588b2_ccnet_tuning/align500_k3_p4848_inner10_ema_10k/metrics.json)、
全候補と評価上の制約は [CCNet調整報告](../../reports/forge_057_ccnet_tuning.md) を参照する。
元の6手法比較の固定CCNet5D条件は置き換えず、これは元のtest集合で選択した結果であり、独立したtest評価ではない。

`common_*` の表示は、時間間引き後の正解トレースの全時刻RMSで
正解・入力・各手法の予測・誤差を共通に正規化し、`cmap="seismic"` で表示する。
保存済み予測から再生成した [共通震源断面](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T083239471275Z_51acdb640a29_trace_rms_figures/figures/common_source.png)、
[共通受振点断面](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T083239471275Z_51acdb640a29_trace_rms_figures/figures/common_receiver.png)、
[誤差図と生成記録](../../runs/study_057_forge_2ms_random70_trace_rms/20260922T083239471275Z_51acdb640a29_trace_rms_figures/manifest.json) は、
元の実験runとは別のrunに保存する。再描画は次で実行できる。

```bash
.venv/bin/python scripts/forge_mvp_sections.py \
  --run runs/study_057_forge_2ms_random70_trace_rms/20260922T032648104777Z_51acdb640a29_run
```
