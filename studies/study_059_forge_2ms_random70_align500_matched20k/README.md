FORGE M1で、欠損率70%・500 m/sの幾何整列・トレース毎RMS正規化を全手法共通の条件とし、
POCS・DRR・CCNet5D・NeRSI・ReGSI（real/grid）を比較する。
全手法の主指標（clean_targetの平均トレース相対MSE）0.9以下を目標とする。

条件の正本は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)。
手法設定は [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、[CCNet5D](ccnet5d_grid.yaml)、
[NeRSI](nersi_real.yaml)、[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml) に固定する。
共通条件と各設定の判断理由は [decisions.md](decisions.md) を参照する。

欠損マスク・正解RMS・観測波形はstudy_057と同一である。
整列、ニューラル3手法の予算（20,000更新・学習率0.001）とパラメータ数の範囲（ReGSIの363,269の±5%）はstudy_058と同じである。
1更新でlossをかけるトレースは、CCNet5D・NeRSI・ReGSIとも全更新で64本である。
パラメータ数はCCNet5D 353,413、NeRSI 369,839、ReGSI 363,269。
CCNet5DとNeRSIの設定は元のtest集合の主指標を参考に選んだため、結果は独立test性能の推定ではない。

実行入口は `scripts/forge_m1_mvp.py`。`prepare`・`preflight`・`run` の順に実行する。
本番runは `--concurrent` で、同じCUDAデバイスの手法だけを固定順で1つずつ実行する。
手法間のruntimeは比較できない。

```bash
.venv/bin/python scripts/forge_m1_mvp.py prepare \
  --study studies/study_059_forge_2ms_random70_align500_matched20k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011629803645Z_25c5193588b2 \
  --preflight runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011634771911Z_25c5193588b2_preflight \
  --concurrent
```

NeRSIの候補探索は `scripts/forge_matched_tuning.py` で準備runに対して行い、
各候補を `runs/study_059_forge_2ms_random70_align500_matched20k/*_matched_tuning/` に記録した。

最終本番run [20260924T011721371526Z_25c5193588b2_run](../../runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011721371526Z_25c5193588b2_run/matrix_manifest.json) は全6手法が評価まで完了した。
結果は [指標一覧](../../runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011721371526Z_25c5193588b2_run/tables/mvp_metrics.csv)、
完了状態は [final_mvp_manifest.json](../../runs/study_059_forge_2ms_random70_align500_matched20k/20260924T011721371526Z_25c5193588b2_run/aggregate/final_mvp_manifest.json) を参照する。

| 手法 | 主指標（clean_target 34,150本） | 0.9以下 |
|---|---:|---|
| POCS | 0.7524 | 達成 |
| DRR | 0.7915 | 達成 |
| CCNet5D | 0.7140 | 達成 |
| NeRSI | 0.8875 | 達成 |
| ReGSI-real | 0.6813 | 達成 |
| ReGSI-grid | 0.6999 | 達成 |

全6手法が目標0.9以下を満たした。
学習ログでは、CCNet5Dの1更新あたりlossトレースは全20,000更新で64.0本、ReGSI real/gridのクエリは全20,000更新で64本だった。
