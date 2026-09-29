FORGE M1で、欠損率80%・500 m/sの幾何整列・トレース毎RMS正規化を全手法共通の条件とし、
POCS・DRR・CCNet5D・NeRSI・ReGSI（real/grid）を比較する。
全手法の主指標（clean_targetの平均トレース相対MSE）0.9以下を目標とする。

条件の正本は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)。
手法設定は [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、[CCNet5D](ccnet5d_grid.yaml)、
[NeRSI](nersi_real.yaml)、[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml) に固定する。
共通条件と揃え方の判断理由は [decisions.md](decisions.md) を参照する。

欠損マスク・正解RMS・観測波形はstudy_056と同一である。
ニューラル3手法は20,000更新・学習率0.001で、1更新あたりのlossトレースはC3 NAの正本どおり64本とする。
パラメータ数はReGSIの363,269の±5%以内に揃える。
各手法の設定は元のtest集合の主指標で選んだため、結果は独立test性能の推定ではない。

実行入口は `scripts/forge_m1_mvp.py`。`prepare`・`preflight`・`run` の順に実行する。
正式runは `--concurrent` で、同じCUDAデバイスの手法だけを固定順で1つずつ実行する。
手法間のruntimeは比較できない。

```bash
.venv/bin/python scripts/forge_m1_mvp.py prepare \
  --study studies/study_058_forge_2ms_random80_align500_matched20k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/forge_m1_mvp.py run \
  --preparation runs/study_058_forge_2ms_random80_align500_matched20k/20260923T075244862150Z_25c5193588b2 \
  --preflight runs/study_058_forge_2ms_random80_align500_matched20k/20260923T075249950424Z_25c5193588b2_preflight \
  --concurrent
```

手法ごとの候補探索は `scripts/forge_matched_tuning.py` で準備runに対して行い、
各候補を `runs/study_058_forge_2ms_random80_align500_matched20k/*_matched_tuning/` に記録した。

本番run [20260923T075338454988Z_25c5193588b2_run](../../runs/study_058_forge_2ms_random80_align500_matched20k/20260923T075338454988Z_25c5193588b2_run/matrix_manifest.json) は全6手法が評価まで完了した。
結果は [指標一覧](../../runs/study_058_forge_2ms_random80_align500_matched20k/20260923T075338454988Z_25c5193588b2_run/tables/mvp_metrics.csv)、
完了状態は [final_mvp_manifest.json](../../runs/study_058_forge_2ms_random80_align500_matched20k/20260923T075338454988Z_25c5193588b2_run/aggregate/final_mvp_manifest.json) を参照する。

| 手法 | 主指標（clean_target 39,035本） | 0.9以下 |
|---|---:|---|
| POCS | 0.8583 | 達成 |
| DRR | 0.8845 | 達成 |
| ReGSI-real | 0.8761 | 達成 |
| ReGSI-grid | 0.9249 | 未達 |
| CCNet5D | 0.9479 | 未達 |
| NeRSI | 0.9528 | 未達 |

3手法が目標0.9以下を満たし、3手法は未達である。
