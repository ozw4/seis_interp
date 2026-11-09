# FORGE random whole-shot50

各source line内で半数のshotをランダムに観測する固定設定5手法比較。
条件の正本は[config.yaml](config.yaml)、入力・study067 FORGE選択元は[inputs.yaml](inputs.yaml)。
periodic shot50の[study068](../study_068_forge_shot50/README.md)は独立した結果として保持する。

現在の結果：標準5手法は全て予測・採点を完了し、失敗・未実行は0件。
[採用図表・CSV](../../results/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0_analysis)、
[生成元の継続run](../../runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0)。
NeRSI・POCS・DRRの予測を引き継ぎ、停止したReGSI・CCNet5Dの再実行も完了した。
[元run](../../runs/study_069_forge_random_shot50/20261105T140101431528Z_4d0ae2be41a0)は保持し、
入力・source・予測の照合と引き継ぎ元は[continuation.json](../../runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0/continuation.json)に記録した。

主評価clean_target（24,397 trace）の結果は次のとおり。表示は小数4桁。
全精度値と副評価は[summary.csv](../../results/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0_analysis/summary.csv)、
全入力・予測hash、test cell被覆、保存指標の再集計、checkpoint、採点順序の照合は[final_audit.json](../../runs/study_069_forge_random_shot50/20261106T010056122444Z_4d0ae2be41a0/final_audit.json)。

| 手法 | mean trace SNR (dB) | masked trace relative MSE |
| --- | --- | --- |
| ReGSI | 2.5702 | 0.6191 |
| CCNet5D | 1.2157 | 0.7898 |
| profile-wise NeRSI | approximately 0 | 1.0000 |
| POCS | 0.7105 | 0.8901 |
| DRR | 1.0033 | 0.8192 |

maskはseed 7302のSHA256順位で事前固定する。キーはUTF-8の空白なしJSON配列
`[study_id,candidate_id,mask_id,mask_seed,source_grid_line,source_cell]`。
各lineでdigest、source_cellの順に昇順ソートし、先頭16 shotを観測とする。
元のsource_grid_point・source_cellを保ち、全eligible receiverを同一roleにする。

```bash
python scripts/forge_trace_experiment.py prepare --study studies/study_069_forge_random_shot50
python scripts/forge_trace_experiment.py launch --experiment runs/study_069_forge_random_shot50/<run-id>
python scripts/forge_trace_experiment.py score --experiment runs/study_069_forge_random_shot50/<run-id>
python scripts/forge_trace_experiment.py figures --experiment runs/study_069_forge_random_shot50/<run-id> --output results/study_069_forge_random_shot50/<result-id>
```

全methodの状態確定後にだけtestを採点する。完了済みmethodは再学習しない。
図のshotは学習前にgeometryだけで固定し、shot別表に観測の片側／両側支持と欠損gapを記録する。
1 mask・1 model seedの記述的比較で、trace_oracleによる条件付き評価とする。
