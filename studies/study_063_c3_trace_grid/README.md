# C3: 同一全領域内の観測holdoutによる5手法比較

Status: partially stopped (ReGSI / CCNet5D / NeRSI). 実行状況の正本はrun内の `workers/*.json` とする。

現在の実行: [supervisor](../../runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/workers/supervisor.json)。
実行前確認: [コマンドと結果](../../runs/study_063_c3_trace_grid/20261011T145500928121Z_05c92e79ce80_traces/prelaunch_checks.json)。

ReGSI・CCNet5D・NeRSIの再探索は [study_064](../study_064_c3_reference_grid/README.md) で実施する。
POCS・DRRはこのrunを継続する。

POCSのForgeと同じ窓の25構成による再探索は、[専用設定](pocs_forge_matched.yaml)を正本とする。
C3の既存入力・分割・前処理を引き継ぎ、検証のみのS1を別runで実行する。
この再探索では追加3 seed評価を行わない。
実行状況は[POCS再探索の進捗](../../runs/study_063_c3_trace_grid/20261022T024554946564Z_05c92e79ce80_pocs_forge_matched_iterations32_grid/workers/pocs_grid.json)、
起動前の検査は[確認記録](../../runs/study_063_c3_trace_grid/20261022T024554946564Z_05c92e79ce80_pocs_forge_matched_iterations32_grid/prelaunch_checks.json)を参照する。

C3の同じ全領域内で、観測トレースの10%を検証用に隠し、残り90%を
学習・再構成に使う。元の欠損トレースがtestとなる。空間ブロックは作らない。
分割規則はstudy_060の `make_validation_holdout` と同じ実装を使う。
各runは入力23,593本、検証2,621本、test104,858本となる。

各手法の25構成を共通の固定マスクで探索し、検証V最小の1設定を選ぶ。
全5手法の探索完了後、seed 7311/7312/7313で観測・検証・testの
トレース割当を変え、選んだ設定で追加3 runを実行する。各seedの割当は
全5手法で共通とする。訓練用乱数は7301固定とし、学習型手法は毎回新規学習する。
3 seedの全結果・平均・標本標準偏差を報告して終了する。全体140 run。

実行条件は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)、
25構成は [grid.yaml](grid.yaml)、各手法の固定値は手法別YAMLを正本とする。
分割の採用理由は [decisions.md](decisions.md) に記録する。

```bash
.venv/bin/python scripts/run_c3_region_experiment.py prepare --study studies/study_063_c3_trace_grid
PYTHONPATH=<experiment>/source/src .venv/bin/python <experiment>/source/scripts/run_c3_region_experiment.py run --repo /workspace --experiment <experiment>
.venv/bin/python scripts/run_c3_region_experiment.py status --experiment <experiment>
```
