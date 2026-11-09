# FORGE periodic whole-shot50

study067のFORGE random70で選ばれた設定を、periodic whole-shot50へ移す固定設定比較。
契約の正本は[config.yaml](config.yaml)、入力・選択元hashは[inputs.yaml](inputs.yaml)。
標準はphase0の1 mask、model seed 7301、5手法各1設定。validationは設けない。

現在の結果：標準5手法は全て完了。
[採用図表・CSV](../../results/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0_analysis)、
[技術報告](../../reports/forge_068_shot50.md)、
[生成元run](../../runs/study_068_forge_shot50/20261105T000750814115Z_4d0ae2be41a0)。

```bash
python scripts/forge_trace_experiment.py prepare --study studies/study_068_forge_shot50
python scripts/forge_trace_experiment.py launch --experiment runs/study_068_forge_shot50/<run-id>
python scripts/forge_trace_experiment.py score --experiment runs/study_068_forge_shot50/<run-id>
python scripts/forge_trace_experiment.py figures --experiment runs/study_068_forge_shot50/<run-id> --output results/study_068_forge_shot50/<result-id>
```

`score`は全手法の完了／失敗が確定するまで採点しない。workerも最後の状態確定時に同じ採点を呼ぶ。
完了済みmethodはhashを検証して再利用し、再学習しない。未完了・失敗runは上書きしない。

反対位相は次で**別runを準備**し、その新しいrunに`launch`する。標準実行には追加しない。

```bash
python scripts/forge_trace_experiment.py prepare --study studies/study_068_forge_shot50 --observed-phase 1
```

phase1もfresh初期化とし、phase0のcheckpointを使わない。2phaseは独立な乱数seedではない。
trace_oracleによる真の欠損trace RMSを使う条件付き評価。clean_targetは既存QCの対象であり、
ノイズのない真値を意味しない。外部testへの汎化実験やshot50で再最適化した比較とは扱わない。
