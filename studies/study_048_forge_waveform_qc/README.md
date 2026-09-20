# FORGE waveform QC and component evidence

Status: `waveform_audit_complete_component_orientation_unconfirmed`.

Study 047のヘッダー候補に実際の波形があり、比較実験へ進む前に何を除外・確認すべきかを調べる。
全1,106ファイル・1,932,182トレースの全サンプルを測定した。
ヘッダー候補1,919,731本のうち35本は全ゼロ、100本は非ゼロ定数。
1,919,596本は有限値の非定数波形。
数値チェック通過1,806,130本、確認フラグ113,466本。フラグは自動除外ではない。
実際のZ/N/E成分・極性・物理振幅校正は未確定。最終benchmark採用は保留。

条件は [config.yaml](config.yaml)、入力は [inputs.yaml](inputs.yaml)、
判断理由は [decisions.md](decisions.md)、結果の説明と図は
[波形品質・成分の確認報告](../../reports/forge_2017_full_audit.md) を参照。

採用波形run:
`runs/study_048_forge_waveform_qc/20260918T052245952355Z_5d660f87b9b4/`。

成分根拠run:
`runs/study_048_forge_waveform_qc/20260918T053204806805Z_5d660f87b9b4_components/`。

## 再実行

```bash
.venv/bin/python scripts/audit_forge_waveforms.py
.venv/bin/python scripts/summarize_forge_components.py runs/study_048_forge_waveform_qc/<waveform-run-id>
```

2番目の引数は1番目のコマンドが表示する実runへ置き換える。
波形runは元SEG-YのSHA-256をStudy 047の入力lockと照合してから読む。
全トレースのQC表を `data/interim/forge_2017/<run-id>/waveform_qc.parquet` に保存する。
SEG-Y振幅は変更せず、設定・入力hash・実行コードsnapshot・環境・実行時刻を保存する。
成分runはヘッダー表と波形QC表のhashを再確認し、その対応から証拠を集計する。
