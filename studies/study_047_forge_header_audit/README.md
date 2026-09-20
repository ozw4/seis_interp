# FORGE 2017 header audit

Status: `header_audit_complete`; waveform QC is recorded in study_048; benchmark eligibility remains pending.

研究質問は、手元の FORGE correlated shot gathers の全ヘッダーを読み取り、
補間用の地震波候補と dead・補助・意味未確認トレースを区別できるか。
入力契約は [inputs.yaml](inputs.yaml)、実行条件は [config.yaml](config.yaml)、
分類判断の根拠は [decisions.md](decisions.md) を参照。

1,106 SEG-Y、1,932,182 traces の監査を完了した。読み取り失敗は0ファイル。
地震波候補1,919,731、dead指定5,815、標準コードの補助3,318、未確認コード3,318。
今回の整合性規則で地震波候補に追加の除外はなく、波形QC候補として採用する。
追加3 ZIP全812 SEG-YとNavigation ZIPのCRC・展開先SHA-256一致を確認した。
5D補間の評価用データとしての最終採用は未判定。

採用run:
`runs/study_047_forge_header_audit/20260918T051827453600Z_5d660f87b9b4/`。

- [監査報告と未解決事項](../../reports/forge_2017_full_audit.md)
- [機械可読集計](../../runs/study_047_forge_header_audit/20260918T051827453600Z_5d660f87b9b4/summary.json)
- [ファイル台帳](../../runs/study_047_forge_header_audit/20260918T051827453600Z_5d660f87b9b4/files.csv)
- [全トレース表](../../data/interim/forge_2017/20260918T051827453600Z_5d660f87b9b4/trace_headers.parquet)

## 実行

リポジトリrootから実行する。依存関係の追加は不要。
再実行時は別の日時付きrunとinterim出力を作成する。

```bash
.venv/bin/python scripts/audit_forge_headers.py
```

元SEG-Yは読み取り専用。入力全ファイルのSHA-256、実行コードsnapshot、設定、
環境、開始・終了時刻をrunに保存する。波形サンプルはハッシュ計算のために
バイト列として読むが、振幅としてはデコードしない。
