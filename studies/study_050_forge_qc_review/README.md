# study_050_forge_qc_review

全ゼロ35本・非ゼロ定数100本の除外を固定し、残る波形のほぼ定数・DC・振幅異常を確認する。

- 状態: 分布集計・代表波形レビュー完了。追加QC規則とbenchmark採用は未確定。
- [条件](config.yaml)、[入力](inputs.yaml)、[判断理由](decisions.md)、[結果レポート](../../reports/forge_2017_qc_review.md)。
- 採用run: [20260918T060947568159Z_5d660f87b9b4](../../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/summary.json)。分布等4図、代表波形16図を保存。
- [固定マスク](../../data/processed/forge_2017/20260918T060947568159Z_5d660f87b9b4/fixed_qc_mask.parquet): 全1,932,182行、固定除外135本、既存ヘッダー除外後の残存候補1,919,596本。
- 診断q<0.01は40本、DC比>0.1は112,542本、生振幅比>100は1,034本、<0.01は387本。これらは追加除外ではない。

実行: `.venv/bin/python scripts/review_forge_qc.py`
