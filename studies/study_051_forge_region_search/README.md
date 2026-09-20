# study_051_forge_region_search

FORGE全データから、震源・受振点の直交等間隔グリッドに近い局所領域を複数サイズで探索する。

- 状態: 探索・成果物検証完了。最終benchmark領域と波形の格子化処理は未採用。
- [設定](config.yaml)、[入力](inputs.yaml)、[判断理由](decisions.md)、[結果レポート](../../reports/forge_2017_region_search.md)。
- 採用run: [20260920T014514223263Z_5d660f87b9b4](../../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/summary.json)。1,575局所格子、1,200組合せ、15候補、31図。
- 小領域先頭候補: 4,096有効トレース、充足率100%、衝突なし。震源p95距離37.83 m、受振p95距離0.36 m。
- 中領域先頭候補: 49,151有効トレース、充足率99.9980%、衝突なし。震源p95距離90.01 m。
- 事前の3目安をすべて満たした候補は0件。閾値を緩めず未達を記録する。
- 全ゼロ35本・非ゼロ定数100本の固定除外を維持。QCフラグは順位に使用しない。

実行: `.venv/bin/python scripts/search_forge_regions.py`
