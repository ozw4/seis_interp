# study_049_forge_geometry_eda

FORGE 2017の取得済み相関処理済み3Dデータ全1,106ファイルについて、取得配置と2種類の4次元空間表現の充足率・セル重複を確認する。

- 状態: EDA完了。
- [条件](config.yaml)、[入力契約](inputs.yaml)、[判断理由](decisions.md)、[結果レポート](../../reports/forge_2017_full_audit.md)。
- 採用run: [20260918T053449083425Z_5d660f87b9b4](../../runs/study_049_forge_geometry_eda/20260918T053449083425Z_5d660f87b9b4/summary.json)。
- 数値的に使用可能な1,919,596トレース、24格子条件、6図を保存。
- ローカルstation組合せに対する有効率99.6910%。UTMの震源・受振50 mビンでは矩形充足率2.1405%、セル重複率4.9918%。
- 本runを配置診断として採用。比較実験の領域・人工欠測・波形格子化の条件は未採用。

実行: `.venv/bin/python scripts/audit_forge_geometry.py`
