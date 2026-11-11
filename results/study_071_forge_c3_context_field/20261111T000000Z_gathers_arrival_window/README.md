# ギャザー比較図（初動前の窓を重ねた版）

[残差なし2行版](../20261110T085000Z_gathers_two_row/README.md)と**同一の図**に、
[初動前後の分割評価](../../../reports/arrival_window_metrics.md)で使っている窓の位置を重ねたものです。
波形・断面・clip・正規化・パネル配置は一切変えていません。

> **更新版あり・本図の破線は誤り**: 破線「Measured arrival」はエンベロープが早期レベルの3倍を
> 超える点であり、初動ではなく立ち上がりの途中である。そのため「実線が破線より上だから窓は初動前」
> という読み方は成立しない。実測に基づく
> [20261111T020000Z_gathers_arrival_window](../20261111T020000Z_gathers_arrival_window/README.md)
> を使うこと。本図は経緯の記録として残す。

## 重ねた2本の線

| 線 | 意味 | FORGE | C3 |
|---|---|---|---|
| 実線（黒） | **宣言した窓の終端**。ここより上が初動前の窓 | `t = 0.22 s + offset/5500 m/s` | `t = 0.25 s + offset/7000 m/s` |
| 破線（灰） | 観測 trace から実測した初動の直線近似 | `t = 0.314 s + offset/3865 m/s` | `t = 0.375 s + offset/5227 m/s` |

どちらも `t = 切片 + offset / 速度` の直線なので、ギャザー上では**表示 trace の offset に沿った曲線**に
なる。受振点ギャザーでは offset が点番号とともに変化するため、線も湾曲する。

**実線が常に破線より上にある**ことが、窓が初動前に収まっていることの視覚的な確認である。
余裕は FORGE で近オフセット約 0.09 s・遠オフセット約 0.24 s、C3 で約 0.09 s 以上ある。
切片を下げ、速度を上げる向きに丸めて宣言したためである。

**破線は直線近似である。** C3 の実測初動は遠オフセットで寝る（0.68–0.70 s で頭打ち）ため、
直線近似は遠オフセット側で実際より早い時刻を示す。これは窓をより保守的にする向きの誤差であり、
実線の位置には影響しない。窓の定数は観測 trace だけから決めており、**初動 pick は一切使っていない**
（target を読むことになり採点規律に反するため）。

## 図から読めること

- 初動前の窓は、**C3 では参照が完全に白い**（合成データなので厳密に無音）。この領域に出た振幅は
  すべて作り出した偽信号である。
- **FORGE では参照の窓内に環境ノイズがある。** POCS・DRR のパネルは窓内も雑音で埋まり、
  学習型（cINR・CCNet-5D・ReGSI）のパネルは窓内が白い。これが報告書の結果2の内容で、
  **窓内で 0 を出すのが二乗誤差の最適**（ノイズは trace ごとに別実現で予測不能）である。

## 図と出どころ

4手法は study_067、`cinr_real` は study_071 の本番 run（seed 7321 partition は byte 一致）。
図の構成・パネルの出どころ・検査（acceptance 全手法 0.0 dB）は
[2行版の README](../20261110T085000Z_gathers_two_row/README.md) と同じである。

## 再現

```bash
.venv/bin/python scripts/export_field_grid_gathers.py --show-window \
  --c3       runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces \
  --c3-field runs/study_071_forge_c3_context_field/20261109T042809568788Z_766f4f10a666_traces \
  --forge       runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge \
  --forge-field runs/study_071_forge_c3_context_field/20261109T042814452404Z_766f4f10a666_forge \
  --output results/study_071_forge_c3_context_field/<result-id>
```

`--show-window` を外すと重ねない版（2行版と同一）になる。窓の定数は
`evaluation/arrival_window_metrics.py` の `ARRIVAL_WINDOWS` と `MEASURED_ARRIVALS` が正本で、
manifest の `arrival_window_overlay` にも記録される。
