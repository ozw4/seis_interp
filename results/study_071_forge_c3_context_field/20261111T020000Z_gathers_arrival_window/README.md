# ギャザー比較図（窓の位置と、窓が清浄かどうかを重ねた版）

[残差なし2行版](../20261110T085000Z_gathers_two_row/README.md)と同一の図に、
[初動前後の分割評価](../../../reports/arrival_window_metrics.md)で使う窓の終端を重ねたものです。
波形・断面・clip・正規化・パネル配置は変えていません。

## 線の意味

**1本の線（窓の終端 `t = 切片 + offset / 速度`）を、trace ごとに色分けしています。**

| 色 | 意味 |
|---|---|
| 黒 | **窓の終端が静かな床の上にある**（その trace では窓は無信号） |
| 赤 | **窓の終端が立ち上がりの途中にある**（その trace では窓は無信号ではない） |

判定は参照波形だけから行う: 窓末尾 10 サンプルの RMS ÷ その trace の最初 25 サンプルの RMS が
**1.05 を超えたら赤**。手法に依存しないので、7 パネルすべてで同じ色分けになる。

| dataset | 窓 |
|---|---|
| FORGE | `t = 0.22 s + offset/5500 m/s` |
| C3 | `t = 0.25 s + offset/7000 m/s` |

**FORGE では最小オフセット付近が赤くなる。** 受振点ギャザーでは中央付近が最小オフセットなので、
線の中央が赤く、両端（遠オフセット）が黒い。**C3 は全区間が黒い**（参照が窓内で厳密に無音）。

## 前の版からの変更

[20261111T000000Z_gathers_arrival_window](../20261111T000000Z_gathers_arrival_window/README.md) は
破線で「実測した初動」を併記していたが、**その線は初動ではなかった**。中央値エンベロープが早期レベルの
3 倍を超える点を初動としており、立ち上がりのかなり上を指していた。そのため実線が破線より上にあることが
「窓は初動前」の根拠に見えていたが、実際には近〜中オフセットで窓は立ち上がりの途中で終わっている。

**正しい初動の線は引けない。** 床が平坦でなくなる点として測り直すと、FORGE の中間オフセットには
**平坦な床が存在せず**（振幅が t = 0 から単調に上がる）、10 分位で測った立ち上がり時刻は
0.152–0.632 s とばらつき、直線あてはめの残差が 0.15 s に達する。推定した初動曲線を描くことは
測定していない精度を主張することになるので、**描かない**。

代わりに、実際に測れる量（窓末尾が床の上かどうか）を線の色で示した。
前の版は経緯の記録として残す。

## 図と出どころ

4手法は study_067、`cinr_real` は study_071 の本番 run（seed 7321 partition は byte 一致、
acceptance 全手法 0.0 dB）。構成は [2行版の README](../20261110T085000Z_gathers_two_row/README.md) と同じ。

## 再現

```bash
.venv/bin/python scripts/export_field_grid_gathers.py --show-window \
  --c3       runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces \
  --c3-field runs/study_071_forge_c3_context_field/20261109T042809568788Z_766f4f10a666_traces \
  --forge       runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge \
  --forge-field runs/study_071_forge_c3_context_field/20261109T042814452404Z_766f4f10a666_forge \
  --output results/study_071_forge_c3_context_field/<result-id>
```

定数の正本は `evaluation/arrival_window_metrics.py` の `ARRIVAL_WINDOWS`・`CLEAN_EDGE_RATIO`・
`FLOOR_SAMPLES`・`EDGE_SAMPLES`。オフセット帯ごとの比は各 `summary.csv` の
`median_window_edge_over_floor` にある（FORGE: 1.079 / 1.144 / 1.178 / 0.951 / 0.923）。
