# ギャザー比較図（`cinr_real`、残差なし2行レイアウト）

[3行版](../20261110T080000Z_gathers/README.md)から**残差行を外し、7パネルを2行に並べ替えた**図です。
パネル1つあたりの面積が約2倍になり、個々のイベントが印刷サイズで判読できます。
描いている内容・数値・出どころは3行版と同一で、**残差を描かないこと以外に変更はありません**。

| | 3行版 | 本図 |
|---|---|---|
| パネル | 12（正解・入力・5手法・残差5） | **7**（正解・入力・5手法） |
| 行 | 2-5-5 | **3-4** |
| 図寸法 | 178 × 142 mm | 178 × **140** mm |
| パネル1つの幅 | 図幅の約 1/5 | **約 1/4** |

レイアウトは (a) 正解 / (b) 入力 / (c) POCS を1行目、(d) DRR / (e) cINR / (f) CCNet-5D / (g) ReGSI を
2行目に置きます。1行目の各パネルは真下に2行目のパネルがあるため横軸ラベルを持たず、
横軸は2行目だけに付きます。

## パネルの出どころ

4手法は study_067 の本番 run、`cinr_real` は study_071 の本番 run から読みます。
**両 study の seed 7321 partition は byte 一致**（C3 `33203b1a…`、FORGE `7268920d…`）しており、
図に出る trace は完全に同じです。

| 手法 | パネル | C3 の cell | FORGE の cell | run |
|---|---|---|---|---|
| POCS | (c) | `a01_b05` | `a04_b01` | study_067 |
| DRR | (d) | `a05_b03` | `a03_b02` | study_067 |
| **cINR** | **(e)** | **`a03_b04`** | **`a05_b01`** | **study_071** |
| CCNet-5D | (f) | `a05_b02` | `a02_b05` | study_067 |
| ReGSI | (g) | `a02_b05` | `a05_b04` | study_067 |

## 図

各図は 178 × 140 mm、PNG 1200 dpi と PDF。表示断面・中心位置・clip 規則・正規化規則は
study_067 の図とすべて同一で、表示した trace はすべて clean_target です。

| dataset | 変化させる軸（他の軸は中心で固定） | trace 数（観測 / validation / test） | 物理振幅 | trace RMS 版 |
|---|---|---|---|---|
| C3 | 受振点 | 32（5 / 2 / 25） | `c3/gather_receiver_grid_point_physical` | — |
| C3 | 震源点 | 32（9 / 0 / 23） | `c3/gather_source_grid_point_physical` | — |
| C3 | 震源測線 | 16（2 / 0 / 14） | `c3/gather_source_grid_line_physical` | — |
| FORGE | 受振点 | 32（2 / 3 / 27） | `forge/gather_receiver_grid_point_physical` | `forge/gather_receiver_grid_point_trace_rms` |
| FORGE | 震源点 | 32（8 / 0 / 24） | `forge/gather_source_grid_point_physical` | `forge/gather_source_grid_point_trace_rms` |
| FORGE | 震源測線 | 6（1 / 0 / 5） | `forge/gather_source_grid_line_physical` | `forge/gather_source_grid_line_trace_rms` |
| FORGE | 受振測線 | 8（3 / 0 / 5） | `forge/gather_receiver_grid_line_physical` | `forge/gather_receiver_grid_line_trace_rms` |

## 検査

`acceptance.csv` は、図に描いた配列から計算し直した trace ごとの SNR が、その run が保存した採点値と
一致するかを記録します。**両 dataset・全5手法・validation と test のすべてで最大差 0.0 dB** です。

## 再現

```bash
.venv/bin/python scripts/export_field_grid_gathers.py \
  --c3       runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces \
  --c3-field runs/study_071_forge_c3_context_field/20261109T042809568788Z_766f4f10a666_traces \
  --forge       runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge \
  --forge-field runs/study_071_forge_c3_context_field/20261109T042814452404Z_766f4f10a666_forge \
  --output results/study_071_forge_c3_context_field/<result-id>
```

数値・hash・規則の正本は [manifest.json](manifest.json) です。
残差を見たい場合は3行版（[20261110T080000Z_gathers](../20261110T080000Z_gathers/README.md)）を参照します。
