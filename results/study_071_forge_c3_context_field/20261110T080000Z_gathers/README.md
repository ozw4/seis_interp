# ギャザー比較図（neural-implicit パネルを `cinr_real` に差し替え）

[study_067 のギャザー比較図](../../study_067_forge_c3_common_grid/20261104T011853Z_35d86916bcc7_gathers/README.md)
と**同一の体裁・同一の規則・同一の表示断面**で、3列目の neural-implicit パネルだけを
profile NeRSI から `cinr_real`（[study_071](../../../studies/study_071_forge_c3_context_field/README.md) の
条件付き implicit field）に差し替えた図です。新たな推論・test 採点・設定選択はしていません。

差し替えの根拠は study_070 / 071 の測定です。profile NeRSI は FORGE で事前登録の床 0.4 dB を満たさず
（3 seed 平均 0.3126 dB、relMSE 0.9426）、`cinr_real` は両 dataset で床を満たしました
（C3 14.2516 dB / FORGE 1.4558 dB）。

## パネルの出どころ

4手法は study_067 の本番 run、`cinr_real` は study_071 の本番 run から読みます。
**両 study の seed 7321 partition は byte 一致**しており（C3 `33203b1a…`、FORGE `7268920d…`）、
図に出る trace は完全に同じです。manifest の `contributing_run_partition_sha256` が run ごとに記録しています。

| 手法 | パネル | C3 の cell | FORGE の cell | run |
|---|---|---|---|---|
| POCS | (c) | `a01_b05` | `a04_b01` | study_067 |
| DRR | (d) | `a05_b03` | `a03_b02` | study_067 |
| **cINR** | **(e)** | **`a03_b04`** | **`a05_b01`** | **study_071** |
| CCNet-5D | (f) | `a05_b02` | `a02_b05` | study_067 |
| ReGSI | (g) | `a02_b05` | `a05_b04` | study_067 |

各 cell はその手法自身の S1 勝者です（`*_selection.json` に記録、test は選択に使っていません）。

## 図

各図は 178 × 142 mm、PNG 1200 dpi と PDF。1行目は正解と入力、2行目は5手法の予測、3行目は残差です。

| dataset | 変化させる軸（他の軸は中心で固定） | trace 数（観測 / validation / test） | 物理振幅 | trace RMS 版 |
|---|---|---|---|---|
| C3 | 受振点 | 32（5 / 2 / 25） | `c3/gather_receiver_grid_point_physical` | — |
| C3 | 震源点 | 32（9 / 0 / 23） | `c3/gather_source_grid_point_physical` | — |
| C3 | 震源測線 | 16（2 / 0 / 14） | `c3/gather_source_grid_line_physical` | — |
| FORGE | 受振点 | 32（2 / 3 / 27） | `forge/gather_receiver_grid_point_physical` | `forge/gather_receiver_grid_point_trace_rms` |
| FORGE | 震源点 | 32（8 / 0 / 24） | `forge/gather_source_grid_point_physical` | `forge/gather_source_grid_point_trace_rms` |
| FORGE | 震源測線 | 6（1 / 0 / 5） | `forge/gather_source_grid_line_physical` | `forge/gather_source_grid_line_trace_rms` |
| FORGE | 受振測線 | 8（3 / 0 / 5） | `forge/gather_receiver_grid_line_physical` | `forge/gather_receiver_grid_line_trace_rms` |

表示断面・中心位置・clip 規則・残差規則・正規化規則は study_067 の図とすべて同一です。
表示した trace はすべて clean_target です。

## 検査

`acceptance.csv` は、図に描いた配列から計算し直した trace ごとの SNR が、その run が保存した採点値と
一致するかを記録します。**両 dataset・全5手法・validation と test のすべてで最大差 0.0 dB**
（相対誤差エネルギー差も 0.0）であり、表示が採点済みの予測そのものであることを確認しています。

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

## 注記

- `common_grid_gathers.completed_section_panels` は表示順を引数で受け取れるようになりました。
  既定は study_067 の順のままで、同じ入力に対して同じパネルを返すことをテストで固定しています
  （`test_a_declared_display_order_changes_which_predictions_are_drawn`）。study_067 の図は
  manifest が記録する commit `35d86916bcc7` で再現できます。
- 本図は **random-trace 70% mask（study_067 の共通 grid 条件）** のものです。whole-shot 50% mask の
  ギャザー図は run ごとに別で、study_069 のものは profile NeRSI を含み、かつ
  [study_076 の開示](../../../studies/study_076_forge_random_shot50_aligned_baselines/README.md)
  のとおり baseline の学習視野が未整合の状態です。whole-shot 側の差し替えは別途行う必要があります。
