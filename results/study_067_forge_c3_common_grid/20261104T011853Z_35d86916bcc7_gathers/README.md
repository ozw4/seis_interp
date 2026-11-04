# study_067 のギャザー比較図

本番で選ばれた cell の、評価 seed 7321 の保存済み test 予測を、過去の図（study_042 の C3 断面、study_060 の FORGE ギャザー）と同じ体裁で並べた図です。新たな推論・test 採点・設定選択はしていません。表示した断面は grid の形だけから決めた中心断面で、振幅を見て選んではいません。数値・hash・規則の正本は [manifest.json](manifest.json) です。

| 手法 | C3 の cell | FORGE の cell |
|---|---|---|
| POCS | `a01_b05` | `a04_b01` |
| DRR | `a05_b03` | `a03_b02` |
| NeRSI | `a02_b03` | `a04_b02` |
| CCNet-5D | `a05_b02` | `a02_b05` |
| ReGSI | `a02_b05` | `a05_b04` |

**図**（各図は 178 × 142 mm、PNG 1200 dpi と PDF。1行目は正解と入力、2行目は5手法の予測、3行目は残差です）

| dataset | 変化させる軸（他の軸は中心で固定） | trace 数（観測 / validation / test） | 物理振幅 | trace RMS 版 |
|---|---|---|---|---|
| C3 | 受振点 | 32（5 / 2 / 25） | `c3/gather_receiver_grid_point_physical` | — |
| C3 | 震源点 | 32（9 / 0 / 23） | `c3/gather_source_grid_point_physical` | — |
| C3 | 震源測線 | 16（2 / 0 / 14） | `c3/gather_source_grid_line_physical` | — |
| FORGE | 受振点 | 32（2 / 3 / 27） | `forge/gather_receiver_grid_point_physical` | `forge/gather_receiver_grid_point_trace_rms` |
| FORGE | 震源点 | 32（8 / 0 / 24） | `forge/gather_source_grid_point_physical` | `forge/gather_source_grid_point_trace_rms` |
| FORGE | 震源測線 | 6（1 / 0 / 5） | `forge/gather_source_grid_line_physical` | `forge/gather_source_grid_line_trace_rms` |
| FORGE | 受振測線 | 8（3 / 0 / 5） | `forge/gather_receiver_grid_line_physical` | `forge/gather_receiver_grid_line_trace_rms` |

中心の位置は C3 が source line 8・source point 16・receiver line 4・receiver point 16、FORGE が source line 3・source point 16・receiver line 4・receiver point 16（FORGE の source cell 112、receiver cell 144）です。表示した trace はすべて clean_target です。FORGE の受振点ギャザーでは、seed 7321 の mask で観測が32本中2本しかありません。断面は幾何だけで決める規則なので、変えていません。

**表示の規則**
- 予測は保存済みの物理振幅です。FORGE の予測には、登録した主指標と同じ trace ごとの oracle RMS が既に掛かっています。
- 観測 trace は seed の入力（`observed_waveforms.npy`、正解と一致することを確認）をそのまま使うため、残差は0です。validation と test の trace は予測で埋めます。残差は「正解 − 補完後の予測」です。
- trace RMS 版は、すべてのパネルを正解 trace の全時間 RMS で割ります（RMS が0なら1で割ります）。
- 色の範囲は、dataset ごと・版ごとに、表示断面の正解の |振幅| の99パーセンタイルを共通に使います（C3 物理 45.83、FORGE 物理 0.01646、FORGE trace RMS 3.836）。手法ごとの正規化はしません。
- `seismic`、`interpolation='None'`、colorbar なし、フォントは DejaVu Sans（この環境に Arial がないため。study_060 と同じ）。手法の並びは POCS、DRR、NeRSI、CCNet-5D、ReGSI です。

**検算**：表示した validation / test の各 trace について、表示に使った配列から計算した trace SNR が、各 run の保存済み `*_per_trace.parquet` と一致することを確認しました。最大差は全手法・両 dataset で 0.0 dB です（`c3/acceptance.csv`、`forge/acceptance.csv`）。

PNG と PDF は容量のため Git に入れていません（`.gitignore`）。repository root から、新しい出力先を指定して再生成します（約1分）。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nice -n 19 .venv/bin/python scripts/export_study067_gathers.py \
  --c3 runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces \
  --forge runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge \
  --output results/study_067_forge_c3_common_grid/<new_output>
```
