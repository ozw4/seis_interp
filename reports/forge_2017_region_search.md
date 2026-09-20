# FORGE 2017：規則グリッドに近い調査領域の探索

2026-09-20。取得済み全1,106ファイルを対象に、1,575件の局所格子と1,200組の震源・受振領域を比較し、小・中・大で各5件、計15件を候補として保存した。

**投影・内挿の予備検証には小領域S1、トレース数を確保する比較候補として中領域M1を推奨する。** S1は4,096セルすべてに1本ずつ観測があり、受振点の投影距離95%点は0.36 m。ただし震源側は37.83 m、最大90.57 mであり、波形をそのまま移動して誤差を無視できるとは言えない。M1は49,152セル中49,151セルが利用可能だが、震源の95%距離が90.01 mに増える。

事前の目安「正規化投影距離95%点≤0.1、充足率≥90%、衝突率≤1%」をすべて満たす候補は、今回の探索範囲では0件だった。閾値を緩めて合格扱いにはしていない。領域の最終採用と波形の投影・内挿は未実施。

[実行サマリー](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/summary.json) / [全組合せの指標](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/candidate_metrics.csv) / [候補15件](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/finalists.csv) / [全局所格子の指標](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/station_candidates.csv) / [格子定義と対応表の場所](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/candidate_grids.json)。

## サイズごとの先頭候補

| 候補 | 空間形状：震源×受振 | 有効トレース | 4D充足率 | 震源距離p95 | 受振距離p95 | 衝突率 | 全時間float32容量 |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | 4×16 × 4×16 | 4,096 | 100.0000% | 37.83 m | 0.36 m | 0.0000% | 0.066 GB |
| M1 | 6×32 × 8×32 | 49,151 | 99.9980% | 90.01 m | 0.57 m | 0.0000% | 0.787 GB |
| L1 | 8×48 × 12×48 | 221,349 | 96.7000% | 134.52 m | 0.81 m | 6.4848% | 3.540 GB |

容量は4,001時刻・1 ms間隔を含む1ボリュームの値。モデルの作業領域・勾配・チェックポイントのメモリは含まない。空間軸順は `[source_grid_line, source_grid_point, receiver_grid_line, receiver_grid_point]` とし、時間を加えて5Dになる。

![サイズ・充足率・投影距離の比較](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/tradeoffs.png)

正規化投影距離は `sqrt((測線方向のずれ/50 m)^2 + (直交方向のずれ/測線間隔)^2)`。震源・受振点ごとに95%点を求め、その大きい方を図に示す。異方的な格子のため、0.1を一律「5 m」と読み替えることはできない。格子を細かくし直して比較した結果でもない。

## 先頭候補の配置と範囲

### S1：small

候補ID: `small_source_b02_u3300.0_receiver_b15_u1700.0`。

震源は64地点、格子角度-4.012°（UTM東向き基準）、測線方向50 m・直交方向326.057 m。投影距離の中央値18.47 m、95%点37.83 m、最大90.57 m。セル領域の面積は1.043 km²。

UTM範囲: E=335179.1–335969.3 m、N=4263081.3–4264116.6 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 503 | 122–137 | 16 |
| 504 | 106–121 | 16 |
| 505 | 136–151 | 16 |
| 506 | 154–169 | 16 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

受振は64地点、格子角度88.971°（UTM東向き基準）、測線方向50 m・直交方向200.000 m。投影距離の中央値0.20 m、95%点0.36 m、最大0.48 m。セル領域の面積は0.640 km²。

UTM範囲: E=335821.3–336434.3 m、N=4261227.3–4261987.4 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 161 | 511–526 | 16 |
| 165 | 511–526 | 16 |
| 169 | 511–526 | 16 |
| 173 | 511–526 | 16 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

DC要確認率11.279%、振幅要確認率0.000%、ほぼ定数要確認率0.000%。offset中央値2133.5 m、5–95%範囲1427.2–2797.3 m。QC率は固定除外後トレースを分母にする。

S1は配置の予備検証を小さな規模で始める候補。震源506の区間端に曲がりがあり、最大投影距離90.57 mの点も残している。小領域の5候補は同じ震源領域で、受振領域の位置が異なる。S5はDC要確認率が約6.93%だが、この率を理由に順位を変更していない。

![S1実座標と投影ベクトル](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u1700.0_geometry.png)

[占有・欠測・衝突の図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u1700.0_occupancy.png) / [トレース対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u1700.0/trace_mapping.parquet)。

### M1：medium

候補ID: `medium_source_b06_u3400.0_receiver_b13_u1900.0`。

震源は192地点、格子角度-15.521°（UTM東向き基準）、測線方向50 m・直交方向352.899 m。投影距離の中央値48.82 m、95%点90.01 m、最大115.79 m。セル領域の面積は3.388 km²。

UTM範囲: E=335116.7–337081.4 m、N=4260848.8–4262911.1 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 507 | 161–192 | 32 |
| 508 | 157–188 | 32 |
| 509 | 155–186 | 32 |
| 510 | 153–184 | 32 |
| 511 | 151–182 | 32 |
| 512 | 149–180 | 32 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

受振は256地点、格子角度88.971°（UTM東向き基準）、測線方向50 m・直交方向200.000 m。投影距離の中央値0.33 m、95%点0.57 m、最大3.41 m。セル領域の面積は2.560 km²。

UTM範囲: E=335417.8–336844.7 m、N=4261020.3–4262594.6 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 153 | 507–538 | 32 |
| 157 | 507–538 | 32 |
| 161 | 507–538 | 32 |
| 165 | 507–538 | 32 |
| 169 | 507–538 | 32 |
| 173 | 507–538 | 32 |
| 177 | 507–538 | 32 |
| 181 | 507–538 | 32 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

DC要確認率0.716%、振幅要確認率0.142%、ほぼ定数要確認率0.000%。offset中央値901.3 m、5–95%範囲243.3–1643.3 m。QC率は固定除外後トレースを分母にする。

M1は衝突なしでほぼ完全な格子を確保する候補。欠けた1セルは既存ヘッダー除外によるもの。震源の測線間隔・向きの差が残るため、投影または内挿が波形へ与える誤差の確認が必要。

![M1実座標と投影ベクトル](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3400.0_receiver_b13_u1900.0_geometry.png)

[占有・欠測・衝突の図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3400.0_receiver_b13_u1900.0_occupancy.png) / [トレース対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3400.0_receiver_b13_u1900.0/trace_mapping.parquet)。

### L1：large

候補ID: `large_source_b05_u2400.0_receiver_b01_u3000.0`。

震源は385地点、格子角度-15.764°（UTM東向き基準）、測線方向50 m・直交方向333.042 m。投影距離の中央値53.20 m、95%点134.52 m、最大166.80 m。セル領域の面積は6.394 km²。

UTM範囲: E=333678.3–336596.5 m、N=4260833.4–4263709.5 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 506 | 135–183 | 49 |
| 507 | 133–180 | 48 |
| 508 | 129–176 | 48 |
| 509 | 127–174 | 48 |
| 510 | 125–172 | 48 |
| 511 | 123–170 | 48 |
| 512 | 121–168 | 48 |
| 513 | 119–166 | 48 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

受振は576地点、格子角度88.973°（UTM東向き基準）、測線方向50 m・直交方向200.000 m。投影距離の中央値0.47 m、95%点0.81 m、最大1.21 m。セル領域の面積は5.760 km²。

UTM範囲: E=333030.4–335271.5 m、N=4261748.0–4264136.1 m。

| 測線 | 含まれる測点の最小–最大 | 地点数 |
|---|---|---:|
| 105 | 521–568 | 48 |
| 109 | 521–568 | 48 |
| 113 | 521–568 | 48 |
| 117 | 521–568 | 48 |
| 121 | 521–568 | 48 |
| 125 | 521–568 | 48 |
| 129 | 521–568 | 48 |
| 133 | 521–568 | 48 |
| 137 | 521–568 | 48 |
| 141 | 521–568 | 48 |
| 145 | 521–568 | 48 |
| 149 | 521–568 | 48 |

最小–最大は索引の便宜で、正確な所属は保存した対応表を参照する。

DC要確認率0.689%、振幅要確認率0.018%、ほぼ定数要確認率0.000%。offset中央値1651.5 m、5–95%範囲484.5–3070.8 m。QC率は固定除外後トレースを分母にする。

L1は拡張時の比較用。震源の有限格子は8×48=384節点だが、物理区間に385地点が含まれ、点を間引いて本数を揃えていない。有効トレース数が格子セル数を上回っても、空セルと衝突は同時に発生する。支持点のない空セル6,912、QC後に空になるセル387、衝突トレース割合6.485%であり、最初の比較実験には優先しない。

![L1実座標と投影ベクトル](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b05_u2400.0_receiver_b01_u3000.0_geometry.png)

[占有・欠測・衝突の図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b05_u2400.0_receiver_b01_u3000.0_occupancy.png) / [トレース対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b05_u2400.0_receiver_b01_u3000.0/trace_mapping.parquet)。

## 候補15件の比較

同じサイズの候補間では、有効トレース集合のJaccard重複率が80%を超える組合せを除いている。別サイズ間の独立性や、学習・評価領域の独立性を保証する選択ではない。

| 候補 | 充足率 | 正規化距離p95 | 震源距離p95 | 衝突率 | DC要確認率 | 振幅要確認率 | offset中央値 | 配置図・対応表 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | 100.0000% | 0.1233 | 37.83 m | 0.000% | 11.279% | 0.000% | 2133.5 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u1700.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u1700.0/trace_mapping.parquet) |
| S2 | 100.0000% | 0.1233 | 37.83 m | 0.000% | 9.375% | 0.000% | 1993.4 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u1850.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u1850.0/trace_mapping.parquet) |
| S3 | 100.0000% | 0.1233 | 37.83 m | 0.000% | 8.545% | 0.000% | 1899.8 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u1950.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u1950.0/trace_mapping.parquet) |
| S4 | 100.0000% | 0.1233 | 37.83 m | 0.000% | 7.324% | 0.000% | 1805.8 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u2050.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u2050.0/trace_mapping.parquet) |
| S5 | 100.0000% | 0.1233 | 37.83 m | 0.000% | 6.934% | 0.317% | 1713.5 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/small_source_b02_u3300.0_receiver_b15_u2150.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/small_source_b02_u3300.0_receiver_b15_u2150.0/trace_mapping.parquet) |
| M1 | 99.9980% | 0.3249 | 90.01 m | 0.000% | 0.716% | 0.142% | 901.3 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3400.0_receiver_b13_u1900.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3400.0_receiver_b13_u1900.0/trace_mapping.parquet) |
| M2 | 99.9980% | 0.3249 | 90.01 m | 0.000% | 0.808% | 0.149% | 889.2 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3400.0_receiver_b13_u2100.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3400.0_receiver_b13_u2100.0/trace_mapping.parquet) |
| M3 | 99.9959% | 0.3365 | 92.39 m | 0.000% | 0.739% | 0.144% | 898.5 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3300.0_receiver_b13_u2000.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3300.0_receiver_b13_u2000.0/trace_mapping.parquet) |
| M4 | 99.9959% | 0.3365 | 92.39 m | 0.000% | 0.645% | 0.146% | 893.4 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3300.0_receiver_b13_u2200.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3300.0_receiver_b13_u2200.0/trace_mapping.parquet) |
| M5 | 99.9959% | 0.3484 | 89.01 m | 0.000% | 0.777% | 0.136% | 924.2 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/medium_source_b06_u3200.0_receiver_b13_u1900.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/medium_source_b06_u3200.0_receiver_b13_u1900.0/trace_mapping.parquet) |
| L1 | 96.7000% | 0.5464 | 134.52 m | 6.485% | 0.689% | 0.018% | 1651.5 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b05_u2400.0_receiver_b01_u3000.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b05_u2400.0_receiver_b01_u3000.0/trace_mapping.parquet) |
| L2 | 96.6991% | 0.5464 | 134.52 m | 6.485% | 0.767% | 0.001% | 1809.6 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b05_u2400.0_receiver_b01_u3350.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b05_u2400.0_receiver_b01_u3350.0/trace_mapping.parquet) |
| L3 | 96.6978% | 0.5332 | 132.18 m | 5.735% | 0.873% | 0.000% | 1831.2 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b05_u2550.0_receiver_b01_u3200.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b05_u2550.0_receiver_b01_u3200.0/trace_mapping.parquet) |
| L4 | 96.6838% | 0.5539 | 131.44 m | 6.489% | 2.236% | 0.001% | 2212.6 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b01_u3300.0_receiver_b01_u3000.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b01_u3300.0_receiver_b01_u3000.0/trace_mapping.parquet) |
| L5 | 96.6838% | 0.5723 | 142.05 m | 7.739% | 0.824% | 0.001% | 1962.1 m | [図](../runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4/figures/large_source_b04_u2750.0_receiver_b01_u3350.0_geometry.png)・[対応表](../data/interim/forge_2017/20260920T014514223263Z_5d660f87b9b4/candidates/large_source_b04_u2750.0_receiver_b01_u3350.0/trace_mapping.parquet) |

## 探索方法と適用範囲

入力はstudy_049の全配置監査、study_048の波形測定表、study_050の固定QCマスク。入力Parquetと前回runの系譜をハッシュ照合した。全ゼロ35本・非ゼロ定数100本の除外と既存ヘッダー除外を維持し、残る1,919,596本を母集団とした。座標不明の受振点141/515は周囲から推定せず、候補に割り当てない。

1. 連続した測線ブロックを1測線ずつ移動し、測線内の物理区間を200 m刻みで走査する。震源は測線ID差1、受振点はID差4を連続とする。測点IDが測線間で一致することは要求しない。
2. 区間内の隣接点方向の軸方向中央値から局所回転を求める。震源・受振点は独立した直交格子を持つ。測線方向は50 m、受振点の直交方向は200 m。震源の直交間隔は、各測線の直交座標中央値を物理順に並べた直線回帰で推定する。
3. 測線方向の共通格子位相は周期的な二乗距離を1 m刻み、最良位置付近を0.1 m刻みで比較する。測線ごとの独立した位相補正はしない。探索中心に最も近い格子包絡を選び、半開区間に含まれる対象測線の既知座標点をすべて含める。最終包絡は格子位相に合わせて走査区間から最大25 m移動する。
4. 直交方向は対象測線の全点を保持し、有限格子の最も近い節点へ対応させる。端の外れ点も残し、端節点への移動距離全体を測る。該当点は`edge_clamped`で識別する。格子の形状は目標節点数であり、実地点数を強制的に一致させない。
5. 充足率・正規化距離・衝突率の3順位から順番に候補を取り、同じ地点集合を重複して採らない。上位20地点集合の周囲±200 mを50 m刻みで再走査し、最終的に各役割・サイズで最大20候補を残す。
6. 各サイズ20×20=400組の実トレースを評価する。目安達成を優先し、充足率降順→正規化距離昇順→衝突率昇順→候補ID順で順位付けする。QC率は順位に使わない。

局所格子は震源658件・受振917件、計1,575件。局所探索・候補絞り込みを含むため、全位置・全回転・全格子間隔についての大域的な最適解ではない。今回の基準間隔と3サイズで見つかった候補として解釈する。目安未達は「FORGE全域に適した領域が存在しない」という証明ではない。

充足率の分子は有効観測があるユニークな4Dセル数。衝突率は複数の有効観測が入るセルに属する全有効トレース数／有効トレース数。支持点がない空セル、支持点はあるが記録がないセル、記録はあるがQC後に空になるセルを分けた。衝突観測を平均したり1本だけ残したりしていない。

## 保存物と再現

[設定](../studies/study_051_forge_region_search/config.yaml) / [入力契約](../studies/study_051_forge_region_search/inputs.yaml) / [判断理由](../studies/study_051_forge_region_search/decisions.md)。

各最終候補のディレクトリに`trace_mapping.parquet`、`source_stations.parquet`、`receiver_stations.parquet`、`grid_cells.parquet`を保存した。元の`source_file, trace_index`、実座標、格子座標、整数添字、投影距離、QCフラグを保持する。deadヘッダーの座標値は`*_header_x_m, *_header_y_m`に残し、実座標列には他ショットでも確認された既知のstation座標を入れる。QC除外行も対応表に残し、`eligible_after_fixed_qc`で選別する。未記録セルには架空の波形行を作らない。

同一候補のセル番号は `source_cell * receiver_cell_count + receiver_cell`、各役割のセル番号は `grid_line * n_point + grid_point`。物理順の測線並びは`physical_line_order`に記録する。測線IDの大小と格子添字の増加方向は一致するとは限らない。逆変換に必要な原点・基底・間隔・開始位置を`candidate_grids.json`に保存する。

実行コマンド:

```bash
.venv/bin/python scripts/search_forge_regions.py
```

実行run: `runs/study_051_forge_region_search/20260920T014514223263Z_5d660f87b9b4`。既存の原データ・QCマスクは変更していない。波形の読込・加工・内挿は実行していない。

検証コマンド:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest tests/unit/test_forge_region_search.py tests/integration/test_forge_region_search.py tests/unit/test_forge_geometry_qc.py tests/integration/test_forge_geometry_audit.py
```

静的検査・書式検査は成功、関連14テストが成功。回転格子、測点番号のずれ・逆順、曲がり、外れ点、欠測、衝突、QC除外、座標不明点、決定的選択、入力破損時の停止を検証した。実データの15候補では入力・出力ハッシュ、全対応行のQC一致、投影距離、セル集計、候補間重複率を独立に再計算して照合した。

## 次の実験への引継ぎ

S1で、人工欠測を分割した後の観測トレースだけを使い、最近傍投影と空間内挿による波形変化を比較する。元の観測位置に戻した予測と実測波形で評価し、未取得セルを正解として扱わない。今回の候補順位は波形品質の最終採用判断ではないため、時間窓と追加QCはその段階で固定する。今回の格子定義は座標だけから決めており、人工欠測の正解振幅を利用していない。
