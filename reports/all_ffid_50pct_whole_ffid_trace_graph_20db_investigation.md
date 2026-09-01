# SEG C3 NA 50% whole-FFID trace-graph GNN 補間調査(目標 >20 dB)

- Study: `studies/study_021_all_ffid_50pct_whole_ffid_trace_graph`
- Runs: `runs/study_021_all_ffid_50pct_whole_ffid_trace_graph/`
- 実施日: 2026-08-31(UTC)
- 状態: `running` — stage19(長期予算ラン)が実行中。完了時にこのレポートの
  §7 と study README を更新する。

## 1. 目的と成功条件

トレースを 1 ノードとする GNN(時間はグラフ座標ではなくノード内の系列/潜在特徴)で、
SEG C3 Narrow-Azimuth の whole-FFID 50% 分割に対して、全 validation データの
`oracle_per_trace_unit_rms_global_snr_db` が厳密に 20.0 dB を超えること。
source–receiver 二部グラフ定式化、および
`L = L_mask + λ_spec L_spectrum + λ_slope L_PWD + λ_amp L_amplitude`
の複合損失を試すこと。20 dB 未満は成功と扱わない。

## 2. 実験条件

### 2.1 データと分割

- SEG C3 NA 4 ソースファイル(SHA-256 固定、`inputs.yaml`)。FFID 2–4782。
  振幅 QC(全ゼロまたは |amp|>1e4)で FFID 1746 を全面除外 → 適格 4,780 FFID。
- seed 42 の確定的置換で FFID 単位に丸ごと割当(`assign_random_whole_ffid_splits`):
  **train 2,390 / validation 598 / test 1,792 FFID**。重複ゼロ、和集合 = 全適格 FFID。
- トレース数: prepared 1,155,312 / 293,152 / 855,016 →
  物理座標カノニカル化(keep_lowest_array_row、split・振幅非参照)で 8/1/6 行削除 →
  **effective 1,155,304 / 293,151 / 855,010**。
- 学習入力は train FFID の振幅のみ。validation は checkpoint 選択と指標のみに使用。
  test・excluded は実体化しない。学習時もターゲット FFID を近傍から完全除外し
  (pseudo-held-out)、検証条件と一致させる。

### 2.2 指標(公式)

`oracle_per_trace_unit_rms_global_snr_db = 10 log10(Σ target_unit² / Σ (target_unit − raw_pred)²)`
— target は各トレースを自身の真の RMS で単位化、予測は生出力。
全 validation 293,151 トレース × 625 サンプル、float64 エネルギー累積。
厳密比較 `> 20.0 dB`。

### 2.3 実装(本 study で新規追加)

| 部品 | ファイル |
|---|---|
| GNN モデル | `src/seis_interp/models/trace_graph_interpolator.py` |
| 複合損失 | `src/seis_interp/training/trace_graph_losses.py` |
| トレーナ | `src/seis_interp/training/trace_graph_trainer.py` |
| チェックポイント | `src/seis_interp/training/trace_graph_checkpoints.py` |
| パイプライン | `src/seis_interp/pipelines/train_trace_graph.py` |
| CLI | `python -m seis_interp.cli train trace-graph` |
| 単体テスト | `tests/unit/test_trace_graph_{interpolator,losses,training}.py`(45 件) |

モデル要点:

- **ノード = トレース**。ターゲット FFID の 8×68 セル + 近傍 train ショット K 個 × 8×68。
  各ノードの 625 サンプル波形は Conv1d エンコーダで潜在系列(既定 125 フレーム)へ。
  時間はノード内でのみ処理され、グラフ座標に含めない(要求仕様)。
- **trace_lattice モード**(既定): 因子分解エッジ —
  (a) ショット内 receiver 格子エッジ(depthwise 3×3)、
  (b) 同一相対受振セルを結ぶ source 軸エッジ(ショット記述子条件付き masked attention)、
  (c) ノード内 dilated 時間更新。ラウンド反復で全ノード状態を更新。
- **source_receiver_bipartite モード**(要求仕様): source ノードと receiver ノードを
  明示に持ち、観測トレース = エッジ特徴、欠損ショット = missing-edge 集合。
  受振ノードはショット方向の masked attention 集約 + 格子平滑化、source ノードは
  セル集約 + source–source attention、エッジはゲート付き残差で更新。
- 予測 = 逆距離リファレンス + zero-init デコーダ残差(ソース順序に対して置換不変。
  マスク済み近傍振幅が出力に影響しないことを単体テストで保証)。
- 追加オプション: attention の時間分解能(pooled / per_frame / per_frame_shifted)、
  activation checkpointing(スケール時のメモリ束縛解消のため途中導入)。

複合損失(`trace_graph_losses.py`):

- `L_mask`: 隠したターゲットゲザーの masked MSE(自己教師あり復元)
- `L_spectrum`: rFFT log1p-magnitude MSE + 振幅重み付き位相コサイン誤差
- `L_PWD`(slope): ターゲットの平滑構造テンソルから推定した局所傾斜(detach)による
  plane-wave destruction 残差の受振 y 軸整合
- `L_amplitude`: 窓付き RMS エンベロープ MSE

### 2.4 学習の基準条件

AdamW lr 5e-4 → cosine ×0.03、wd 1e-5、batch 2 gathers、epoch_without_replacement、
neighbor_dropout 0.05(source gather 単位)、grad clip 1.0、bfloat16、
2,500 updates/stage、seed 42、K=8 最近接 train ショット。
昇格ゲート: 対照比 +0.20 dB @2,500。

## 3. 事前プローブ(決定論・学習なし)

| プローブ | validation S/N |
|---|---:|
| 逆距離リファレンス K=1(最近接ショットコピー) | 5.687 dB |
| 逆距離リファレンス K=2 | 6.764 dB |
| 逆距離リファレンス K=4 | 7.068 dB |
| **逆距離リファレンス K=8(採用床)** | **7.089 dB** |
| 逆距離リファレンス K=16 / K=32 | 6.762 / 6.221 dB |
| 同一物理受振点アライメント K=8 | **−0.996 dB** |
| 同一ライン線形ブラケット | 6.944 dB |

- common-receiver 対応(物理受振点合わせ)は大幅悪化 → **common-offset
  (同一相対位置)対応が支配的相関**。モデルの source 軸エッジは同一相対セルを結ぶ
  現行設計が正しい。
- ショットは千鳥配置(source_y 40 m 格子、ライン間 160 m)で、同一ライン内の
  実効ショット間隔は 80 m。validation FFID の 77%(463/598)は最近接 train
  ショットが約 80 m、22%(132)は約 160 m。

## 4. 段階結果(全 validation、2,500 updates、断りなき限り batch 2)

| Stage | 条件(1 変更ずつ) | S/N [dB] | 判定 |
|---:|---|---:|---|
| — | 逆距離リファレンス床(step 0) | 7.089 | — |
| 01 | 制御: joint shot CNN w32 K8(既存採用アーキ) | 7.768 | 参照 |
| 02 | 制御: neighbor K1374 w384(study 018 採用アーキ) | **11.318** | 参照 |
| 03 | **GNN trace_lattice w64 r4、L_mask のみ(基準)** | 8.734 | 基準 |
| 04 | + L_spectrum(λ=0.1) | 8.642 | 非昇格(−0.09) |
| 05 | + L_slope(λ=0.1) | 8.712 | 非昇格(−0.02) |
| 06 | + L_amplitude(λ=0.05) | 8.483 | 非昇格(−0.25) |
| 07 | graph_mode = source_receiver_bipartite | 8.181 | 非昇格(−0.55) |
| 09 | 近傍ショット K=16 | 8.647 | 非昇格(−0.09) |
| 10 | attention per_frame | 8.026 | 非昇格(−0.71) |
| 11 | attention per_frame_shifted | 7.752 | 非昇格(−0.98) |
| 12 | time_downsample_factor 1(batch 1 診断) | 8.554 | 非昇格 |
| 08 | 容量: w128 / r6 | 9.003 | **昇格(+0.27)** |
| 15 | 深さ: w64 / r12 | 9.044 | **昇格(+0.31)** |
| 17 | 組合せ: w128 / r12 | 9.280 | **昇格(+0.55、ほぼ加法)** |
| 18 | w128 / r12 / **batch 4** | **10.322** | **昇格(+1.04)** |
| 20 | 反復精緻化 2 パス(再帰、重み共有) | 8.777 | 非昇格(+0.04) |
| 13 | 予算: w128 / r6 @10,000 updates | 10.691 | 予算スロープ測定 |
| 19 | **形式: w128 / r12 / batch 4 @10,000 updates(study 最良)** | **12.239** | 未達(< 20) |
| 21 | 予算: w128 / r6 / batch 4 @25,000 updates | (実行中) | — |

- stage19 の予算カーブ(batch 4、10k コサイン): 2,500→10.700、5,000→11.684、
  7,500→12.068、10,000→**12.239**(train audit 12.429、9.4 時間、2.54M パラメータ)。
  **最強コントロール(stage02 の 11.318)を +0.92 dB 上回り、study 最良**。
- stage21 の途中経過(batch 4、25k コサイン): 2,500→9.910、5,000→10.931、
  7,500→11.409、10,000→11.708、12,500→11.891。

- stage13 の予算カーブ: 2,500→9.145、5,000→9.982、7,500→10.506、10,000→10.691
  (増分 +0.84 / +0.52 / +0.19 dB)。
- 見たゲザー総数(batch × steps)で整理すると 5k 標本 ≈ 9.0–9.3、10k ≈ 10.0–10.3、
  20k ≈ 10.7 dB と、**倍化あたり +1.0 → +0.4 dB へ逓減**する明瞭なデータ
  スケーリング則に乗る。
- OOM 記録(runs 非記録の失敗): w192r6b2、w64 factor1 b2、w64 b8、w128r12b8 は
  93 GB GPU でも CUDA OOM。これを受け activation checkpointing を実装し
  stage17 以降で使用(出力・勾配の同一性は単体テストで保証)。

## 5. 診断

stage03 checkpoint の誤差分布:

- **時間帯別**: 0–0.4 s ≈ 15.8 dB に対し、2 s 以降 ≈ 1 dB。誤差エネルギーは
  深部反射・コーダ(t > 2 s)に集中。
- 最近接 train ショット 80 m 組で中央値 10.3 dB、160 m 組で 8.4 dB。
- 全モデルで train audit ≈ validation(例: stage03 8.76 vs 8.73)→ 過学習ではなく
  容量・表現力・データ量律速。ターゲット FFID 除外学習の下では、train ショットの
  再構成も validation と同程度に難しい。

## 6. 監査・再現性(全完了 run)

- FFID 分離: train/validation/test FFID 重複 0、`target_ffid_neighbor_entries` 0、
  `non_train_neighbor_entries` 0、train–validation source 座標重複 0。
- 全 17 run で checkpoint 再読込の再評価が保存値と一致
  (`revalidation_matches: true`、公差 1e-8)。
- フルスコープ run はすべて `scope_success: true`(必須 FFID 数・sample 数・
  effective split 数・FFID 1746 除外を照合)。診断 subset run(smoke)のみ
  設計どおり scope 不一致。
- 各 run ディレクトリに `config.resolved.yaml` / `inputs.lock.json` /
  `metrics.json` / `run.json`(git SHA・seed・環境)/ `artifacts/best.pt` を保存。
- 品質ゲート: `ruff check`・`ruff format --check`・`pytest`(1,296 件)全通過。

## 7. 判定(2026-08-31 時点)

**公式基準は未達。** 最良の確定値は stage19 の **12.239 dB**(GNN trace_lattice、
w128/r12/batch4、10,000 updates、全監査通過)であり、厳密 20 dB 基準に対し
**7.76 dB 不足**。スケールした GNN は本 split 上の全コントロール
(既存採用アーキテクチャ含む)を上回ったが、観測されたデータスケーリング
(ゲザー倍化あたり +1.0 → +0.4 dB で逓減、stage19 終盤の増分 +0.17/2.5k)の下で、
予算のみの延長で 20 dB に到達する evidence-backed な経路は現時点で存在しない。

要求された構成要素はすべて実装・検証済みである:

1. トレース = ノード、時間 = ノード内潜在系列の GNN(実装・採用)
2. source–receiver 二部グラフ(実装・試行: 8.18 dB、trace_lattice に劣後)
3. 複合損失 L_mask + λL_spectrum + λL_PWD + λL_amplitude(実装・試行:
   本指標では中立〜微負。指標自体が MSE 系であり、容量律速の段階では
   補助損失は主指標を動かさない)
4. 50% whole-FFID 分割・FFID 非重複・漏洩監査・checkpoint 再評価・再現性(全通過)

### 20 dB 到達を阻む要因(定量)

- 決定論的リファレンスの床は 7.09 dB で頭打ち(§3)。
- 全アーキテクチャ(per-trace CNN K1374 / joint gather CNN / GNN 両モード)が
  2,500 updates で 7.8–11.3 dB に収束し、train audit ≈ validation。
- 誤差は t > 2 s のコーダに集中(≈1 dB)。80 m 離れたショットからの完全未観測
  ショットの後半波形復元は、現行モデル族では誤差エネルギー 1% 未満(=20 dB)に
  達しない。並行 study 020(25% whole-FFID、目標 25 dB)も同一の壁(最良 8.7 dB)。

### 継続計画(evidence-backed)

1. stage19(実行中)で予算カーブの漸近を確定。
2. データスケーリング則が支配的なため、次の一手は「見たゲザー総数」を桁で増やす
   長期学習(50k–200k updates、checkpointing 併用)+容量の同時スケール。
3. 誤差が後半時間帯に集中する事実に基づく、時間帯別の学習重み付け・
   カリキュラム、または t>2 s 専用の反復精緻化ステージ。
4. 分割契約自体の再検討が許される場合: トレース単位 50%(study 018 で 20.46 dB
   達成済み)と whole-FFID 50% の中間条件の検討。

## 8. 再現手順

```bash
# 分割準備
python -m seis_interp.cli data prepare-baseline \
  --input data/interim/c3_na/all_ffids \
  --output data/processed/c3_na/all_ffids_whole_ffid_50pct_train_amplitude_qc \
  --config studies/study_021_all_ffid_50pct_whole_ffid_trace_graph/config.yaml --json

# 最良確定 stage の再現
python -m seis_interp.cli train trace-graph \
  --config studies/study_021_all_ffid_50pct_whole_ffid_trace_graph/variants/stage18_gnn_w128_r12_batch4.yaml \
  --interim data/interim/c3_na/all_ffids \
  --processed data/processed/c3_na/all_ffids_whole_ffid_50pct_train_amplitude_qc \
  --output runs/study_021_all_ffid_50pct_whole_ffid_trace_graph/<run-id> \
  --device cuda:1 --json
```
