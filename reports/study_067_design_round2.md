<!-- 2026-10-22 study_067 設計案（第2弾 pilot と本番）。Claude の設計ワークフロー（手法別6分析・統合・3批評・改訂）の出力。未承認の案であり、Q1–Q4 の回答で確定する。設計中にサブエージェントが CPU で validation のみの screening を74 run 実行しており、P-1/D-1 の規則は screening 由来として開示している。 -->

# study_067 最終設計：pilot第2弾と本番（2026-10-22 改訂）

## (1) 方針と目標

**目標（ユーザー規則、2026-10-22）**
- 5手法を2つの dataset で比較します。各手法の探索は5×5の grid を1つだけ使い、軸は現状維持で、両 dataset で同じ grid を使います。
- 候補の選択は dataset ごとに行い、validation の `masked_trace_relative_mse`（以下 relMSE）が最小の候補を選びます。
- 目標は次の2つです。
  - 各 dataset で ReGSI が最良であること（ReGSI の SNR は高いほど良い）。
  - 各手法で、S1 の25候補のうち最良の validation mean trace SNR が C3 で 10 dB 超、FORGE で 0.4 dB 超であること。
- 固定値は両 dataset 共通を原則とし、粗い調整だけを認めます。ReGSI を超えない範囲でベースラインの SNR が上がるのは歓迎です。CCNet の inner mask は ReGSI に合わせて 0.25 にします。
- 手順は次のとおりです。
  - S1：固定 split で25候補を学習・採点します。
  - barrier の後、3 seed（7311–7313、Q3 で変更を提案）で observed / validation / test を引き直します。
  - test は seed ごとに1回だけ採点し、平均 ± 標本SD を報告します。

**設計の立場**
1. ReGSI は V3 で凍結します。残る確認は R-1（C3 の k6 でも V3 が成り立つか）だけです。ReGSI 専用の新しい改良は入れません（R-3 は削除）。
2. ベースラインには、証拠のある粗い二値の改良だけを試します。採否の規則はすべて事前に登録し、規則は ReGSI のスコアを参照しません。
3. pilot は第2弾で終わりです（第3弾はありません）。S1 の結果を見た後は、どの手法の固定値も grid も変えません。
4. 床（下限目標）を下回る手法の救済は第2弾の中だけで行います。S1 で床割れが分かった手法は、ReGSI を含めてそのまま報告します。

**現状**（S1 固定 split の validation mean trace SNR、単位 dB、relMSE で選んだ候補の値。登録文書には予測値を載せません）

| 手法 | C3 | FORGE | 床に対する状態 |
|---|---|---|---|
| ReGSI V3 | 19.07（k2, [1,2]）。旧固定値の k6 は 20.01 | 3.01（k2, [1,2]） | 余裕あり |
| CCNet | 19.07（study_064、mask 0.1、a05_b02）。mask 0.25 は未測定 | 4 ms で最後まで完走した run がない（study_066 の [4,8,4,8] は 50k 更新後の予測で OOM） | **FORGE は未検証** |
| NeRSI | 12.66（P3、jitter 0.5） | **−0.69**（P3、relMSE 1.214 でゼロ予測より悪い） | **FORGE は床割れ** |
| POCS | 10.88（16反復、受振窓 [4,8]、overlap [2,4]） | 0.895 | C3 の余裕は 0.88 dB |
| DRR | 15.18 | 0.71 | FORGE の余裕が小さい |

## (2) 手法別の固定パラメータ

**状態ラベル**：【決定】決まっている／【pilot】第2弾 pilot の結果で決まる／【Q】ユーザーの回答待ち／【感度】報告のみで、何も決めない

**ニューラル3手法に共通の設定【決定】**
- 最適化：AdamW、学習率 1e-3 で一定、weight decay 0、50,000 更新。
- 1 更新あたりの loss trace は64本。loss は `masked_trace_mse`。
- EMA は 0.999（`fixed_final_ema`）。model seed は 7301。
- 新しく実行する run にはすべて shadow EMA（0.999 / 0.9995 / 0.9998）を記録します。shadow は診断専用で、選択には使いません。主予測（EMA 0.999）が shadow の有無で bitwise に一致することを test します。

**ReGSI（`regsi_real`）**：両 dataset とも 346,885 params（363,269 比 −4.51%）。grid は dilations {[1,2],[2,4],[3,6],[4,8],[6,12]} × k {2..6}。
- 【決定】
  - width 128、message passing 2 回、attention rms、`node_fourier_components` 0。
  - inner_mask 0.25。実効値は C3 で 0.2496（5,888/23,593）、FORGE で 0.2460（3,264/13,271）。
  - gradient clip 1.0。ReGSI だけの設定として宣言し、実測の発動率を併記します（C3 の S1 で平均 0.53%・最大 1.26%、FORGE で 0.01% 以下）。
- 【pilot R-1】V3 が C3 の k6 でも成り立つかを確認します。成り立たない場合は V2、V1、study_064 の値の順に戻します。CCNet の mask は、ReGSI で採用した値に従います。
- **凍結（不採用）**
  - wd 0.01（+0.013 dB）、EMA 0.9995（−0.007）、EMA 0.9998（−0.13）、mask 0.30（−0.04）。
  - 3 rounds（params +24.7%）、width 120/136 と time_downsample 1/4（いずれもパラメータ帯域外）、anisotropy（+0.004）。
  - augmentation と midpoint-free features（R-3）は、証拠のない ReGSI 専用の改良なので除外します。

**CCNet5D（`ccnet5d_grid`）**：hidden 53 / intermediate 63 / time kernel 3、363,292 params。grid は変えません。
- 【決定】
  - inner_mask 0.25 を両 dataset に適用します。C3 の25候補はすべて再実行します。study_064 の mask 0.1 の run は感度の比較にだけ使います。
  - patch の時間方向は full にします。
  - 精度：C3 は FP32（TF32 有効）。FORGE は学習も予測も bf16 です。FP32 だと a05 で約 110 GiB 必要になり、GPU 1 枚の 93 GiB を超えるためです。
  - FORGE の予測 core は [4,8,4,8]。
  - **乱数系列を独立にする（バグ修正）。** 現状は placement / mask / supervision の3つの generator がすべて 7301 で初期化されています（`forge_mvp_methods.py:439-440`、`ccnet5d_observed_patches.py:55-56`）。docstring にある「独立した系列」と食い違うため、`SeedSequence(7301).spawn(3)` に直します。CCNet は全 run を新しく実行するので、追加コストはありません。
- 【Q1】FORGE の学習手順を C3 と同じにする（`reuse_patch: false`、`cache_patch_gradients: false`）。cuDNN benchmark を使うかどうか。
- 【pilot C-0】FORGE のメモリ予算（現在 17/17/17/30/64 GiB）と速度。
- 【pilot C-1／Q4】time kernel 3→9（hidden 36 / intermediate 37、363,227 params）。既定は tk3 です。
- 【感度】
  - S-a：mask 0.1 と 0.25 を同じ cell で比べます。
  - S-b：FORGE で FP32 と bf16 を比べます。

**NeRSI（`nersi_profile`）**：C3 は 363,269 params。
- 【決定】Fourier 40、encoder 40、latent 28、decoder [29,14,8]、upsample [2,2,2]、kernel 5。2つの軸は変えません。
- 【Q2＋pilot N-1】
  - (a) FORGE に時間方向だけの×2段を3段加えます（s=3）。latent は 34×4 になり、params は 1,391,877 から 341,485 に減ります。C3 は変わりません。
  - (b) loss を各 trace の記録範囲に限ります（support-masked loss）。
  - (c) jitter 0.5 を残すかどうか（現状は採用済み）。
  - (d) EMA 0.9998 の救済策（NeRSI だけ）。
- **不採用**
  - λ を 0.1 より大きくする（C3 は λ 0.1 で既に 1.8–2.5 dB に崩れる）。
  - 周波数の hard cap（C3 で −0.45 dB）。
  - rank-2 latent（FORGE で −0.33〜−0.41 dB）。
  - encoder/decoder の幅の変更、profile batch の変更。

**POCS（`pocs_grid`）**
- 【決定】threshold は 1.0 から 0.01 へ等比で下げます。時間軸と震源軸は 50% overlap。CPU の単一スレッドで実行します。
- 【pilot P-1】K1 案：32 反復、受振窓 [4,8] で overlap [3,6]（75%）。不採用なら現状の16反復・overlap [2,4] のままにします。
- **不採用**
  - threshold_end 0.001（C3 で −2.5 dB）。
  - threshold_end 0.2（FORGE で 0.321 対 0.440）。
  - 受振窓 [8,8]（FORGE で −0.36）。
  - 予測への low-pass（+0.004 以下）。
  - K2 案。

**DRR（`drr_grid`）**
- 【決定】60 反復、damping 2、全帯域。
- 【pilot D-1／D-2】window から決める Hann 重み付き overlap を導入します。
  - line 軸は 0、point 軸は 50% です。対応は [4,6,4,6]→[0,3,0,3]、[4,8,4,8]→[0,4,0,4]、[6,8,6,8]→[0,4,0,4]、[8,12,8,12]→[0,6,0,6]、[8,16,8,16]→[0,8,0,8]。
  - 実装は既存の `interpolate_drr_volume_batched` を使います。境界では、軸長に合わせて window を切り詰め、最後の window を volume の内側にずらします。この境界規則を宣言します。
- **不採用**
  - 12 反復（C3 10.12、FORGE 0.82）。
  - 30 反復（C3 14.00）。
  - damping 1（C3 10.24、FORGE 0.449）、damping 3（C3 13.77、FORGE 0.054）。

## (3) pilot 第2弾（事前登録）

**採否の型**（どちらも ReGSI のスコアを参照しません）
- **T1**（ニューラル手法の構造・手順の変更。ReGSI の P2 と同じ規則）：FORGE で ΔSNR ≥ +0.2 dB、かつ C3 で ΔSNR ≥ −0.2 dB なら採用します。比較は指定したペアの cell で行います。
- **T2**（古典手法の固定値）：両 dataset で ΔSNR ≥ −0.05 dB、かつ少なくとも一方で ≥ +0.2 dB なら採用します。
- 共通事項
  - 指標は S1 固定 split の validation mean trace SNR です（FORGE は clean_target）。relMSE も併記します。
  - 全規則を、run の開始前に decisions.md へ登録します。
  - P-1 と D-1 は、screening の結果を見た後に作った規則です。このため「screening 由来（事後的、開示）」と明記します。

**G0：バグ検出用の gate**（C-0 と、各ニューラル手法の最初の FORGE run に適用）
- 基準 L0 は、同じ擬似 target に対するゼロ予測の loss です。厳密に計算し、FORGE では約 0.461 になります。
- 判定条件は次の3つです。
  - 更新 901–1000 の平均 loss が 0.95·L0 以下。
  - 同じ区間で出力 RMS / target RMS が 0.2 以上。
  - 全値が有限。
- 参考値：mask 0.1 の健全な run は、1000 更新で 0.86–0.88·L0 でした。張り付いた a05 の cached run は 1.000·L0 でした。
- 不合格のときもハイパーパラメータは変えません。Codex が最大1日、実装バグを探します（bf16 や padding の経路を、a01 の FP32 と比べる）。
  - バグが見つかった場合：修正して gate をやり直します。
  - 見つからない場合：その設定のまま実行し、ユーザーに報告します。張り付いた候補は relMSE の順位で最下位になるだけです。

| ID | 優先度 | 内容 | run | コスト | S1 への import | 判定規則 |
|---|---|---|---|---|---|---|
| R-1 | must | C3 の regsi a01_b05（[1,2], k6）を V3 の S1 config のまま、本番 runner で実行 | 1 | 2.5–4.4 GPU-h（混雑時は壁時計 ~10 h）、41 GiB | する | 19.812 以上（= 20.012 − 0.2）なら V3 を維持します。満たさない場合は同じ cell で V2、次に V1 を試し、最初に 19.812 以上になったものを採用します（追加 9 GPU-h 以下）。どれも満たさなければ study_064 の値に戻します。FORGE の +1.1 dB と引き換えに、C3 の看板 cell で最大 0.2 dB の損を許容することを明記します。 |
| C-0 | must | FORGE CCNet の起動 gate（mask 0.25、Q1 で決めた手順） | (a) a01–a05 の各行を d6 で 300 更新：s/更新と学習・予測の peak reserved。(b) a01_b02 と a05_b02 を 1000 更新し、予測→整列解除→採点→manifest まで通す（値は判断に使わない）。(c) a05 で cuDNN benchmark の on/off を 100 更新（Q1 が許す場合のみ）。(d) C-1 を行う場合は tk9 の a05 も測る | 4–6 GPU-h、68 GiB 以上空いていること | しない | G0 を適用します。メモリ予算は max(現在値, ceil(学習・予測の最大 reserved) + 2 GiB) とします。速度の値は自動で止める基準にせず、外部負荷を併記した総コスト見積りとしてユーザーに示します。Q1 で合意した上限を超えたときだけ開始前に報告します。cuDNN benchmark は 1.3 倍以上速く、かつ Q1 が許せば、CCNet の両 dataset で使います。 |
| C-1 | should（Q4 次第） | CCNet の time kernel 3 と 9 を同じパラメータ数で比較。証拠は study_057（relMSE 0.7198 対 0.8175）と study_060（0.7019 対 0.777）で、いずれも FORGE 2 ms のみ | C3 の a05_b02 で tk3/tk9（FP32）、FORGE の a03_b02 で tk3/tk9（bf16、毎更新新しい勾配）。4 run | 約30 GPU-h | 勝った方の2 run | T1。両 dataset の CCNet S1 より前に結論を出します。Codex の sprint 2 と並行して走るため、FORGE の critical path には乗りません。 |
| N-1 | must（Q2 次第） | FORGE NeRSI の床の救済 | FORGE の a02_b04（λ 0.001、base 1.15）、s=3。{jitter 0.5, なし} × {canvas 全体の loss, 記録範囲だけの loss} の 2×2 で4 run。shadow EMA を記録 | 1 GiB/run、4 run 同時で壁時計 4–8 h。GPU 相乗りの効率も同時に測定 | 採用した構成と一致する arm | (i) [jitter, 全体] − P3（−0.689）≥ +0.2 なら s=3 を採用（C3 は構造上変わらない）。不採用なら NeRSI の調整を止めて報告します。(ii) [jitter, 記録範囲] − [jitter, 全体] ≥ +0.2 なら support mask を採用（C3 では bitwise に不変であることを test で確認）。(iii) 採用した loss の下で、jitter なし ≥ jitter あり − 0.05 なら、jitter を両 dataset で外します（Q2 の同意が必要。C3 の S1 は study_064 から import できます）。(iv) 採用構成の EMA 0.999 での値が 0.55 dB 未満なら N-2 に進みます。 |
| N-2 | could（(iv) で発動） | NeRSI だけ EMA 0.9998 | C3 の a02_b04 を EMA 0.9998 で1 run | 0.5–3 GPU-h | — | 採用構成で 0.9998 の shadow が 0.999 より +0.15 dB 以上高く、かつ C3 が 10.5 dB 以上・Δ ≥ −0.3 なら、両 dataset で採用します。満たさなければ調整を止め、床割れの可能性を報告します。 |
| P-1 | should | POCS の K1 案（screening 由来） | 5×5 × {C3, FORGE} の50 run を本番 runner で実行（validation のみ） | 45–70 CPU-h、nice で並列25–50、壁時計約3 h | 採用時は50 run すべて | T2：現在の選択値（C3 10.877、FORGE 0.895）と比べます。開示事項：K1 は約15案の screening の最上位です（2 cell での値は C3 13.79/13.76、FORGE 1.081/1.166）。 |
| D-1 | must（実行中） | DRR の overlap（screening 由来） | D7：FORGE の rank 12、[4,8,4,8]、60 反復、overlap [0,4,0,4]。scratch 実装で実行中、約20 UTC に終了見込み | CPU | しない | T2：FORGE が 0.656 以上、かつ C3 が 15.125 以上なら採用します。C3 は 16.087 で達成済みです。本番実装で D7 と D5b（1.089）を 0.01 dB 以内で再現することを受け入れ条件にします。 |
| D-2 | must（D-1 を採用した場合） | 最も重い DRR cell（rank 30 × [8,16,8,16]、[8,12,8,12]）の時間と RSS を測る | 両 dataset で1–2 反復、worker 1、RSS 上限付き | CPU で数十分 | — | 測定した RSS から、window ごとの worker 数と同時候補数の上限を決めます。どれかの候補が壁時計 72 h を超える見込み、または DRR の S1 と seed が FORGE の GPU critical path に収まらない場合は、開始前に報告します。 |

**感度 run**（報告のみ。gate にしません。C-1 の後、GPU に空きがあるときに実行）
- **S-a**：FORGE CCNet の a03_b02 を mask 0.1 で実行します（約10–14 GPU-h）。C3 のペアは、S1 の a05_b02（mask 0.25）と study_064 の 19.073（mask 0.1）で追加コストなしです。
  - 事前登録：どちらかの dataset で 0.1 が 0.25 より 0.2 dB 以上良ければ、ReGSI − CCNet@0.1 を co-primary の感度として報告します。
  - C3 で tk3 を維持した場合は、study_064 の 0.1 grid の勝者を新 seed で3 run 追加します（約24 GPU-h）。FORGE は単一 cell の差だけを報告します。
- **S-b**：FORGE CCNet の a03_b02 を FP32 で実行します（約30 GiB、15–25 GPU-h）。
  - 事前登録：FP32 − bf16 ≥ 0.2 dB なら、FORGE の seed を始める前にユーザーへ報告します。選ばれた cell を FP32 で3 seed 実行する案を、co-primary の感度として提示します。a05 の場合は、厳密な activation checkpointing の実装が必要です。

**第2弾から削除したもの**
- R-2：FORGE ReGSI の S1 の最初の run として扱います。
- R-3、C-0 の cached 勾配の証拠 run、C3 の bf16 run（C-2）、POCS の K2 案、N-1 の a03_b01 cell。

**合計**
- 判断用 GPU は約 45–60 GPU-h で、そのうち約 20–25 GPU-h は S1 に import します。
- 感度 run は約 25–40 GPU-h、CPU は P-1 の 45–70 CPU-h と D-2 です。

**依存関係**
- Q1 と sprint 1 → C-0。
- R-1 → C-1 の C3 arm と、C3 CCNet の S1（mask が決まるため）。
- C-0 → C-1 の FORGE arm。
- Q2 と sprint 1 → N-1 → C3 NeRSI の S1。
- Q4 → C-1。
- sprint 2 → FORGE CCNet と FORGE ReGSI の S1。

## (4) 実装作業（Codex 向けの要点）

**Sprint 1（pilot に必要。目安 1.5–2 日）**
1. **CCNet の config**
   - 両 dataset で `inner_mask_fraction: 0.25` にします。
   - Q1 が A なら、FORGE の `reuse_patch` と `cache_patch_gradients` を false にし、dataset_differences.yaml と README からこの例外を削除します。
   - RNG を `SeedSequence.spawn` で独立にします。
   - manifest に次を記録します：1 更新あたりの patch 数、教師ありの trace 数、patch ごとの実効 inner 比率、would-clip 率。
2. **probe**（`probe_forge_ccnet_memory.py:34-44` の cached 必須の制約を外す）
   - timing mode（N 更新）と、end-to-end mode（予測→整列解除→採点→manifest）を追加します。
   - G0 の指標を出します：L0 の厳密値と出力 RMS 比。fresh mode では step 1 の loss が記録されないため、L0 を基準にします。
   - cuDNN benchmark を実行専用 key として追加します（既定は off。現状は `forge_mvp_methods.py:271-272` が全手法を deterministic に固定）。benchmark on のときの peak も記録します。
3. **NeRSI**
   - `canvas_method_config` に s = min{s : ceil(T_profile / 2^(3+s)) ≤ 48} を入れます（`time_upsample_channels=[28]*s`、kernel 3）。`parameter_count_policy` も更新します。
   - `loss_support_mask` を追加します。FORGE では整列後に各 trace の有効な 1,001 sample だけを使います。
   - test：C3 の予測が bitwise に不変で、import の identity にも影響しないこと。FORGE が 341,485 params になること。
4. **shadow EMA**：3手法すべてで `ema_eval_decays: [0.999, 0.9995, 0.9998]` を診断専用にし、identity から除外します。
   - 主予測が bitwise に一致することを手法ごとに test します。失敗した手法では shadow を使いません。
   - shadow は指標だけを保存し、予測全体は保存しません（ディスク使用率が 94% のため）。
5. **DRR**：`drr_grid` を `interpolate_drr_volume_batched`（`drr_batched.py:124`）に切り替えます。
   - **周波数 bin を `batch_matrix_limit` で分割します。** 現状は `max(1, …)` で、1 block 分の全 2049 bin を一度に処理します。FORGE の [8,16,8,16] では Hankel 行列の積み重ねだけで block あたり約 38 GiB になり、SVD を含めると worker 1 つで 100 GiB を超えます。
   - window から overlap を決める規則と、切り詰め・ずらしの境界規則を config と dataset_differences.yaml に宣言し、`c3_nonoverlapping_unknown_padding` の記述を置き換えます。
   - test：overlap 0 で現在の backend と一致すること（C3 の rank 12、[4,8,4,8]、30 反復で 14.0047）。両 shape で window が割り切れない場合。FORGE で D5b と D7 を再現すること。
6. **POCS**：K1 の pilot config を作り、本番 runner で実行できるようにします。
7. **R-1 の identity**：sprint 1 の変更後も ReGSI の resolved config が同一（または identity に影響しない）であることを確認します。R-1 と P2 V3 a01_b01 を import できる状態に保つためです。
8. **provenance と文書**
   - 誤った長さの記述を直します：forge/nersi_profile.yaml の 1008 → 2176、forge/ccnet5d_grid.yaml の 1001 → 2172（実行時の値）。
   - README：P3 が完了したこと、床の基準を best-of-25 に改めたこと（README 13 行目）。
   - decisions.md に第2弾の全規則と SHA256 の表を登録します。

**Sprint 2（FORGE の長い S1 に必要。目安 2 日）**
1. **厳密な再開**：2,500 更新ごと、または 30 分ごとに原子的に checkpoint を保存します。
   - 中身：model、AdamW、EMA と shadow、全 RNG（CCNet の3系列、ReGSI の sampler）、step、history、区間の集計値。
   - test：200 更新を通しで実行した結果と、100 更新→再開→100 更新の結果が bitwise に一致すること。FP32 と bf16（決定的 cuDNN）の両方で確認します。
2. **失敗への対応**
   - `before_prediction` の checkpoint から予測だけをやり直せるようにします。
   - OOM や基盤の失敗は最大2回まで自動で再実行し、失敗した試行は `attempts/<n>/` に移します。非有限 loss は diverged として記録し、再実行しません。
   - barrier は再実行の待ちがあれば待ちます。
   - 進捗が予定間隔の3倍途絶えたら、watchdog が止めて再実行します。
3. **予測メモリの事前確保**：学習の前に、最大 halo でゼロ入力の予測 forward を1回流します。run が終わるまで `empty_cache` を呼びません。全 GPU 手法が対象です。
4. **GPU 並列化**（現状は `candidate_schedule.py:28` が GPU の並列実行を禁止）
   - worker を `--shard i/n` で分割し、候補リストを重ならないように分けます。
   - 次の候補は「優先度が最も高く、いまの空きメモリに収まるもの」から選びます。結果ではなく、メモリだけで順序を決めます。
   - seed は2つの device に分けて並列に実行します。
   - Q1 が許せば、seed の学習と予測を barrier 前に始め、test の指標計算だけを barrier 後に行います。
5. **CCNet の patch 先読み**：CPU のスレッドで次の patch を準備します。RNG の消費順は変えず、bitwise に一致することを test します。GPU 側での patch 組み立ては採用しません。
6. **運用**
   - admission でディスクの空きを確認します。
   - nvidia-smi で外部の GPU 利用率を取り、manifest に記録します。
   - CPU 手法は nice で実行します。
   - 勾配ノルムの記録は、更新ごとの同期が発生しない場合だけ入れます。

**Sprint 3（起動後、最初の test 採点より前）**
- contrast / bootstrap 用の script を作ります。
  - grid.yaml に evaluation 節を追加します（RNG 670703、cluster は `source_cell`、族の定義）。
  - parquet に `prediction_energy` 列と FORGE の IDW scale 列を追加し、既存の run にも埋め戻します。
  - 受け入れ条件：study_063/064 の C3 ReGSI − CCNet = +0.920 dB、CI ≈ [0.83, 1.00] を再現すること。
- 評価 seed（Q3）が、どの namespace でも partition seed として使われていないことを確認します。
- 図表を作ります。

## (5) 公平性の宣言

以下を decisions.md、README、論文の補遺に載せます。

1. **手法ごとの protocol 表**
   - 項目：optimizer、学習率、wd、更新数、loss trace 数、inner mask（名目値と実効値）、データの再利用、clip（発動率つき）、EMA（shadow を含む）、精度（TF32 を含む）、勾配の鮮度、dataset ごとの params。
   - データの再利用は次のとおりです。ReGSI は同じ mask で C3 は 92 更新、FORGE は 51 更新を続けます（毎回新しい勾配）。CCNet は毎更新 patch を引き直し、隠した trace のうち 64 本だけを教師にします。NeRSI は観測した全 trace を当てはめます。
   - 共通でも宣言済みでもない項目があれば、`prepare --check-only` が失敗するようにします。
2. inner mask は入力を条件とする手法（ReGSI、CCNet）だけに適用します。support mask を採用した場合、それは NeRSI だけに適用します。座標ネットは、offset で位置が動く padding の段差を自分で合成しなければならないためです。
3. FORGE の CCNet の bf16 はメモリ上必要です。ReGSI が FP32 のままでいられるのは、厳密な省メモリ機能（checkpointing、pruning、fused aggregation）があるためです。この非対称を宣言し、大きさは S-b で測ります。
4. clip は ReGSI だけに適用し、発動率を示します。他の手法では would-clip 率（clip していたら発動した率）を記録します。
5. **パラメータ数の規則は1つ**にします。C3 ではニューラル3手法とも 363,269 ±5% に入ります。FORGE では C3 と同じ構造を使い、宣言した canvas 適応だけを例外にします（NeRSI の s=3 で 341,485、−6.0%）。
6. grid の値は sample 単位です。FORGE は 4 ms、C3 は 8 ms なので、dilation 6 は FORGE で 24 ms、C3 で 48 ms に当たります。
7. FORGE の loss は 2,172 sample の canvas 全体にかかります。有効な sample は 46% です（NeRSI は N-1 の結果に従います）。
8. **格子と実座標の差**
   - grid 系の手法は cell 中心の座標で、ReGSI と NeRSI は実座標で予測します。
   - binned offset から計算した遅延は、実 offset から計算した遅延と中央値 13 sample（p90 32）ずれます。ただし隣り合う trace 同士のずれは中央値 1 sample です。
   - 隣接 trace のゼロ lag 相関は、格子と実座標でほぼ同じです（震源点方向 0.285 対 0.285、受振点方向 0.135 対 0.151）。
   - study_060 では regsi_grid と regsi_real の差は約 0.07–0.1 dB でした。
9. FORGE の SNR は全手法で、trace ごとの oracle test RMS を使います。oracle を使わない IDW 版と scale 不変の指標は副次として報告します。
10. model seed は 7301 に固定します。study_060 の FORGE では model seed による差は約 0.03 dB でした（ReGSI、CCNet）。したがって SD は partition とデータ順の変動を表します。
11. **調整量の表と、過去の勝ち設定の扱いの表**
    - 調整量の表には過去の target-informed な調整も含めます（C3：ReGSI は study_044 の28 run、CCNet は study_043 の6 run、NeRSI は study_040/046 の76+28 run。C3 の窓自体も study_041 で target を見て選ばれています）。
    - studies 057–066 で各手法が勝った設定を一覧にし、「第2弾で試した／ユーザー規則で不可／証拠により不採用」のどれかを付けます。
12. **pilot の記録**：規則を登録した時点で既に見ていた結果を開示します。対象は POCS の P1b grid、K1 の screening、C3 DRR の S1（study_063）、D7 の C3 と D5b、ReGSI の P2 a01_b01、NeRSI の P3 です。
13. **負の結果の段落**：(2) の不採用項目を載せます。
14. **床の基準**（2026-10-22）：S1 validation で best-of-25 とします。選ばれた cell の3 seed test も併せて報告します。
15. **選択規則**
    - validation の relMSE が最小の候補を選び、同値なら candidate ID 順にします。
    - diverged の run は最下位として数に含めます。
    - OOM の後は同じ yaml を再開します。
    - early stopping はしません。途中の結果を見て run を止めたり、優先順位を変えたりしません（メモリによる並べ替えだけは可）。
16. **S1 barrier**：確認するのは完了と床だけで、ReGSI が最良かどうかは確認しません。S1 を見た後は固定値も grid も変えません。seed は自動で進めます。ユーザーが止めて変更した場合は「post-S1」と明記します。
17. 最初の test 採点の後は、何も変えません。未達の目標は未達として報告します。
18. grid.yaml、各 config、各手法の yaml、評価 script と config の SHA256、さらに code の commit / tag を記録します。dirty worktree のままでは開始しません。

## (6) 報告・統計

**主指標**
- relMSE で選んだ cell の test mean trace SNR です。C3 は全 test trace、FORGE は clean_target で計算します。
- seed ごとに、その seed 自身の test 集合で計算します。
- 3 seed の等重み平均 ± 標本SD と、seed ごとの3つの値を示します。

**「dataset D で ReGSI が最良」と言う条件**：4つのベースラインすべてについて、次の両方を満たすこと（intersection-union test なので多重性の補正は不要です）。
- seed 平均のペア差 Δ の 95% CI（source-cell 単位の cluster bootstrap、B = 10,000）の下限が 0 より大きい。
- 3 seed すべてで Δ_s > 0。

**主張の書き方**
- 8 つの対比（各 dataset 4 つ）には、Holm 補正済みの両側 p 値を付けます。
- relMSE の差の向きも併記します。SNR と relMSE で向きが食い違う場合は、主張を取り消すのではなく「SNR では最良、relMSE では区別不能／逆」と限定して書きます。
- CI が 0 を含む場合は「区別不能」と書きます。

**感度**
- seed 単位のペア t 区間（df = 2）。
- S1 の validation に入っていない test trace だけでの再計算。
- EMA shadow の表（validation と test）。
- S-a（mask）と S-b（精度）。
- dataset ごとに最適化した古典手法の値。既存の grid から計算する報告専用の上限で、CPU だけで済みます。
- Q3(b) を実施する場合は C3 の別窓。

**副次指標**（向きにかかわらず報告）
- relMSE、global SNR と NMSE、相関の中央値。
- trace ごとの SNR の p10 / p50 / p90 と、0 dB 未満の trace の割合。
- FORGE の all_eligible、IDW 版、scale 不変の relMSE。
- seed ごとの validation と test の表。

**図表**
- 手法 × dataset ごとの 5×5 heatmap（relMSE と SNR）。relMSE の勝者と SNR 最良の cell に印を付け、両者の順位の Spearman 相関を示します。
- **全手法の grid 端フラグ**（C3 の ReGSI の k6、CCNet の a05、POCS の角）。grid は拡張しません。
- expected-best-of-k と winner's curse。S1 validation で採点した**全構成（pilot を含む）**で計算します。
- コスト表、protocol 表、調整量の表、過去の勝ち設定の表、負の結果。

**本文に書く注意**
- FORGE の 0 dB はゼロ予測と同じ水準です。
- seed 間で test 集合が約 80% 重なります。
- 初期化の seed は固定です。

## (7) スケジュール

**全体計画（承認用）**

| 区分 | run 数 | GPU-h（外部負荷なし） | 備考 |
|---|---|---|---|
| 第2弾（判断用） | R-1 1（+最大2）、C-0 の短い run 約8、C-1 4、N-1 4（+1） | 45–60 | 約 20–25 GPU-h は S1 に import |
| 第2弾（感度） | 2（+条件付きで C3 seed 3） | 25–40 | gate にしない |
| C3 の S1 と seed | 論理 140 | 115–215 | NeRSI は jitter を外せば 0、相乗りしなければ最大 75 |
| FORGE の S1 と seed | 論理 140 | 515–700 | うち CCNet が S1 290–390、seed 40–85 |
| 合計 | 論理 280 + pilot | **約 700–1,000** | CPU は約 6,000–7,000 CPU-h（主に DRR） |

**日程**

| 日付 | 作業 |
|---|---|
| 10-22 夜 | Q1–Q4 に回答をもらい、規則を登録します。R-1 を GPU0 で起動します。D7 が終わり次第 D-1 を判定します。sprint 1 を開始します。 |
| 10-23 | R-1 の結果が出ます。合格すれば C3 ReGSI の S1 を lane B で始めます（短い run なので再開機能なしでも許容）。 |
| 10-24 | sprint 1 を merge します。C-0（68 GiB 以上空いた GPU）、N-1（4 run 相乗り）、P-1 と D-2（CPU）を実行します。sprint 2 を開始します。 |
| 10-25〜26 | C-1 の4 arm を実行します（C3 は lane B、FORGE は lane A）。N-1 と N-2 を判定します。POCS の S1 が確定します。DRR の S1（CPU、上限 64 threads）を開始します。 |
| 10-27 | C-1 を判定し、sprint 2 を merge します。凍結と hash、`prepare --check-only` を行い、ユーザーに commit / tag を依頼します。FORGE の S1 を起動します。 |
| 約 11-01〜03 | C3 の S1、seed、test 採点が完了します。 |
| 約 11-06〜09 | lane B（C3、FORGE ReGSI、FORGE NeRSI）が空き、shard で FORGE CCNet を分担します。 |
| 約 11-09〜11 | FORGE の S1 が完了し、barrier になります。 |
| 外部負荷なしで 11-12〜15、現状程度の外部負荷で 11-18〜24 | FORGE CCNet の seed（a05 は 3 run で2巡、2.2–5 日）を終え、最終報告を出します。 |

**GPU の割り当て**
- **lane A**（68 GiB 以上空いている GPU。現在は GPU1 で、外部が約 13.5 GB 使用）：C-0 → C-1 の FORGE arm → FORGE CCNet の S1 を a05 → a04 → a03 → a02 → a01 の順に。空きが足りないときは、収まる行から先に始めます。
- **lane B**：R-1 → C3 ReGSI（k6 の列から）→ C-1 の C3 arm → C3 CCNet（a05 の行から）→ C3 NeRSI（相乗り）→ FORGE ReGSI（49 GiB 以上空いていれば k6 から）→ FORGE NeRSI（λ 0.01/0.03 の行から）→ seed → FORGE CCNet の分担。
- 相乗りはメモリを詰めるためだけに使います。throughput の改善は最大でも約 1.1 倍です。
- a05 の seed は2つを同じ GPU に載せられません。

**監視**
- 毎日、完了予定日の見積りを更新します。FORGE の barrier が計画より3日以上遅れそうならユーザーに報告します。
- C-0 の cuDNN benchmark で 1.3–2 倍速くなれば、CCNet の時間は 20–45% 短くなり、FORGE の完了は 2–5 日早まります。

## (8) ユーザーに確認したい事項

**Q1. FORGE CCNet の学習手順と、全体の計算量・日程の承認**
- **(A) 推奨**：C3 と同じ手順にします。毎更新で新しい patch を引き、現在の重みで forward と backward を1回ずつ行い、AdamW と EMA を1回更新します。FORGE だけの違いは bf16 だけになります。
  - 理由：手順が両 dataset で同じになります。cached 方式では、1 forward で使う group の数が patch の大きさとともに増えます（mask 0.25 で a01 は約1、a05 は約12）。そのため、勾配の古さが grid の軸と交絡します。
  - 参考：a05 の cached run は、1,800 更新の間ゼロ予測の loss（0.4609）に張り付いたままで、ユーザーの指示で止まりました。崩壊したという証明にはなりませんが、懸念材料です。
  - コスト：a05 は約 1.9–2.0 秒/更新で、1 run 26–28 h です。FORGE CCNet の S1 は 290–390 GPU-h、study 全体では約 700–1,000 GPU-h です。
  - 付随事項（いずれも推奨は「はい」）
    - (i) C-0 で 1.3 倍以上速ければ、CCNet だけ cuDNN benchmark（非決定的）を使います。精度は変わりませんが、bitwise の再現性は失われます。
    - (ii) S1 を終えた手法から seed の学習を先に始め、test の採点だけは barrier の後に行います。1–2 日短くなります。
    - (iii) FORGE CCNet の S1 の見積りが 450 GPU-h を超えたら、開始前に報告します。
- **(B)**：cached 勾配を、1 forward あたり最大2 group に制限して使います。実装が必要で、FORGE だけの手順差として宣言します。約 20–25% 安くなります。
- **(C) 非推奨**：制限なしの cached 勾配。

**Q2. FORGE NeRSI（P3 で −0.69 dB）について、次の4つを N-1 で一度に判定してよいか（推奨：すべて承認）**
- **(a)** 時間方向だけの×2段を加え、latent の時間長を C3 の 48 以下にします。
  - FORGE は s=3 で latent 34×4、params 341,485 になります。C3 は変わりません。
  - これは latent の大きさを揃える規則で、10-20 の「パラメータ数を合わせるためだけの構造変更はしない」という判断を覆します。
  - 物理的には study_061 の deep-time と同じ構造です（latent 1 sample = 256 ms）。C3 の 64 ms に揃えるなら s=1 ですが、s=1 には証拠がありません。
  - +0.2 dB 以上で採用します。
- **(b)** loss を各 trace の記録範囲に限ります。C3 では全 sample が有効なので何も変わりません。+0.2 dB 以上で採用します。
- **(c)** jitter なしが jitter ありと比べて −0.05 dB 以内なら、jitter を両 dataset で外します。本日の決定を取り消すことになりますが、C3 は 12.66 から 13.51 dB に戻り、study_064 の run を import できます。
- **(d)** それでも 0.55 dB 未満なら、NeRSI だけ EMA を 0.9998 にします。study_061 では FORGE 2 ms で relMSE が 0.9696 から 0.9251 に下がりました。C3 の1 run で確認します。満たさなければ床割れとして報告します。

**Q3. 評価 seed と C3 の外部検証**
- **(a) 推奨**：評価 seed を、どの namespace でも未使用の 7321–7323 に変えます。7311–7313 の test は、C3 でも FORGE でも既に採点済みです。今回は固定値が変わるため seed の run は元々再利用できず、追加コストはありません。model seed 7301 は維持します。
- **(b) 推奨（本結果が確定した後に実施）**：C3 の別の窓で外部妥当性を確認します。
  - 窓は、どの手法の出力も見る前に、幾何の規則で決めます。
  - S1 で選んだ設定をそのまま使い、3 seed で実行して副次として報告します。費用は約 40–60 GPU-h と CPU です。
  - 理由：現在の C3 の窓は target を見て選ばれ、その target で各手法が調整されています。

**Q4. 「ReGSI を超えない範囲で」の運用と、CCNet の time kernel**
- **(A) 推奨**：ベースラインの改良は、ReGSI を参照しない T1/T2 の規則だけで採否を決めます。
  - 採用した結果ベースラインが ReGSI を上回っても、捨てずにそのまま報告します。S1 barrier でも、ReGSI が最良であることを条件にしません。
  - C-1（time kernel 3→9）は T1 で試します。
  - 理由：「超えない範囲」を採否の条件にすると、査読で「上限を付けたベースライン」と読まれ、ReGSI が最良という主張そのものが弱くなります。
  - 参考の実測値：C3 は ReGSI 20.01（旧 k6）対 CCNet 19.07。FORGE 2 ms の study_060 の test は ReGSI 2.81 対 CCNet 2.16。
- **(B)**：(A) と同じ規則で、C-1 だけは行いません。tk9 の証拠は FORGE 2 ms だけで、dilation 軸が覆う範囲も 3–13 sample から 9–49 sample に変わるためです。tk9 は S1 の後の感度として報告するにとどめます。
- **(C) 非推奨**：ReGSI のペアの値を超えない場合だけ改良を採用する、という上限規則。使う場合は論文での開示が必要です。

---

## Changes from critique

**Adopted**
- **S1 barrier and the ReGSI-conditioning risk (fairness #1, overtuning #8).**
  - The barrier now checks only completeness and the floors. The "ReGSI best" clause is removed.
  - No fixed value or grid of any method changes after S1 is viewed. Seeds proceed automatically; a user halt-and-change is labelled post-S1.
  - The projection table and the "none can plausibly overtake ReGSI" sentence are removed. They are replaced by measured values and a lever-transfer table.
- **Round-2 cuts.**
  - R-3 cut (all three critiques).
  - R-2 relabelled as the first FORGE ReGSI S1 run.
  - The cached-gradient evidence run C-0(c) cut.
  - C-2 on C3 replaced by the FORGE FP32 vs bf16 sensitivity S-b, with a pre-declared remedy (fairness #5, overtuning #7).
- **POCS (overtuning #3, fairness #7).** P-1 is reduced to K1 only, judged by T2, demoted to should, and labelled screening-derived. K2 is dropped, which removes the hidden best-of-75 and the C3-dominated summed-dB rule.
- **DRR (feasibility #5, overtuning).**
  - D-1 is labelled screening-derived.
  - The production backend must reproduce D7 and D5b.
  - Frequency-bin chunking is now a must.
  - D-2 times the most expensive cell, rank 30 × [8,16,8,16], with workers=1 under an RSS cap.
  - The clip-and-shift boundary rule is declared.
- **Q1 rationale (overtuning #4).** The a05 cached run is now described accurately: it stalled at the zero-predictor loss for 1,800 updates and was stopped by the user, so it is not proof of collapse. The rationale is now protocol identity plus the staleness confound with the patch axis. cuDNN benchmark is split out as a Q1 sub-item.
- **Smoke gate (all three critiques).**
  - It is now normalised to the exact zero-predictor loss L0 and judged over updates 901–1000.
  - An output-RMS check is added.
  - It applies to every neural method's first FORGE run.
  - A failure triggers a bug search only, never tuning.
- **C-0 scope (overtuning #5, feasibility #4).** C-0 now includes an end-to-end prediction, scoring and manifest check at a01 and a05. The timing gate is replaced by a cost projection that shows contention, against a GPU-h ceiling agreed with the user in Q1.
- **Codex work split into three sprints (feasibility #1–3, #9).**
  - Exact resume, retry, watchdog, prediction-memory warm-up, GPU sharding and memory-aware candidate choice are must items before the long FORGE runs.
  - GPU-side patch assembly is replaced by a CPU prefetch thread.
  - The contrast script moves after launch.
  - Pilots run through the production runner so they can be imported.
- **NeRSI (overtuning #2, fairness #6).**
  - The sequential N-1 → N-2 chain is replaced by one parallel 2×2 batch at s=3 (jitter × support-masked loss), with shadow EMAs. The a03_b01 cell is dropped.
  - Q2 now says plainly that the latent rule is a count rule that reverses the 2026-10-20 decision, and that s=3 equals study_061's deep-time geometry.
- **Other additions.**
  - Shadow EMAs in all new neural runs, plus an EMA sensitivity table (fairness #3, partially).
  - Mandatory CCNet mask-sensitivity arms with a co-primary trigger (fairness #2).
  - CCNet RNG stream fix (feasibility #10, confirmed in code).
  - Seeds may train before the barrier, with test scored only after it (feasibility, in Q1).
  - The schedule is rebased on the loaded scenario, with daily projections and a 3-day slip trigger.
  - The external C3 window is offered as an optional secondary in Q3 (fairness #4).
  - Provenance fixes (1001 → 2172, 1008 → 2176), a commit/tag before launch, and a disk-space check.
- **Statistics.**
  - A relMSE/SNR disagreement now qualifies the "best" claim instead of vetoing it.
  - Added: test restricted to traces outside S1 validation, expected-best-of-k and winner's curse over all scored configurations including pilots, grid-edge flags for every method, the alignment numbers, the 0.03 dB model-seed evidence, a single parameter tolerance, the extra negative results, and the per-dataset-tuned classical rows.

**Rejected**
- **Common EMA switch to 0.9995 (fairness #3).**
  - EMA 0.999 is fixed by the user.
  - The available evidence shows no material baseline gain at 0.9995 (NeRSI in study_061: relMSE 0.9696 → 0.9631).
  - The shadow-EMA sensitivity answers the reviewer objection at no training cost.
  - NeRSI-only 0.9998 stays as a floor fallback, subject to Q2.
- **Cutting C-1 (overtuning #1) and making C-1 a must (fairness).** Neither. C-1 stays a should, subject to Q4.
  - It is off the critical path, because sprint 2 must land before FORGE CCNet S1 anyway.
  - It is CCNet's best-evidenced lever.
  - But its evidence is FORGE 2 ms only, and it changes what the dilation axis covers.
- **CCNet option D (reuse_patch with a fresh forward per group).** It would be a new procedure on both datasets with no evidence. Option A reproduces study_064 except for the mask. The mechanism difference is declared in the protocol table instead.
- **Clipping all methods, and running ReGSI's FORGE seeds in bf16.** The bf16 option would weaken the proposed method without helping fairness. FP32 CCNet seeds are the direct remedy if S-b shows a gap.
- **A generic floor-rescue ladder.** Rescue happens only in round 2. Any floor miss found at S1 is reported, ReGSI included.
- **s=1 for NeRSI.** There is no evidence for it.
- **Varying the model seed.** It stays fixed per the user rule, with the 0.03 dB evidence cited.
- **The external C3 window as a must.** It is optional (Q3) because of its cost and the user's preference for lean designs.
