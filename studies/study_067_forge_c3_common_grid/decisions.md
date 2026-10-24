# 現行条件を選ぶ理由

## 本番 G0（2026-10-24）：FORGE の最初のニューラル run

G0 は manifest に記録するだけの診断で、run を止めず、選択・解析にも入らない。

| run | 更新 901–1000 の loss / L0 | 出力 RMS / target RMS | 有限 | G0 | validation SNR |
|---|---:|---:|---|---|---:|
| CCNet `a01_b02` | 0.9191 | 0.2838 | true | passed | 1.582 dB |
| NeRSI `a03_b01` | 0.9985 | 0.0360 | true | failed | 0.269 dB |

NeRSI の不合格は実装バグではなく、座標ネットの遅い立ち上がりと判断した。登録規則どおりハイパーパラメータは変えず、そのまま実行してユーザーに報告する。
- 採用構成を決めた N-1 の FORGE run（本番へ import 済みの `a02_b04`、10-22 の同じ code path）も、更新 1000 で 0.996·L0 に留まっていた。その後は 2000 で 0.975、5000 で 0.779、50,000 で 0.500·L0 まで下がった。
- 本番の `a03_b01` も同じ経過をたどった（1000 で 1.000、5000 で 0.886、50,000 で 0.625·L0）。validation SNR は N-1 の −0.038 dB を上回る。
- G0 の閾値は CCNet の 1000 更新で較正したもので、NeRSI の 1000 更新では「張り付き」と「遅い立ち上がり」を区別できない。バグ探索は、この参照 run との比較で終えた。NeRSI には bf16 の経路がなく、学習後の loss と予測が正常なため、それ以上の探索は行わない。
- 以後の FORGE NeRSI の run も同じ理由で G0 不合格を記録しうる。判定には最終 loss と validation 指標を使う。ReGSI の最初の FORGE run（`a01_b05`）の G0 は、実行後に同様に確認する。

## Sprint 3（2026-10-24）：登録済み解析の実装と盲検の維持

評価設定は `evaluation.yaml` に独立させた。`grid.yaml` は launch tag `study-067-launch-v5` で凍結され、hash が run identity に含まれるため、設計書の「grid.yaml に evaluation 節を追加」は適用しない。両 dataset の config と各手法 YAML、runner・training・scoring は変更していない。`prediction_energy` と FORGE の IDW scale `k_i` は、保存された prediction・物理 target・observed input から解析側で計算し、解析出力だけに保存する。稼働中実験の所有物である run parquet への埋め戻しは行わない。EMA shadow は元々 scalar 指標だけの保存であり、shadow prediction/weight のない run で派生 per-trace 列は復元できない。import 元に shadow 指標がなければ欠測と明示する。必要なら将来の別途実行で shadow prediction と cell ID の保存を追加するのが最小変更であり、今回の本番コードには加えない。

本番開始は 2026-10-24 11:08 UTC。POCS の S1 は起動から約1.5時間以内に完了し、seed test の採点はこの script の完成より先に始まった。解析規則自体は [事前登録設計](../../reports/study_067_design_round2.md) (5) 9・14・15 と (6) に登録済みである。script 固定前に誰も本番 test 値を閲覧していない。本実装の開発・検証には rehearsal v4、study_063/064、および synthetic fixture だけを用い、本番2 run の評価 seed 出力・test 値を含む summary は開いていない。seed 監査でも本番2 root を `os.walk` の子ディレクトリから除いてから走査する。

受入1：study_064 `20261012T235609226847Z_05c92e79ce80_traces` の ReGSI `a01_b05`（dilation 1、k=6）と CCNet `a05_b02`（patch `[8,16,8,16]`、dilation 2）を validation relMSE から再選択した。seed 7311–7313 の partition は study_063 `20261011T145500928121Z_05c92e79ce80_traces` と一致する。新しい seed 別の対比計算で **+0.9198 dB、95% CI [0.8347, 1.0068]** を再現した（表示は小数4桁、B=10,000、RNG=670703、512 source cells、共有 multinomial 重み）。指定の +0.920、約 [0.83, 1.00] に対する許容幅内であり、定義や seed・cell を調整していない。

受入3：`runs/` と `studies/` の全 YAML、および experiment/manifest/config JSON を、frozen source snapshot も含めて検索した。評価 seed の宣言と partition の実体を区別する。**「7321–7323 は一度も partition seed として使われていない」は文字どおりには成立しない。** 10/22–23 の study_067 pilot 準備14 run の seal に、これらの評価 seed の `input/seed_*/partition.parquet` が記録されていた。該当 run の evaluation ディレクトリは0、実行された S1 の seed は C3=620702、FORGE=600702 である。他 study の partition 使用は検出されていない。これらは未採点の評価用 partition の事前生成であり、以前の「既存使用なし」という表現はこの区別で限定する。監査の機械判定は strict な不合格を保持し、事前生成をなかったことにはしない。

受入2：rehearsal v4 の C3 / FORGE、全5手法を保存配列だけから解析し、選択表10行、対比表16行、seed 別指標279行、seed 集計630行、EMA 表242行、cost 表62行、探索 score 表20行、expected-best-of-k 表40行、winner's-curse 表20行を生成した。図は validation heatmap 20枚、expected-best 曲線10枚、winner's-curse 10枚、派生 per-trace parquet は77個。未実行の grid セルは空欄で、diverged セルは D とする。意味のない rehearsal の値を科学的な結論には用いない。出力は作業用一時領域に置き、commit しない。manifest の142 SHA256は解析終了時のコード・入力とすべて一致した。

受入4：共有重み bootstrap と seed ごとに異なる test 集合を使う逐次参照実装の一致、Holm と8対比の族、選択の tie-break / diverged 最下位、S1 validation 和集合の除外、df=2 の paired t 区間を synthetic fixture で確認した。必要な配列がない場合の欠測表示も確認した。変更箇所と既存 fair-grid 統計の focused tests は30件通過。全 pytest は 7,080 passed / 32 skipped / 1 failed（`evaluation.yaml` の SHA256 が下の hash 表に未登録）で、登録後に study_067 の test 68件が通過した。

rehearsal v4 には FORGE NeRSI seed 990672 の `nan_prediction` fault injection による divergence がある。事前登録にない欠測値の補完、2 seed への母数変更、有限 penalty は導入しない。該当する3 seed 平均・CI・t 区間を算出不能とし、欠けた対比がある族では Holm p も欠測のまま表示する。これは解析規則の変更ではなく、必要なデータが存在しない箇所の表示である。S1 の divergence は登録どおり最下位として候補数に含める。

共通 grid は study_064 の ReGSI / CCNet / NeRSI、study_063 の DRR、および C3 POCS forge-matched 探索の組合せとする。候補順序は configuration ID を安定させる。共通 grid の空間値は dataset 別に変更せず、CCNet の時間長だけを固定設定の dataset 差として扱う。固定値の例外は leaf key ごとに理由を記録し、探索軸とは別に検証する。

ニューラル 3 手法は C3 study_064 の MSE、AdamW、定数学習率、更新数、loss trace 数、EMA と初期化 seed を引き継ぐ。FORGE の BF16 CCNet は study_066 の実行条件を引き継ぐが、丸め誤差を持つため import の execution-only 例外にはしない。CCNet の cached patch gradients / reuse_patch も学習入力順序への影響を無視できないため、import では一致を要求する。ReGSI の pruning / fused aggregation / activation checkpointing は既存の exact-by-design memory 実装であり、数学的設定から除く。

NeRSI の `training.coordinate_jitter_cells` はユーザー決定により両 dataset に固定で追加する。C3 と FORGE は同じ profile fitter を通るため、訓練座標だけに適用され、予測座標は変わらない。jitter のない既存 C3 / FORGE 学習は同一計算ではなく、再利用できない。

pilot P3 の C3 validation mean trace SNR は jitter 0.5 で 12.6600 dB、jitter なし baseline は 13.5100 dB（提示された丸め値）。C3 ≥ 10.5 dB の事前規則を満たす。FORGE P3 は完了し、validation mean trace SNR −0.6900 dB（比較基準は −0.689 dB）、relMSE 1.2140 である。NeRSI の救済は第2弾 N-1 / N-2 でのみ判断する。証拠は `runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p3_c3_nersi_jitter05` と `20261022T035444744870Z_05c92e79ce80_p3_forge_nersi_jitter05` および C3 parent baseline。

POCS は第2弾 P-1 の T2 判定により、両 dataset で K1（32反復、receiver window `[4,8]` / overlap `[3,6]`）を採用する。threshold 1.0 → 0.01 と時間窓 × `[source line, source point]` の共通探索軸は保持する。以下の P1 / P1b は K1 採用前の判断である。各 grid の候補は validation relative MSE 最小、同値なら candidate ID で選び、以下の mean trace SNR はその選択候補の値とする。FORGE は clean_target の validation を使い、test は判断に使わない。

P1 の事前規則は、{8,16,32} のうち FORGE validation mean trace SNR ≥ 0.5 dB かつ C3 ≥ 10.5 dB を満たす最大反復数を選び、該当なしなら報告するものだった。receiver `[8,16]` では、FORGE の 8 / 16 / 32 反復は 0.4020 / 0.4400 / 0.2050 dB、C3 は 7.0500 / 12.1200 / 14.4700 dB であり、P1 単独では該当しない。P1b は 16 反復で receiver `[8,16]` を基準に、FORGE の改善 ≥ 0.1 dB かつ C3 ≥ 10.5 dB を満たす variant がある場合だけ receiver を変更する事前規則である。FORGE の `[8,16]` / `[4,8]` / `[2,8]` は 0.4400 / 0.8950 / 1.1540 dB、C3 は 12.1200 / 10.8800 / 9.9800 dB。`[4,8]` は FORGE +0.4550 dB かつ C3 10.8800 dB で採用条件を満たし、`[2,8]` は C3 条件を満たさない。当時は 16 反復と `[4,8]` を両 dataset に採用したが、反復数と receiver overlap は P-1 の K1 採用で superseded。証拠は `runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p1_*`、`runs/study_067_forge_c3_common_grid/20261022T054443970014Z_05c92e79ce80_p1b_*`、および各 pilot が参照する既存 16 / 32 反復 run の validation 記録である。上記数値は提示された丸め値であり、full-precision の一致を主張しない。

ReGSI は pilot P2 の **V3 を両 dataset に採用済み**とし、`training.inner_mask_fraction: 0.25` と `model.node_fourier_components: 0` を固定する。P2 は V1: mask 0.25 のみ、V2: Fourier 0 のみ、V3: 両方を比較した。事前規則は baseline 比で FORGE ΔSNR ≥ +0.2 dB かつ C3 ΔSNR ≥ −0.2 dB を満たす variant のうち FORGE 改善最大を採用し、それ以外は現行値を保持するもの。k=2、dilations `[1,2]` の validation mean trace SNR は次の通り（提示された丸め値であり、full-precision の一致を主張しない）。

| variant | FORGE SNR (dB) | FORGE ΔSNR (dB) | C3 SNR (dB) | C3 ΔSNR (dB) |
|---|---:|---:|---:|---:|
| baseline | 1.8990 | — | 18.8600 | — |
| V1: mask 0.25 | 2.2110 | +0.3120 | 19.1400 | +0.2800 |
| V2: Fourier 0 | 2.7100 | +0.8110 | 19.1400 | +0.2800 |
| V3: 両方 | 3.0070 | +1.1080 | 19.0700 | +0.2100 |

全 variant が両条件を満たし、FORGE 改善最大の V3 を採用する。parameter 数は 363,269 から 346,885（−4.5102%）となり、既存の matched-budget ±5% tolerance 内である。全 25 候補は同じ parameter 数を共有し、target と tolerance は変更しない。study_064 の previous best と旧固定値を保持する判断は、この pilot 決定で superseded とする。旧 `model.node_fourier_components: 16` を含む reference setting は現行 grid に存在しないため、study_067 の `reference_settings` でその包含を要求しない。study_064 自体の履歴は保持する。証拠は `runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p2_c3_regsi_v{1,2,3}` / `20261022T035444744870Z_05c92e79ce80_p2_forge_regsi_v{1,2,3}` と各 parent baseline。DRR の反復数と damping は保持し、overlap は以下の D-1 の採用結果に従う。

FORGE reference の物理振幅波形は既存 anti-alias decimation 後の値を byte 検証して使用する。trace RMS は trace の固定属性であるため seed ごとに再計算する必要はない。各 seed では新しい観測集合を選び、観測用 RMS と復元用 RMS をその集合へ再対応付けする。RMS parquet の source label は各 seed の役割へ付け直す。test waveform の model 入力・S1 採点への使用を許可しない。

GPU admission は明示 opt-in とする。既存 study の per-experiment lock を変えず、本 study は repo 全体の `runs/.gpu_admission/` に startup lock と job slot を置く。空き memory から、他の slot の「宣言量 − 現在の reserved」の正の部分を差し引いて判定する。allocator cap によって cache が解放されても同じ job が再成長できるため、到達済み peak だけでは予約を減らせない。開始後は現在の reserved と診断用 peak を定期更新し、宣言量の 90% 到達または bounded timeout で startup lock を解放する。timeout 後も再成長分の予約は run 終了まで維持する。複数 run の同一 GPU 共存を許す。死亡 / zombie / PID reuse は `/proc/<pid>/stat` の state と start time で判定し、slot を除去する。admission は process へ signal を送らない。両 supervisor は全 worker を poll し、未報告の終了は failed state と exit code / missing state の理由を保存して barrier の他 worker に失敗を伝える。最初の失敗で待機を打ち切り、残った worker を停止せず記録する。CPU 候補は独立 process とし、同時に submit する候補を parallel_candidates 以下に制限し、失敗後の未開始候補を増やさない。各候補が自分の thread 数を保持する。priority は prepare で ID を検証し、実行順序だけを変え、選択規則を変えない。

C3 の GPU 要求量は study_064 の候補別の値を保持する。FORGE ReGSI は k=6 の実測 40.2 GiB reserved と C3 の k ごとの比率、4 GiB の余裕に基づく既存値を保持する。FORGE CCNet は下記 C-0 の fresh 更新・予測の実測から 17 / 17 / 17 / 30 / 64 GiB を維持する。旧 cached single-group probe は現行予算の根拠に使わない。NeRSI は指定値を保持する。`admission_margin_gib` は minimum に加える明示余裕であり、prepare / admission は device total から `context_allowance_gib` を引いた容量と照合する。精度と予測 core は変更しない。

legacy study_066 ReGSI grid worker（host PID 3958956、child 381291）は global admission に参加せず、再成長分の予約が見えない。両 dataset の config の `excluded_devices_while_pids_alive` に PID と kernel start ticks を固定し、いずれかが生存し zombie でない間は GPU0 を C3 / FORGE の admission から除外する。start ticks を照合するため PID reuse で無関係な process を除外し続けない。既存 worker の設定変更や停止で解決せず、終了を待ち、自動的に除外を解除する。他の non-opt-in GPU lane も現在 allocation 以外の予約は見えない。

S1 は全候補の validation を確認するが、新しい正式評価の test は全手法 S1 barrier の後に限定する。既存 single-config run に test prediction があっても import するのは S1 の validation 行のみとし、その過去の test score を選択へ用いない。再利用済みの計算を独立な新規実験と数えず、論理 run 数と再計算の有無を provenance で区別する。

pilot の import 元には両 dataset の P1b rx4x8、P2 全 variant、P3 jitter05 を宣言する。P2 V3 の `a01_b01` は採用した固定値と一致するため両 dataset で import 対象とする。V1 / V2 と旧 baseline は固定値が一致せず再利用しない。P3 NeRSI の `a02_b04` は jitter 0.5 を含め現行 config と同一計算である。FORGE の保存 profile 時間長1008と現行2176は、canvas 2172から実行時に2176へ再導出されるため identity から除く。completed と prediction の要件も満たす場合だけ coverage に数える。pilot の summary のみで完了とは判断せず、candidate manifest / prediction / config と保存済み入力の同一性を検証し、現在の code で再採点する。

## 2026-10-22 承認された第2弾と Sprint 1

ユーザー承認は Q1=A、Q2=すべて、Q3=評価 seed 7321–7323、Q4=A。以下に設計書 (2)、(3)、(5) を原文のまま登録する。原文の【Q】と「回答待ち」はこの承認で解消しており、pilot の現在の採否は次節に記録する。過去の cached 勾配を使う判断は Q1=A により superseded。C3 / FORGE とも reuse_patch / cache_patch_gradients は false、FORGE の学習差は bf16 のみとする。

CCNet の RNG は、placement / mask / supervision が同一 seed の3 generator だった bug を修正し、SeedSequence(7301).spawn(3) へ分離する。mask や supervision の消費が placement を動かさない。旧 CCNet run はこの乱数手順と mask が一致しないため、主結果へ import しない。cuDNN benchmark と ema_eval_decays は identity から除外するが、precision と ema_decay は一致を要求する。NeRSI の新しい2つの option は S1 既定へ入れず、C3 で no-op となる場合だけ identity に影響しない。

ReGSI の50候補は、Sprint 1 前に取得した resolved config identity をすべて維持する。P2 V3 a01_b01 と R-1 a01_b05 の既存 import を保持する。新しい shadow 診断は、追加の prediction / weight ファイルを保存しない。import 元にない shadow 指標は欠測として開示する。

評価 seed 7321–7323 は、2026-10-22 の read-only 監査で、studies / configs / data の設定と保存済み runs の config / experiment / manifest / metadata（source snapshot 除外）に partition seed としての既存使用がなかった。新しい study_067 の評価 seed 宣言だけが一致する。model seed は7301を保持する。

C-0 前は現行メモリ予算を暫定下限としていた。max(現行値, ceil(学習・予測の最大 reserved GiB) + 2) を使う。旧 cached single-group probe から fresh 更新の速度・メモリを推定しない。FORGE NeRSI の profile 2176 と CCNet の patch / prediction 時間2172は、1001 samples + 共通整列遅延1171 samples の実行時長である。NeRSI だけ decoder の8倍数へ切り上げる。

### Q1=A の付随事項（2026-10-22 ユーザー承認）

以下の3項目も option A とともに承認済み。cuDNN benchmark は CCNet の両 dataset に適用し、精度設定は保持する。seed の先行学習は Sprint 2 の実装対象であり、test の採点は barrier 後だけとする。

- (i) C-0 で 1.3 倍以上速ければ、CCNet だけ cuDNN benchmark（非決定的）を使います。精度は変わりませんが、bitwise の再現性は失われます。
- (ii) S1 を終えた手法から seed の学習を先に始め、test の採点だけは barrier の後に行います。1–2 日短くなります。
- (iii) FORGE CCNet の S1 の見積りが 450 GPU-h を超えたら、開始前に報告します。

### Sprint 1 review 修正（2026-10-22）

NeRSI の `training.coordinate_jitter_cells: null` は absent と同じ無効値として import identity を正規化する。no-jitter N-1 template は key を省略する。canvas から毎回再導出する profile 時間長も identity から除くが、空間長、jitter の有効値、alignment、upsampling、support mask、precision と primary EMA は従来どおり区別する。実行時 profile 時間長は FORGE 2176。

R-1 C3 a01_b05 と両 dataset の P-1 K1 を `import_from` に登録する。未完了は再利用せず、R-1 は現行 V3、K1 は採用した固定値と一致する場合だけ再利用する。

C-0 の cuDNN 速度比較は trial ごとに10回の untimed warm-up 更新を別に行い、続く N 更新の同期した所要時間の中央値を使う。順序は off,on,on,off とし、off→on と on→off の両比較、および全 timed update の中央値の速度比を記録する。1.3 倍の採否はこの定常部分の速度比に基づき、初期化と autotune を含む総時間では判定しない。

DRR は実効 overlap 0 のとき `drr_nonoverlapping` / `unknown_padding` を維持し、overlap > 0 のときだけ `drr_batched` / `clip_and_shift` と正の interior-Hann 合成を使う。resolved config に backend と boundary_rule を明示し、旧 config の省略値は実効 overlap から同じ規則へ正規化する。以下の承認設計の無条件な batched 切替の記述は、この zero-overlap の例外で訂正する。study_063 C3 DRR の再利用は overlap 0 の場合に限る。

D-2 の「CPU で数十分」は楽観的であり、計画値として使わない。FORGE [8,16,8,16] の60反復は候補あたり数十時間、worker 1 の1–2反復の D-2 自体も数時間以上になり得る。D-2 で時間と RSS を実測し、window ごとの worker 数、同時候補数、全60反復と S1 / seed の見積りを決める。72 h / cell または GPU critical path 超過は開始前に報告する。

## 第2弾の確定結果（2026-10-23）

R-1：C3 ReGSI V3 `a01_b05` は validation mean trace SNR **19.9140 dB**、relMSE **0.012633**。事前閾値 19.8120 dB 以上を満たすため **V3 を維持**する。証拠は `runs/study_067_forge_c3_common_grid/20261022T162308560491Z_05c92e79ce80_r1_c3_regsi_v3_a01_b05`。

P-1：**K1 を両 dataset に採用**する。選択候補の validation mean trace SNR は C3 13.7920 dB（現行 10.8770、Δ +2.9150）、FORGE 1.1650 dB（現行 0.8950、Δ +0.2700）。T2 の両 dataset Δ ≥ −0.05 dB、少なくとも一方 Δ ≥ +0.2 dB を満たす。**screening 由来（事後的、開示）**：K1 と規則は約15案の screening を見た後に決めた。証拠は `runs/study_067_forge_c3_common_grid/20261022T162308560491Z_05c92e79ce80_p1_c3_pocs_k1_iterations32_rx4x8_ov3x6` と同 prefix の `p1_forge_pocs_k1_iterations32_rx4x8_ov3x6`。両 dataset の `pocs_grid.yaml` を K1 と一致させ、CPU `prepare --check-only` で、両 dataset とも K1 の25/25候補を import 可能と確認した。全手法の coverage は [README](README.md) を参照。

C-0：FORGE CCNet（mask 0.25、fresh 更新、bf16）の dilation 6 の測定結果。予算は `max(現行値, ceil(学習・予測の最大 reserved GiB) + 2)` により維持する。

| 行 | s/更新 | h/50k 更新 | peak train / infer (GiB) | 予算 (GiB) |
|---|---:|---:|---:|---:|
| a01 | 0.2300 | 3.2000 | 5.6000 / 14.5000 | 17 |
| a02 | 0.4000 | 5.5000 | 6.3000 / 14.5000 | 17 |
| a03 | 0.8000 | 11.1000 | 15.0000 / 14.5000 | 17 |
| a04 | 1.3300 | 18.4000 | 24.3000 / 19.6000 | 30 |
| a05 | 3.1000 | 43.1000 | 57.5000 / 19.6000 | 64 |

cuDNN benchmark の a05 は、交互順序で off 3.8800 / 4.1300 s、on 3.0800 / 2.9200 s、集約速度比 **1.34 倍 ≥ 1.3**。Q1(i) により CCNet の **両 dataset で `execution.cudnn_benchmark: true`** を採用する。execution-only key とし、精度設定は保持する。bitwise 再現性は保証しない。tk9 a05 は 4.1500 s/更新、peak train / infer 27.4000 / 12.8000 GiB。FORGE CCNet S1 は約406 GPU-h で、事前報告閾値450 GPU-h未満。数値はユーザー提示の丸め値。

probe 証拠（すべて `runs/study_067_forge_c3_common_grid/` 配下）：

- `20261022T180757010559Z_05c92e79ce80_ccnet_memory_probe`
- `20261022T183903728400Z_05c92e79ce80_ccnet_memory_probe`
- `20261022T190456696007Z_05c92e79ce80_ccnet_memory_probe`

G0（C-0 step b）：FORGE CCNet の `a01_b02` / `a05_b02` をそれぞれ fresh 1000 更新した [end-to-end probe](../../runs/study_067_forge_c3_common_grid/20261022T192704131837Z_05c92e79ce80_ccnet_memory_probe/memory_probe.json) で **両方 PASS**。更新 901–1000 の平均 loss / L0 ≤ 0.95、同区間の出力 RMS / target RMS ≥ 0.2、全値有限の登録済み G0 を満たす。L0 は同じ擬似 target に対する厳密なゼロ予測 loss。上の timing probe と別の end-to-end run を証拠へ追加する。

| cell | L0 | 平均 loss（901–1000） | 平均 loss / L0 | 出力 RMS / target RMS | 有限 / 判定 |
|---|---:|---:|---:|---:|---|
| `a01_b02` | 0.46086556168940895 | 0.423367923 | 0.918636 | 0.284084 | true / PASS |
| `a05_b02` | 0.46086556169270365 | 0.388296801 | 0.842538 | 0.394574 | true / PASS |

N-1：FORGE `a02_b04` の time-only s=3、jitter × loss 範囲の全 arm が完了。以下は各 manifest の primary EMA 0.999、S1 validation `clean_target`。**s=3 を採用、support-mask loss は不採用、jitter は保持**と判断する。

| jitter | loss 範囲 | mean trace SNR (dB) | relMSE | artifact |
|---|---|---:|---:|---|
| あり | canvas 全体 | -0.036867405 | 1.045480641 | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183518698062Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) |
| あり | 記録範囲のみ | -0.101841884 | 1.062622223 | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183524117130Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) |
| なし | canvas 全体 | -0.338598665 | 1.126510918 | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183529511220Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) |
| なし | 記録範囲のみ | -0.499721532 | 1.166332817 | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183534795056Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) |

判定規則は [design の N-1 行](../../reports/study_067_design_round2.md) の原文どおり（閾値は設計 artifact、実測値と差分は [集計 artifact](round2_outcomes.json)（原 artifact の path / SHA256、差分、import identity / input fingerprint、CPU 推定式、prepare 出力を収録））：

> (i) [jitter, 全体] − P3（−0.689）≥ +0.2 なら s=3 を採用（C3 は構造上変わらない）。不採用なら NeRSI の調整を止めて報告します。(ii) [jitter, 記録範囲] − [jitter, 全体] ≥ +0.2 なら support mask を採用（C3 では bitwise に不変であることを test で確認）。(iii) 採用した loss の下で、jitter なし ≥ jitter あり − 0.05 なら、jitter を両 dataset で外します（Q2 の同意が必要。C3 の S1 は study_064 から import できます）。(iv) 採用構成の EMA 0.999 での値が 0.55 dB 未満なら N-2 に進みます。

(i) [jitter, 全体] は -0.036867405 dB、[P3 manifest](../../runs/study_067_forge_c3_common_grid/20261022T035444744870Z_05c92e79ce80_p3_forge_nersi_jitter05/method_runs/nersi_profile/manifest.json) の -0.688776336 dB に対し Δ +0.651908931 dB。設計の丸め基準 −0.689 でも条件を満たし、FORGE の s=3 を採用する。C3 は整列がなく、`canvas_time_stages` が `ceil(T / 2^(3+s)) <= 48` となる最小 s を選ぶため、保存入力の T=384 では s=0。time-only 段を追加せず構造が変わらない（[実装](../../src/seis_interp/forge_mvp_config.py)、[C3 入力契約](../../runs/study_067_forge_c3_common_grid/20261022T183550980564Z_05c92e79ce80_traces/study/config.yaml)、`test_canvas_rule_keeps_c3_and_normalizes_forge`）。

(ii) jitter ありで support − canvas は -0.064974478 dB < +0.2 dB のため support mask は不採用。(iii) 採用した canvas loss で no-jitter − jitter は -0.301731260 dB < −0.05 dB のため、両 dataset の jitter を保持する。(iv) 採用 arm は 0.55 dB 未満のため N-2 へ進む。

N-2：採用 N-1 arm の [同じ manifest](../../runs/study_067_forge_c3_common_grid/20261022T183518698062Z_05c92e79ce80_forge/method_runs/nersi_profile/S1/a02_b04/manifest.json) の shadow EMA 0.9998 は -0.028062201 dB / relMSE 1.044421866、primary EMA 0.999 は -0.036867405 dB / relMSE 1.045480641。Δ +0.008805205 dB < +0.15 dB で必要条件を満たさず **不採用**。C3 の確認 run は不要。[登録規則](../../reports/study_067_design_round2.md) に従って FORGE NeRSI の床救済はここで終了し、FORGE の床 0.4 dB 未達が予想されることを報告し、追加調整しない。

ユーザーが N-1 規則 (i) と Q2 の救済策の本番適用を承認したため、両 dataset の `nersi_profile.yaml` に `canvas_time_upsampling: true` を適用した。jitter 0.5、全 canvas loss（`loss_support_mask` は absent）、primary EMA 0.999 を保持する。先の本番適用を保留する判断は superseded。C3 は `canvas_time_stages(384) == 0`、time-only 段を追加せず、入力時間長を使う import identity は flag を正規化して除く。C3 NeRSI は従来の P3 `a02_b04` の1件を維持する。FORGE は `canvas_time_stages(2172) == 3` で flag が identity に残り、採用 N-1 arm の `a02_b04` だけが現行 resolved config と input fingerprint に一致する。upsampling のない FORGE P3 は input が一致しても identity が異なり、`import_from` に残した状態で拒否される。C-1 完了後の CPU prepare coverage は C3 / FORGE の順に ReGSI 2 / 1、CCNet 1 / 1、NeRSI 1 / 1、POCS 25 / 25、DRR 1 / 1（各25候補）。全手法の評価 seed import は0 / 3。source path、identity、input fingerprint は [適用後の artifact](round2_outcomes.json) に記録する。

採用設定の NeRSI parameter 数は C3 363,269、FORGE 341,485（全25候補で一定）。FORGE は C3 target 363,269 比 −5.9967% で ±5% band（整数では345,106–381,432）を下回る。FORGE の契約は同じ architecture 幅と宣言済み canvas 適応を許す policy であり、この例外を報告して幅や tolerance は変えない。N-1 全4 arm の recorded peak CUDA reserved は全て 1,046,478,848 bytes（0.9746 GiB）、最大 allocated は 914,378,752 bytes（0.8516 GiB）。宣言済み `nersi_profile: 4` GiB はこの実測 peak をカバーする。新たな GPU 測定や実験は行わない。

D-1：**window から決める Hann 重み付き overlap を両 dataset に採用**。**screening 由来（事後的、開示）**。CPU `scripts/launch_study067_pilots.py check-d1` は次の全再現で差 0.0 dB、許容 0.01 dB 内の PASS。[T2 の閾値](../../reports/study_067_design_round2.md) FORGE ≥ 0.656 dB、C3 ≥ 15.125 dB を D7 が満たす。

| 再現 | validation mean trace SNR (dB) | scratch 差 (dB) | 判定 | artifact |
|---|---:|---:|---|---|
| FORGE D7（再実行） | 1.097869355580972 | 0.0 | PASS | [manifest](../../runs/study_067_forge_c3_common_grid/20261023T003937042544Z_05c92e79ce80_forge/method_runs/drr_grid/S1/a02_b02/manifest.json) / [scratch reference](../../runs/study_067_forge_c3_common_grid/20261023T003937042544Z_05c92e79ce80_forge/study/d1_reference.json) |
| C3 D7 | 16.08733966197576 | 0.0 | PASS | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183550980564Z_05c92e79ce80_traces/method_runs/drr_grid/S1/a02_b02/manifest.json) / [scratch reference](../../runs/study_067_forge_c3_common_grid/20261022T183550980564Z_05c92e79ce80_traces/study/d1_reference.json) |
| FORGE D5b | 1.089401082766802 | 0.0 | PASS | [manifest](../../runs/study_067_forge_c3_common_grid/20261022T183556619725Z_05c92e79ce80_forge/method_runs/drr_grid/S1/a02_b02/manifest.json) / [scratch reference](../../runs/study_067_forge_c3_common_grid/20261022T183556619725Z_05c92e79ce80_forge/study/d1_reference.json) |

最初の FORGE D7 は [旧 watchdog の記録](../../runs/study_067_forge_c3_common_grid/20261022T183540163875Z_05c92e79ce80_forge/method_runs/drr_grid/S1/a02_b02/watchdog.json)（`progress stalled for three report intervals`）で停止された run であり、証拠には数えない。結果には上表の再実行を使う。

共通 [grid.yaml](grid.yaml) の window 軸に、line 軸0 / point 軸50% の明示 overlap を加える。rank × window の軸値・軸 ID・candidate ID は保持し、resolved config は `drr_batched` / `clip_and_shift` と明示 overlap を持つ。正の interior-Hann 合成を使い、実効 overlap 0 の旧 backend は維持する。対応は `[4,6,4,6]→[0,3,0,3]`、`[4,8,4,8]→[0,4,0,4]`、`[6,8,6,8]→[0,4,0,4]`、`[8,12,8,12]→[0,6,0,6]`、`[8,16,8,16]→[0,8,0,8]`（[設定と検証 artifact](round2_outcomes.json)）。

D7 の両 run は rank 12 × `[4,8,4,8]`、60反復、overlap `[0,4,0,4]` の `a02_b02` と resolved identity / 保存入力 fingerprint / 完了 prediction が一致したので `import_from` に登録した。D5b は30反復のため登録しない。study_063 は宣言を残して identity check を通し、旧 overlap 0 の DRR 全候補が拒否されることを確認した。差分と hash は [import 監査 artifact](round2_outcomes.json) を参照。

D-2：[a05_b05/result.json](../../runs/study_067_forge_c3_common_grid/20261022T183601578711Z_05c92e79ce80_forge/d2_measurements/a05_b05/result.json) の実測は rank 30 × `[8,16,8,16]`、**overlap `[0,4,0,4]`**、2反復、1 worker。提示された `[0,8,0,8]` とは異なるため、採用 overlap の測定結果とは扱わない。wall 31526.971176 s（8.757492 h）、9 blocks、平均 3502.487869 s/block、process peak RSS 5303988224 bytes（4.939724 GiB、入力ロードを含む lifetime peak）。60反復の線形外挿は 105074.636057 s/block（29.187399 h）、全候補の serial CPU 時間 262.687862 CPU-h。FFT/setup も反復比で拡大する保守的な外挿である。

FORGE `a05_b04` の [result.json](../../runs/study_067_forge_c3_common_grid/20261022T183601578711Z_05c92e79ce80_forge/d2_measurements/a05_b04/result.json) も **完了**。rank 30 × `[8,12,8,12]`、測定 overlap `[0,4,0,4]`、2反復、1 worker、16 blocks。wall 10971.3912 s（3.0476 h）、block 合計 10964.2346 s、平均 685.2647 s/block、non-block overhead 7.1567 s。process peak RSS は4075323392 bytes（3.7954 GiB、入力ロードを含む lifetime peak）。60反復の線形外挿は20557.9398 s/block（5.7105 h）、測定した16 blocks の serial CPU 時間は91.3706 CPU-h。validation / test 採点はともに0回。

従来の b04 計画値は、採用 overlap `[0,6,0,6]`、25 blocks、5.5867 h/block。測定由来の5.7105 h/block は **+2.2176%** で、採用25 blocks へ代理適用すると142.7655 CPU-h/candidate、12 workers の3巡で17.1336 h/candidate（従来139.6675 CPU-h / 16.7612 h）。16 blocks の測定を採用25 blocks の実測とは扱わない。FORGE 全体の見積りは下表へ更新し、増分は15.4896 CPU-h / 0.7448 h wall と小さいため並列計画は維持する。

依頼時の「C3 に result がない」という前提とは異なり、監査時点では C3 D-2 の両 result.json が **completed** で存在する。[b04](../../runs/study_067_forge_c3_common_grid/20261022T183612495378Z_05c92e79ce80_traces/d2_measurements/a05_b04/result.json) / [b05](../../runs/study_067_forge_c3_common_grid/20261022T183612495378Z_05c92e79ce80_traces/d2_measurements/a05_b05/result.json) を記録し、C3 自身の測定値を下記の外挿へ使う。ともに rank 30、2反復、1 worker、測定 overlap `[0,4,0,4]`、validation / test 採点0回。

| C3 cell | blocks | wall (s) | 平均 s/block | peak RSS (GiB) | 60反復 h/block |
|---|---:|---:|---:|---:|---:|
| `a05_b04` | 32 | 5788.9017 | 180.7896 | 1.8141 | 1.5066 |
| `a05_b05` | 18 | 17724.1515 | 984.5385 | 2.0048 | 8.2045 |

C3 b04 の採用 overlap は `[0,6,0,6]` / 50 blocks、b05 は `[0,8,0,8]` / 18 blocks。いずれも採用 overlap の実測ではなく、per-block 時間の代理値である。C3 計画の合計は1615.9171 → 1613.6910 CPU-h、96.2356 → 95.7743 h wall となる。

[block 数と推定式の artifact](round2_outcomes.json) により、採用 overlap の最大 window は FORGE 9 blocks、C3 18 blocks。FORGE は測定時と block 数・shape が同じなので時間の代理値として使う。FORGE は `workers: 12` / `parallel_candidates: 4`、C3 は `workers: 12` / `parallel_candidates: 2` とする。C3 の18 blocks は12 workersを使うので、worker 数を減らす理由はない。両 dataset で b05 → b04 → 残りの window、各 window は rank 30 から開始する `candidate_priority` を指定し、候補選択順序は保持する。

FORGE の重い候補は12 workersで約29.1887 h（1巡）、C3 の最大候補は約16.4097 h（2巡）。[設計の72 h規則](../../reports/study_067_design_round2.md) は採用 worker 数での候補 wall に適用し、worker 1 の artifact の `exceeds_72_hours: true`（serial 時間）と区別する。FORGE b04 の測定16 blocks の serial 外挿も91.3706 hで72 hを超えるが、採用25 blocks / 12 workers の候補 wall は17.1336 h。現計画では72 h未満かつ設計の FORGE GPU critical path 内に収まる見積りなので、開始前報告の発動条件には該当しない。

| dataset | workers × S1 同時候補 | S1 残り CPU-h | seed CPU-h | 合計 CPU-h | S1 + seed wall (h) |
|---|---:|---:|---:|---:|---:|
| FORGE | 12 × 4 | 2184.9 | 788.1 | 2973.0 | 63.5 + 58.4 ≈ 121.8 |
| C3 | 12 × 2 | 1170.6 | 443.0 | 1613.7 | 63.0 + 32.8 = 95.8 |

推定は [artifact の式と窓別表](round2_outcomes.json) に従う。各 dataset の b04 / b05 は対応する実測 per-block 時間と overhead を使い、b01–b03 は同じ dataset の b05 から SVD 行列の `M*N*min(M,N)` 比でスケールする（同じ dataset 内の FFT bin 数は一定）。SVD は rank 切詰め前に全分解するので全 rank に重い測定のコストを使う。S1 は D7 import を除外し priority 順に list scheduling、seed は未確定 winner の代わりに最大候補、seed 同時数は runner の上限を使う。採用 overlap への代理適用、他 window のスケーリング、worker 理想並列性、候補間競合と入力複製 RSS は未検証であり、D-2 の実測と混同しない。

CPU 計画は FORGE DRR 48 + C3 DRR 24 + POCS 両 dataset 2 + 2 + GPU 側（seed / ReGSI CPU threads / CCNet 先読み込み）の予約20 = **96 cores 以下**（[CPU 配分 artifact](round2_outcomes.json)）。起動時は OMP / OpenBLAS / MKL を各1 thread とし、GPU 側は予約内へ収まるよう method / seed の同時起動を調整する運用条件。現在の config に全体 CPU cap の強制機能はない。今回、起動は行わない。

`execution.workers` と `execution.batch_matrix_limit` は import identity から除外する。`test_overlapping_grid_execution_settings_preserve_bits_and_import_identity` は正の overlap を持つ小 volume で worker 数・周波数分割・block batching を変え、出力の bitwise 一致と両 key の identity 除外を確認する。overlap / boundary / rank / 反復数は引き続き identity に含む。CPU prepare coverage は [README](README.md) と [出力 artifact](round2_outcomes.json) を参照。

C-1：4実験すべて完了し、**T1 不成立、両 dataset で tk3 を維持**する。設定の正本は `pilots/c1_{c3,forge}_tk{3,9}/`。tk3 は hidden 53 / intermediate 63（363,292 params）、tk9 は hidden 36 / intermediate 37（363,227 params）で同じパラメータ予算。C3 は `a05_b02` / FP32、FORGE は `a03_b02` / bf16。mask 0.25、fresh patch / gradient、cuDNN benchmark on、主 EMA 0.999 と shadow 0.999 / 0.9995 / 0.9998 を共通にする。

**T1**（設計書から転記）：FORGE で ΔSNR ≥ +0.2 dB、かつ C3 で ΔSNR ≥ −0.2 dB なら tk9 を採用する。Δ はそれぞれ上記の同一 cell の tk9 − tk3、指標は S1 validation mean trace SNR（FORGE は clean_target）。relMSE も併記する。満たさなければ tk3 を維持し、両 dataset の CCNet S1 より前に確定する。

以下は完了 manifest の primary EMA 0.999。表示は丸め値で、full precision、全 manifest の SHA256、runtime 秒、shadow 指標、identity 監査は [c1 artifact](round2_outcomes.json) に保存した。各 manifest の `test_scoring_calls` は0。

| dataset / cell | tk3 SNR (dB) | tk9 SNR (dB) | ΔSNR (dB) | tk3 relMSE | tk9 relMSE | tk3 / tk9 runtime (h) | T1 条件 |
|---|---:|---:|---:|---:|---:|---:|---|
| C3 `a05_b02` | 18.8033 | 18.7228 | -0.0805 | 0.0167 | 0.0170 | 10.3941 / 16.5190 | ≥ −0.2、達成 |
| FORGE `a03_b02` clean_target | 1.5923 | 1.6616 | +0.0693 | 0.7376 | 0.7277 | 13.7093 / 15.9392 | < +0.2、未達 |

Δ の artifact 値は C3 −0.08052168555821737、FORGE +0.06929602288582104 dB で、ユーザー提示の6桁丸めと一致する。FORGE が必要条件を満たさないため tk9 は不採用。両 `ccnet5d_grid.yaml` は **time kernel 3 / hidden 53 / intermediate 63 のまま byte 不変**であり、SHA256 は下記登録表と一致する。

shadow EMA は診断専用で、T1 / production EMA の判断には使わない。各欄は validation SNR (dB) / relMSE、FORGE は clean_target。0.999 は主予測と一致する。

| arm | EMA 0.999 | EMA 0.9995 | EMA 0.9998 |
|---|---:|---:|---:|
| C3 tk3 | 18.8033 / 0.0167 | 18.8122 / 0.0166 | 18.8299 / 0.0166 |
| C3 tk9 | 18.7228 / 0.0170 | 18.7370 / 0.0170 | 18.7486 / 0.0170 |
| FORGE tk3 | 1.5923 / 0.7376 | 1.5968 / 0.7369 | 1.6048 / 0.7356 |
| FORGE tk9 | 1.6616 / 0.7277 | 1.6663 / 0.7270 | 1.6770 / 0.7252 |

pilot は local grid の25候補と元の candidate ID を保持する。[準備先と identity hash](pilots/c1_preparation.md) の両 arm を `import_from` に残し、CPU の公式 prepare を一時領域へ行って、現行 S1 の resolved config と保存入力 fingerprint に対して各 source 単独の `find_prediction_import` を検証した。**tk3 は C3 `a05_b02` / FORGE `a03_b02` として import 可能**。両 tk9 は入力 fingerprint が一致しても model の time kernel / hidden / intermediate が異なり、identity で拒否される。`import_from` は編集していない。run / prediction / manifest は読み取り専用であり、この記録作業では実験を起動しない。

C-1 の UUID 指定とメモリ事前確認は N-1 と同じ launcher 方式を使う（pilot の global admission は off）。各 arm の宣言は C3 32 GiB、FORGE 17 GiB、追加 margin 各2 GiB。各 dataset の2 arm を同じ GPU に起動するため必要な空きは C3 68 GiB、FORGE 38 GiB。同一 UUID なら合計106 GiBを確認する。launcher は physical UUID と live pinned legacy worker を照合し、実験別 `c1.log`、detached PID を出力する。既存 process へ signal は送らない。

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


## SHA256 登録（DRR overlap / N-1 upsampling 適用後）

| ファイル | SHA256 |
|---|---|
| [c3/ccnet5d_grid.yaml](c3/ccnet5d_grid.yaml) | `cb8713085c8ffdca30f3576612195c7d971761aac0db16f45443e75eb66e1f2f` |
| [c3/config.yaml](c3/config.yaml) | `7e74db1748510a67d86b29421f3a04680f473fab8baa7e94c23d1638367bfb81` |
| [c3/drr_grid.yaml](c3/drr_grid.yaml) | `6e465080712272a313adef7802698b0a61436ba7386a060b3231953da4d96336` |
| [c3/inputs.yaml](c3/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [c3/nersi_profile.yaml](c3/nersi_profile.yaml) | `eb7da44217858c180a7f44e7fcea0a288eeb8ceccdabd189a38505cf887a9dc9` |
| [c3/pocs_grid.yaml](c3/pocs_grid.yaml) | `e291a855d2c26872e9e84a06ee17486274f67e00d9c9fc6b98cb4784590375fc` |
| [c3/pocs_k1.yaml](c3/pocs_k1.yaml) | `e291a855d2c26872e9e84a06ee17486274f67e00d9c9fc6b98cb4784590375fc` |
| [c3/regsi_real.yaml](c3/regsi_real.yaml) | `0289dc66f4afd589d8a6f80c0ec62d6daaa6144767f448b1a9d8a7c68910f19e` |
| [dataset_differences.yaml](dataset_differences.yaml) | `b5760ca9e76cf8c2e5e0aae5b25c4dbf3bc83d7396997673db73b5cbadba5be5` |
| [evaluation.yaml](evaluation.yaml) | `3c76742dea34f85b0317d2778f02c46bb32fe5c387ee27d180bedd31833f4ed0` |
| [forge/ccnet5d_grid.yaml](forge/ccnet5d_grid.yaml) | `6d5bf508ed44f93a7ab15d4a36d418e40fb47a4d095714fdf8af74abd6ff9dc3` |
| [forge/config.yaml](forge/config.yaml) | `9e27bde0ac5a8acc4f5e30ca3f1e853d6d2b654b0439f9a307e3b2264b1a8611` |
| [forge/drr_grid.yaml](forge/drr_grid.yaml) | `6e465080712272a313adef7802698b0a61436ba7386a060b3231953da4d96336` |
| [forge/inputs.yaml](forge/inputs.yaml) | `257e389b8b3aff6d8cae1beeeb2c0169636398cd96aca9241eca08ffe6ebedaa` |
| [forge/nersi_n1_jitter_canvas.yaml](forge/nersi_n1_jitter_canvas.yaml) | `596cb540da80665e4f57792fe96709673a1ac0a7944431d661ed3645414b1ffc` |
| [forge/nersi_n1_jitter_support.yaml](forge/nersi_n1_jitter_support.yaml) | `c714c83d1b145c6ad657ace4b79f0d99ccff86c00334a69334b8771f9f3acf48` |
| [forge/nersi_n1_no_jitter_canvas.yaml](forge/nersi_n1_no_jitter_canvas.yaml) | `c9e6a6e73ea3382656e9f9c5e3cb0e09488076fcf035e46a846844f5177bcaac` |
| [forge/nersi_n1_no_jitter_support.yaml](forge/nersi_n1_no_jitter_support.yaml) | `400d968f387d893a2784542eacf219e712790fa4b1b165aa95c723c2977274f9` |
| [forge/nersi_profile.yaml](forge/nersi_profile.yaml) | `eb9fb9cbee0705df61272588fe53cb2499b7800ce0f40c6d5821abcecb751b0a` |
| [forge/pocs_grid.yaml](forge/pocs_grid.yaml) | `e291a855d2c26872e9e84a06ee17486274f67e00d9c9fc6b98cb4784590375fc` |
| [forge/pocs_k1.yaml](forge/pocs_k1.yaml) | `e291a855d2c26872e9e84a06ee17486274f67e00d9c9fc6b98cb4784590375fc` |
| [forge/regsi_real.yaml](forge/regsi_real.yaml) | `b2947880c8bc04ab40b8a84180442cce2ba6d521a404e5bda2429d5bb6f83dfe` |
| [grid.yaml](grid.yaml) | `1a48bb2532c52353b535e828286b769bc49115487b316d5e05374f4034690098` |
| [pilots/c1_c3_tk3/ccnet5d_grid.yaml](pilots/c1_c3_tk3/ccnet5d_grid.yaml) | `a766499c98bd0798063cb52787218b48868705dca89f98ecb5ce9f4cec2bc2bf` |
| [pilots/c1_c3_tk3/config.yaml](pilots/c1_c3_tk3/config.yaml) | `f5d9fac353c1a0ec8ecea457d9be45454aca06c4883a571f13b40fbfbbe30ac3` |
| [pilots/c1_c3_tk3/grid.yaml](pilots/c1_c3_tk3/grid.yaml) | `7cbd91f31fd2a1914cc8f1d28e67101b4233e3849f4156cd02e605a0c3f8335d` |
| [pilots/c1_c3_tk3/inputs.yaml](pilots/c1_c3_tk3/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [pilots/c1_c3_tk9/ccnet5d_grid.yaml](pilots/c1_c3_tk9/ccnet5d_grid.yaml) | `95e7951b39f5d508d139422079dd8510ef1785b9287689f945cd17139ccc8970` |
| [pilots/c1_c3_tk9/config.yaml](pilots/c1_c3_tk9/config.yaml) | `f5d9fac353c1a0ec8ecea457d9be45454aca06c4883a571f13b40fbfbbe30ac3` |
| [pilots/c1_c3_tk9/grid.yaml](pilots/c1_c3_tk9/grid.yaml) | `7cbd91f31fd2a1914cc8f1d28e67101b4233e3849f4156cd02e605a0c3f8335d` |
| [pilots/c1_c3_tk9/inputs.yaml](pilots/c1_c3_tk9/inputs.yaml) | `02db0cdeca2570fa8736495e2c8102e82076caf71de5413832475854d4f50ede` |
| [pilots/c1_forge_tk3/ccnet5d_grid.yaml](pilots/c1_forge_tk3/ccnet5d_grid.yaml) | `d92ced4a859920b54864c7f9ec5650447f86936bf96d73d8cdcea63aafed4e9e` |
| [pilots/c1_forge_tk3/config.yaml](pilots/c1_forge_tk3/config.yaml) | `1ee356748d00dcffdc25bdc01928389361e1957ee182676300f7f7a368499593` |
| [pilots/c1_forge_tk3/grid.yaml](pilots/c1_forge_tk3/grid.yaml) | `7cbd91f31fd2a1914cc8f1d28e67101b4233e3849f4156cd02e605a0c3f8335d` |
| [pilots/c1_forge_tk3/inputs.yaml](pilots/c1_forge_tk3/inputs.yaml) | `257e389b8b3aff6d8cae1beeeb2c0169636398cd96aca9241eca08ffe6ebedaa` |
| [pilots/c1_forge_tk9/ccnet5d_grid.yaml](pilots/c1_forge_tk9/ccnet5d_grid.yaml) | `73a5a8afbab65e0d800899021ae0aba4d0a578ffd72c4a1a46702a5875145828` |
| [pilots/c1_forge_tk9/config.yaml](pilots/c1_forge_tk9/config.yaml) | `1ee356748d00dcffdc25bdc01928389361e1957ee182676300f7f7a368499593` |
| [pilots/c1_forge_tk9/grid.yaml](pilots/c1_forge_tk9/grid.yaml) | `7cbd91f31fd2a1914cc8f1d28e67101b4233e3849f4156cd02e605a0c3f8335d` |
| [pilots/c1_forge_tk9/inputs.yaml](pilots/c1_forge_tk9/inputs.yaml) | `257e389b8b3aff6d8cae1beeeb2c0169636398cd96aca9241eca08ffe6ebedaa` |
| [pilots/c1_preparation.md](pilots/c1_preparation.md) | `97912538e266a4294334bcb13bbaf108a27ca0e9c0d318b46d5a7c46ad7655ab` |

評価 code・launcher と承認設計の SHA256 は更新時点の snapshot として次に記録する。最初の test 採点前に commit / tag を含めて再凍結する。

| ファイル | SHA256 |
|---|---|
| `reports/study_067_design_round2.md` | `5483483b6316b3d714816a813c9b96114d0cecfacb18caf442213556e5ba3651` |
| `scripts/run_c3_region_experiment.py` | `01054c5eb2eed0a1a424a7c5e3c09991c5cc5ec19d489b0a8877080f19bee994` |
| `scripts/run_forge_grid_experiment.py` | `67bd24f2afa12c4d766d11d29ce1f2de187f106892b726024cfec11fe98a7298` |
| `scripts/probe_forge_ccnet_memory.py` | `e542d02dc7505b8047299968b696667fc167d17ffc5b250800001f44da5054cf` |
| `scripts/launch_study067_pilots.py` | `9f343c006f7ffa5ef5be4abdca4ae44710afa5de2601e0059e4ce672a469ca4a` |
| `tests/unit/test_study067_pilot_launch.py` | `5e1cb8ce23a3d9e1a70348cf34fb012a4744115312b5c5bdb519f611509f43c4` |
| `src/seis_interp/pipelines/run_c3_region_experiment.py` | `305584a6e925b0b22da18c92fead54bbd7a8861d27194480bfe471fda5316385` |
| `src/seis_interp/pipelines/run_forge_grid_experiment.py` | `a8d0422de5bbfccf7b04ae6f3b9db9fe9554082f3682e21979b807cffcbc91a1` |
| `src/seis_interp/evaluation/c3_region_results.py` | `8362881f1ee3df6851425cb0094eb62526c9105689c757461be9c8142e96b987` |
| `src/seis_interp/evaluation/forge_grid_results.py` | `97306c53956da6f4d16883b4336efb16efbe7f7fa3f0bec3bbafc5b2415c70d1` |
| `src/seis_interp/evaluation/forge_mvp_metrics.py` | `d8b4648bf669f6ec2448114574dc88f6c74a37c74e92df3a66538663ed02b99c` |
| `src/seis_interp/evaluation/fair_grid_trace_snr.py` | `2b5a78ee15ecbcd7a009daa3fc740acd9b387c3171d5b231c1e4d2bc7a177ea2` |
| `src/seis_interp/evaluation/trace_snr.py` | `709332bdec556bb9f7a8a7f34690c94c28398ce1211655e0484c2cdbf637bb62` |

現在の base commit は `05c92e79ce80ca26f8b60f25a11f7ae4732e1b73`。ユーザー指示により Sprint 1・2 では commit / tag と実験起動を行わない。現在は dirty worktree であり、本番開始前の code commit / tag 凍結は未実施。

## 長時間実行の判断

厳密再開には EMA・optimizer だけでなく、消費済み batch の sampler cursor と先読み済み patch を含める。先読み worker の先行した RNG をそのまま再開点にすると更新列が変わるためである。時刻による history の差は性能計測に限り、学習値・全乱数状態は保存する。

Q1(ii) に従い、手法内の選択後は seed の学習・予測を先行させ、test 振幅の読み出しと採点を全手法の barrier 後へ分離する。C3 の元振幅全体の hash 確認も preparation と barrier 後の test 読み出しで行い、S1 中に test 波形を hash のために読むことを避ける。

watchdog は自 worker の終了と supervisor による再起動を使い、別 process の停止操作を必要としない。候補の順序変更はメモリ適合だけに限定し、barrier 通過前は再試行待ちや shard 未完了を S1 完了として扱わない。GPU 上の実測受入と本番起動は、この CPU 実装作業に含めない。上の SHA256 表は更新時点の snapshot であり、並行編集中の Sprint 2 code は起動前に再凍結する。


長時間 run の復旧では、再試行枠を run ごとに数え、資源不足の待機と実行失敗を分ける。CUDA context の障害と watchdog の終了は独立 process で復旧し、正常な隣接 run の進捗・試行枠を失わせない。POCS / DRR は更新単位の再開と進捗報告を持たないため watchdog を適用せず、既定の壁時計上限も設けない。

S1 barrier は一度通過した事実を永続化する。後続 worker の障害で採点許可を取り消すと、未採点 seed が3 seed の集計に紛れ込むためである。発散は手法全体の中止ではなく最下位の最終結果とし、集計は欠測・発散 seed を明示して3完了未満の平均・SDを公表しない。候補選択時のメモリ予約を admission と同じ lock で確保し、完了・import 対象に不要なメモリ待ちを課さない。

## 本番 launch rehearsal と開始承認（2026-10-24）

ユーザーは2026-10-24に本番 launch と、FORGE CCNet の2 shard・重い候補優先の scheduling を
承認した。この準備作業では GPU 実行・実験の本 prepare・signal・commit / tag を行わない。
本番 code の基準は v4 `11ebcc78d39cd50f8d5012d2b8fb2bfc79c6a7d9`
（`study-067-launch-v4`）。今回の production `src/` 差分は rehearsal checker に限り、
学習・予測・admission・supervisor は v4 と同一に保つ。最終 scheduling と文書を含む clean な
launch commit / tag は operator が確定する。上の古い base commit / dirty 状態の記述は当時の
snapshot であり、現行 launch の基準は本節とする。

rehearsal 評価 seed **990671–990673 は burned**。科学的評価へ再利用しない。
本番の7321–7323と model seed 7301 は変更しない。

### 証拠の所在と当時の acceptance

experiment の共通親は `runs/study_067_forge_c3_common_grid_rehearsal/`。
以下の C3 / FORGE はその下の directory 名、証拠 root には `env.sh`、GPU sample、当時の
`acceptance.txt` がある。これらは read-only の履歴として保持する。

| 版 / commit | 証拠 root | C3 / FORGE experiment |
|---|---|---|
| v1 / `ebe3c9306268` | `runs/study_067_forge_c3_common_grid_rehearsal/` | `20261023T150752712084Z_ebe3c9306268_traces` / `20261023T150757511431Z_ebe3c9306268_forge` |
| v2 / `5130d2edea8e` | 同じ親。独立した full rehearsal / acceptance は未実施 | `20261023T170034621262Z_5130d2edea8e_traces` / `20261023T170041027002Z_5130d2edea8e_forge`（prepare 証拠。独立完走としない） |
| v3 / `7d7d4d1f0e00` | `runs/study_067_forge_c3_common_grid_rehearsal_v3/` | `20261023T182955930826Z_7d7d4d1f0e00_traces` / `20261023T183003367014Z_7d7d4d1f0e00_forge` |
| v4 / `11ebcc78d39c` | `runs/study_067_forge_c3_common_grid_rehearsal_v4/` | `20261023T222329917833Z_11ebcc78d39c_traces` / `20261023T222334889775Z_11ebcc78d39c_forge` |

| acceptance area | v1 保存記録 | v2 | v3 保存記録 | v4 保存記録 | v4 今回の checker |
|---|---|---|---|---|---|
| Freeze and launch | PASS | 未実施 | PASS | PASS | PASS |
| Admission and sharding | FAIL | 未実施 | PASS | FAIL | PASS |
| Warm-up | FAIL | 未実施 | PASS | PASS | PASS |
| Resume | FAIL | 未実施 | FAIL | FAIL | PASS |
| Specific faults | FAIL | 未実施 | PASS | PASS | PASS |
| Divergence | PASS | 未実施 | PASS | PASS | PASS |
| Barrier | PASS | 未実施 | PASS | PASS | PASS |
| Summaries | PASS | 未実施 | PASS | PASS | PASS |
| R0 (imports) | PASS | 未実施 | PASS | PASS | PASS |
| 合計 | 5 PASS / 4 FAIL | — | 8 PASS / 1 FAIL | 7 PASS / 2 FAIL | **9 PASS / 0 FAIL** |

v1 の保存記録には capacity 超過・予測時 reserved 増加・C3 seal の `dataset_id` 欠落・
retry peak の下方変化による FAIL がある。v3 の Resume FAIL は C3 ReGSI seed child の
予期しない再起動。v4 の残る2 FAIL は下記の checker 基準であり、他の7基準は変更しない。
過去の失敗表を現在の判定で書き換えず、v4 の再確認結果を別に記録する。

実際に見つけて修正した4件は次のとおり。

1. **worker-lock の restart race**：v1 で ReGSI の fork DataLoader child が worker の
   `flock` を約5秒保持し、親の終了後すぐの再起動が失敗した。v2 `5130d2e` で lock 待ちを
   restart credit を消費しない pending とし、最大120秒再確認する。fork child は継承した
   GPU startup-lock の fd を親の unlock なしに閉じる。
2. **使用しない GPU の stray CUDA context**：全 device の `mem_get_info` が process ごとに
   約0.5 GiBの context を残し、rehearsal 全体では各 GPU 約5 GiBになった。v3 `7d7d4d1` で
   free / total を CUDA context を作らない NVML / nvidia-smi に変更し、allocator の参照は
   選択した device だけに限定した。
3. **C3 prediction warm-up の不足**：単一の最大 tile では境界 shape と allocation lifetime を
   再現できず、予測時に reserved が1.94から3.04 GiBへ増えた。v3 でゼロ入力による実際の
   tile 列を replay する。出力と RNG は変更しない。retry peak は autotuning を再利用して
   11.7%下がるため、既に v3 で上限110%だけを検査する cleanup gate に修正済み。
4. **無報告の正常な prediction を watchdog が stall と誤認**：v3 の C3 ReGSI で主 EMA と
   shadow EMA の予測等が長く無報告となり、seed child を再起動した。v4 `11ebcc7` で
   prediction・warm-up・checkpoint / array write・alignment・restoration・scoring の実作業に
   よる30秒間隔の heartbeat を既存 report channel に追加した。watchdog の閾値、p95 の
   標本、計算順・出力・RNG は保持する。

### 今回の checker 2基準の改訂理由

benchmark-on CCNet twin は fault 前から非決定性で分岐し、FORGE v4 は更新4で約1.6e-7、更新7で約4.6e-6の差がある。
FORGE loss は約0.403–0.462、100更新単位の標準偏差は約0.003–0.009と平坦で、Pearson は
v3 0.9921 / v4 0.9877と noise に敏感。v4 の平均絶対差は0.0027（約0.6%）で、validation の
relMSE 差は約0.003%、mean trace SNR 差は約0.0001 dB。relative 2%案も C3 の小さい late loss
（約0.06–0.1）で2.2–3.4%となるため採用しない。

ユーザーの revision 2 に従い、**再実行更新の mean |primary − twin| ≤ 3 × resume 前の任意の
連続100更新窓の mean |primary − twin| の最大値**を使う。resume step は log から読む。
今回の fault@250 は200から再開するため、post は201–300。sliding window での実測比は、
v3 C3 / FORGE が0.3211 / 0.8521、v4 が0.4224 / 0.8130。
初期重みの模擬再開（primary の1–100で post を置換）は38.3630 / 16.6500 / 29.8822 / 9.6524、
step-100 の模擬再開（101–200で置換）は5.8045 / 7.6948 / 4.5321 / 5.1285となり、全て3を超える。
提示された0.36 / 0.88 / 0.53 / 0.85は固定窓1–100と101–200だけなら再現する。
全ての sliding window を使う今回の数値・baseline・証拠・再確認 command は
[rehearsal.md](rehearsal.md#revised-benchmark-on-and-seed-admission-gates-2026-10-24) に記録する。
relMSE ≤ 1%、SNR ≤ 0.05 dB、checkpoint digest の完全一致、benchmark-off twin の byte 一致は
変更しない。

FORGE a05 seed の別 GPU 同時実行は必須の正しさではなく、空き容量に依存する。
v4 の seed 学習・予測は2026-10-23 UTCで22:59–23:12、23:12–23:23、23:23–23:32と直列になった。
cuda:0 は66 GiBの seed budget、cuda:1 には他 slot があり、待機は正しい。
全 seed の最終 admission と、全 memory wait で eligible device の未予約容量が必要量未満で
あることを要求する。1,041件中1,040件は raw free 自体が70.9 GB未満。
残る1件は cuda:0 raw free 81.1 GBだが、23:00:54の sample が他 seed の66 GiB slot と
7.6 GBの allocator 使用を記録し、63.2 GBが regrowth 用に予約済み。未予約は約17.9 GBで不足。
その待機期間内の free-memory vector に対応する sample の前後 slot 読み取りの両方で
検証する。corroboration がなければ FAIL とする。a05 S1 shard が別 GPU で重なる条件は保持し、
v3 では seed 同時実行自体も確認できる。

### 実行専用 scheduling と identity

両 `config.yaml` の priority だけを重い順へ変更し、FORGE CCNet の `worker_shards` を
rehearsal と同じ2にした。変更した5 list は全25 IDを一度ずつ列挙する。行内は b 昇順、列内は
a 昇順。FORGE CCNet は G0 `a01_b02` を最初にして a05→a04→a03→a02→a01残り、
C3 CCNet は a05→a04→a03→a02→a01。ReGSI は両 dataset とも b05→b04→b03→b02→b01で、
FORGE の gate `a01_b05` は先頭のまま。FORGE NeRSI は a03→a04→a01→a02→a05。
C3 NeRSI と DRR / POCS は不変。

priority を full grid に適用してから位置の剰余で shard を分ける。FORGE `0/2` は
`a01_b02, a05_b02, a05_b04, ...`、`1/2` は `a05_b01, a05_b03, a05_b05, ...`から始まり、
両方へ大きな候補が分配される。各 shard はその中の最初にメモリへ収まる候補を共通 lock 内で
予約する。空き不足で軽い候補へ進む例外は lane plan の承認どおり。shard 間・method 間に
厳密な全順序はなく、G0 は shard 0 の先頭であって全 shard の barrier ではない。
そのため両 a05 用の容量を開始時に確保するには、rehearsal と同じ staged startup を本番にも
使う。G0 の本番50,000更新を経るため、rehearsal の5分期限は使わない。
operator の本 prepare、環境、frozen supervisor、監視、`--only-dead-workers` の正確な command は
[README の launch 節](README.md#本番-launchoperator) に置く。

CPU の `prepare --check-only` で coverage は C3 **30 import / 95 remaining**、FORGE
**29 / 96**のまま。temporary check-only preparation 上で旧 HEAD の候補と新 schedule の候補を
同じ実入力の alignment・観測数で resolve し、各 dataset の5手法×25候補×4 seed、計500ずつの
identity hash が全て不変であることを確認した。candidate IDを key とした全 seed の hash 表の
aggregate SHA256 は C3 `3fabe14597b94d2f8f4e7bf2f998d3a40a09d1ed65c94d186a6971f95fa1a23a`、
FORGE `50b14152b698aab9251eb6795cc7a558b05e0a86f2fcbb6e46dffbed512373be`。
`runs/` の証拠は変更せず、再確認出力は `/tmp/study067-launch-prep/` に保存した。
