# study_061_forge_nersi_profile

Status: `completed` — 探索・選択・最終再学習（3 seed）・test評価が完了。最終3 seedはすべて学習が出力ほぼ0の状態から抜け出せず、主指標は1.0001（ゼロ予測と同等）。

## 目的

- study_060のNeRSI（トレース単位への改変版）を、原論文（Gao et al., 2025, Geophysics 90(1) V29–V42）と同じ断面単位の形で取り直す。
- 入力データ・マスク・検証集合V・探索段階と選択規則・予算・seed・bootstrapはstudy_060と同一にする。
- 断面単位NeRSIを、study_060の6手法の最終予測と同じtest集合で対比する。

study_060のtest結果（NeRSIが最下位）を見た後に決めた追加実験である。手法内の設定はVだけで選ぶが、
対比はstudy_060の事前登録の対比群の外にある探索的な比較として扱う。
study_060のトレース単位版との差は出力単位だけではない。入力座標の表現、S1・S2の探索範囲、
1更新あたりのサンプリング（断面をまとめて引く）も同時に変わるため、対比の差を出力単位だけの効果とはみなさない。
原論文からの逸脱は [decisions.md](decisions.md) にまとめた。

## 条件

実行条件の正本は次のファイルとする。

- 共通条件: [`config.yaml`](config.yaml)（study_060の`config.yaml`と`study_id`以外は同一）
- 入力: [`inputs.yaml`](inputs.yaml)（study_060の準備run 2本と最終評価run）
- 探索グリッド・seed・対比: [`grid.yaml`](grid.yaml)
- 中心設定: [`nersi_profile.yaml`](nersi_profile.yaml)

判断理由は [decisions.md](decisions.md) に置く。

断面は「震源セル1つ×受振測線1本」で、受振点32本を列とする（6×32×8 = 1,536断面）。
入力3座標は、震源の連続line・point座標と、受振測線に属する受振点の連続line座標の平均で、
いずれも丸めない。格子に丸めるのは受振点（断面の列）だけである。

| 格子からのずれ（study_060最終準備runの実座標） | 中央値 | 最大 |
|---|---:|---:|
| 震源のline方向（入力に連続値で使用） | 48.8 m | 115.8 m |
| 震源のpoint方向（入力に連続値で使用） | 2.8 m | 16.4 m |
| 受振点のpoint方向（格子の列に丸める） | 0.2 m | 3.4 m |
| 受振点のline方向（測線内） | 0.2 m | 0.6 m |

受振測線ごとの平均line座標は格子番号から最大0.04 mしか離れていないため、このsurveyでは r_x は実質的に格子番号と同じになる。

中心設定のパラメータ数は360,239（整列キャンバス4,343サンプル、目標363,269の−0.83%）。
S2の3候補は352,295・348,692・348,975で、S1〜Eの全候補が±5%に入る。
decoderは時間4,344×受振点32の断面を出し、最後の1サンプルを捨てて4,343サンプルのキャンバスに戻す。
検証用準備runでの事前確認（500更新）では、1更新あたり平均7.9断面（約250トレース分）をdecodeした。
lossに使うのはそのうちちょうど64本で、トレース単位版（64本だけをdecode）とFLOPsは揃っていない。

configに書けない規則はstudy_060と同じ（S1 9 run、S2 3、S3 3、E 1、R 4、最終3 seed、
各runをEMA 3水準でVの平均 `masked_trace_relative_mse` により採点）。違いは次のとおり。

- 探索は`nersi_profile`だけで、S1の軸は震源座標と受振測線座標のFourier帯域（`grid.yaml`）。
- 最終評価は`nersi_profile`の3 seedだけを採点し、study_060の6手法は同じtest集合での
  seed平均誤差を[最終評価run](../../runs/study_060_forge_fair_grid/20261008T030804955521Z_05c92e79ce80_final_evaluation/manifest.json)から読み込んで対比する。
  test波形の読み込みは`nersi_profile`の採点で1回だけ行う。

## 結果

最終評価: [`20261008T104151697886Z_05c92e79ce80_final_evaluation`](../../runs/study_061_forge_nersi_profile/20261008T104151697886Z_05c92e79ce80_final_evaluation/manifest.json)
（評価設定 [`final_evaluation.yaml`](../../runs/study_061_forge_nersi_profile/final_evaluation.yaml)）。
2026-10-08 10:42 UTCに正常終了し、test波形34,406本を1回だけ読んだ。参照したstudy_060の最終評価runとは、
test集合・行順・非oracle復元のスケールが一致することを確認済み。

### 探索（検証集合V、各runの最良EMAの平均 `masked_trace_relative_mse`）

| 段階 | run | 最良EMA | V | 学習loss（最後の1,000更新の平均） |
|---|---|---:|---:|---:|
| S1 | src_q025 × rl_q025 / rl_q05 / rl_q1 | 0.9998 | 0.9773 / 0.9800 / 0.9857 | 0.8372 / 0.8157 / 0.7989 |
| S1 | src_q05 × rl_q025 / rl_q05 / rl_q1 | 0.9998 | 1.0117 / 1.0115 / 1.0135 | 0.7989 / 0.7713 / 0.7616 |
| S1 | src_q1 × rl_q025 / rl_q05 / rl_q1 | 0.9998 | 1.0347 / 1.0390 / 1.0349 | 0.7825 / 0.7607 / 0.7519 |
| S2 | l2w74 / l1w123dec32 / k3 | 0.9998 / 0.9995 / 0.9998 | 0.9754 / 1.0000 / 1.0000 | 0.7870 / 0.9999 / 0.9999 |
| S3 | wd 3e-5 / 7e-5 / 1e-4 | 0.9998 | 0.9895 / 0.9752 / 0.9716 | 0.7912 / 0.8028 / 0.8319 |
| E | src_flat（震源帯域の下側延長） | 0.9998 | 0.9697 | 0.8913 |
| R | e_src_flat seed 7302 / 7303 | 0.9998 / 0.9995 | 0.9698 / 1.0000 | 0.8967 / 0.9999 |
| R | s3_wd0.0001 seed 7302 / 7303 | 0.9998 / 0.9995 | 0.9684 / 1.0000 | 0.8306 / 0.9999 |

選択はe_src_flat・EMA 0.9998（3 seed平均V 0.9799）。設定はlatent 2・encoder幅74・decoder [16, 8, 4]・
kernel 5・weight decay 1e-4・震源base 1.00001・受振測線base 1.035595（q = 0.25）。
探索20 runのうち4 run（S2の2 run、Rのseed 7303の2 run）と最終3 runは、学習lossが最初から最後まで約1.0のままで、
出力がほぼ0の状態から抜け出さなかった。lossは有限なので発散ではなく、規則どおりそのまま採点した。
選択した設定では、6 seedのうち学習が進んだのは7301と7302の2つだった。

### 最終評価（clean test 34,145本、`trace_oracle`）

| seed | 主指標 | global NMSE | 相関中央値 |
|---|---:|---:|---:|
| 7311 | 1.0002 | 1.0001 | approximately 0 |
| 7312 | 1.0000 | 1.0000 | approximately 0 |
| 7313 | 1.0000 | 1.0000 | approximately 0 |

seed平均±SDは1.0001 ± 0.0001。副次評価のseed平均は、all-eligible 0.9999、非oracle復元1.0001、スケール不変0.9999。

study_060の6手法との対比（差は `nersi_profile` − 他手法、source cell 192群のcluster bootstrap 10,000回、Holm補正は6対比内）:

| 対比 | 主指標の差 | 95%信頼区間 | Holm補正p値 |
|---|---:|---:|---:|
| nersi_profile − nersi_real | 0.1136 | [0.1045, 0.1227] | 0.0006 |
| nersi_profile − regsi_real | 0.4222 | [0.4084, 0.4364] | 0.0006 |
| nersi_profile − regsi_grid | 0.4117 | [0.3978, 0.4260] | 0.0006 |
| nersi_profile − ccnet5d_grid | 0.3475 | [0.3335, 0.3616] | 0.0006 |
| nersi_profile − pocs_grid | 0.2690 | [0.2550, 0.2833] | 0.0006 |
| nersi_profile − drr_grid | 0.2247 | [0.2122, 0.2375] | 0.0006 |

副次の3対比群でも、6対比すべての95%信頼区間が0より大きい側にある。

正本: [手法別集計](../../runs/study_061_forge_nersi_profile/20261008T104151697886Z_05c92e79ce80_final_evaluation/aggregate/method_summary.csv)、
[seed別指標](../../runs/study_061_forge_nersi_profile/20261008T104151697886Z_05c92e79ce80_final_evaluation/aggregate/seed_metrics.csv)、
[全対比](../../runs/study_061_forge_nersi_profile/20261008T104151697886Z_05c92e79ce80_final_evaluation/aggregate/contrasts.csv)、
[探索の記録](../../runs/study_061_forge_nersi_profile/fair_grid/nersi_profile/final/stage.json)。

### 参照実験（パラメータ帯域外、Vのみで採点）

C3の363,269パラメータ版（study_046）の断面単位NeRSIの構成をそのままFORGEのキャンバスで使うと
2,636,309パラメータになる。帯域外の参照実験として1 runだけ実行した
（[`reference_c3_layout.yaml`](reference_c3_layout.yaml)、run
[`20261008T111017877122Z_05c92e79ce80_nersi_profile_matched_tuning`](../../runs/study_061_forge_nersi_profile/20261008T111017877122Z_05c92e79ce80_nersi_profile_matched_tuning/manifest.json)）。
更新数・学習率・1更新64本・loss・seed 7301は本探索と同じで、testは読んでいない。

| EMA | V | global NMSE | 相関中央値 |
|---|---:|---:|---:|
| 0.999（主） | 1.4976 | 1.5701 | 0.1116 |
| 0.9995 | 1.4892 | 1.5643 | 0.1114 |
| 0.9998 | 1.3698 | 1.4445 | 0.1188 |

学習loss（最後の1,000更新の平均）は0.2119。学習時間は7.8分、GPUの最大確保メモリは1.1 GB。

### 構造を変えて帯域に収める参照実験（事前登録外、Vのみで採点）

latentの格子を小さくする2つの構造で、C3に近いlatentチャネル数を帯域内に収めた。
Fourier 40成分・FORGEの格子ナイキストのbase・weight decay 0・EMA 0.999（主）は上の参照実験と同じで、
seed 7301〜7303の3 seedずつ実行した（`reference_only`、testは読んでいない）。

- deep-time（[`reference_structure_deep_time.yaml`](reference_structure_deep_time.yaml)）: C3の363k構成
  （encoder 40、latent 28、decoder [29, 14, 8]）の前に、時間方向だけを2倍にする段（28チャネル、kernel 3）を4段加える。
  latentの格子は543×4から34×4になり、355,653パラメータ。
- rank-2 latent（[`reference_structure_rank2_latent.yaml`](reference_structure_rank2_latent.yaml)）: 各latentチャネルを
  時間×受振点の外積2つの和で作る（C3のstudy_046のrank2_latentと同じ機能）。encoder 30、latent 8、decoder [29, 14, 8]で353,755パラメータ。

| 構造 | seed | V（EMA 0.999 / 0.9995 / 0.9998） | global NMSE（0.9998） | 相関中央値（0.9998） | 学習loss |
|---|---:|---|---:|---:|---:|
| deep-time | 7301 | 1.0974 / 1.0891 / 0.9958 | 1.1640 | 0.2325 | 0.5535 |
| deep-time | 7302 | 1.0887 / 1.0825 / 1.0094 | 1.1036 | 0.2424 | 0.5467 |
| deep-time | 7303 | 1.0999 / 1.0928 / 1.0110 | 1.1132 | 0.2228 | 0.5794 |
| rank-2 latent | 7301 | 1.1181 / 1.1095 / 1.0370 | 1.0915 | 0.1297 | 0.7109 |
| rank-2 latent | 7302 | 1.0981 / 1.0916 / 1.0318 | 1.0840 | 0.1366 | 0.7156 |
| rank-2 latent | 7303 | 1.1106 / 1.1035 / 1.0391 | 1.0756 | 0.1342 | 0.7102 |

runは [`20261008T113312208369Z_05c92e79ce80_nersi_profile_matched_tuning`](../../runs/study_061_forge_nersi_profile/20261008T113312208369Z_05c92e79ce80_nersi_profile_matched_tuning/manifest.json)（deep-time）と
[`20261008T113312195206Z_05c92e79ce80_nersi_profile_matched_tuning`](../../runs/study_061_forge_nersi_profile/20261008T113312195206Z_05c92e79ce80_nersi_profile_matched_tuning/manifest.json)（rank-2 latent）。
6 runとも学習lossは最初の1,000更新で1.0を下回り、出力ほぼ0での停滞は起きなかった。

### deep-time構造の過学習対策（パラメータ数355,653のまま、事前登録外、Vのみで採点）

パラメータを増やさない3種類の対策を、deep-timeのseed 7301で1つずつ試した（他の条件は上表のdeep-timeと同じ）。

- 核ノルム: 復元した断面（時間×受振点）の核ノルムを `sqrt(時間×受振点)` で割った値に重み λ を掛けてlossに足す
  （論文の式11の低ランク正則化、[`reference_regularization_nuclear.yaml`](reference_regularization_nuclear.yaml)）。
- 座標ジッタ: 学習中だけ、断面の入力3座標に格子1セルの0.25倍・0.5倍以内の一様乱数を足す。ラベルは変えない
  （[`reference_regularization_jitter.yaml`](reference_regularization_jitter.yaml)）。乱数はトレース抽選と別の系列で、抽選順は変わらない。
- weight decay（[`reference_regularization_weight_decay.yaml`](reference_regularization_weight_decay.yaml)）。

| 対策（seed 7301） | V（EMA 0.999 / 0.9995 / 0.9998） | global NMSE（0.9998） | 相関中央値（0.9998） | 学習loss |
|---|---|---:|---:|---:|
| なし（上表のdeep-time） | 1.0974 / 1.0891 / 0.9958 | 1.1640 | 0.2325 | 0.5535 |
| 核ノルム λ = 0.01 | 1.0742 / 1.0664 / 0.9824 | 1.1555 | 0.2368 | 0.5631 |
| 核ノルム λ = 0.03 | 1.0771 / 1.0697 / 0.9926 | 1.0721 | 0.1971 | 0.5858 |
| 核ノルム λ = 0.1 | 1.0673 / 1.0626 / 1.0133 | 1.0407 | 0.1161 | 0.6452 |
| 座標ジッタ 0.25セル | 1.0248 / 1.0184 / 0.9510 | 1.1609 | 0.2700 | 0.6146 |
| 座標ジッタ 0.5セル | 0.9696 / 0.9631 / 0.9251 | 1.0815 | 0.2786 | 0.7362 |
| weight decay 1e-4 | 1.0000 / 1.0000 / 1.0002 | 1.0002 | approximately 0 | 0.9999 |
| weight decay 1e-3 | 1.0000 / 1.0000 / 1.0001 | 1.0000 | approximately 0 | 1.0000 |

weight decayの2 runは学習lossが約1.0のまま下がらず、出力ほぼ0にとどまった。

座標ジッタ0.5セルと、それに核ノルム λ = 0.01 を加えた組合せを3 seedで確認した
（[`reference_regularization_jitter05_seeds.yaml`](reference_regularization_jitter05_seeds.yaml)、
[`reference_regularization_jitter05_nuclear001.yaml`](reference_regularization_jitter05_nuclear001.yaml)）。

| 構成（EMA 0.9998） | seed 7301 / 7302 / 7303 のV | 3 seed平均 ± SD | 相関中央値 |
|---|---|---:|---|
| deep-time（対策なし） | 0.9958 / 1.0094 / 1.0110 | 1.0054 ± 0.0083 | 0.2325 / 0.2424 / 0.2228 |
| deep-time + 座標ジッタ0.5 | 0.9251 / 0.9371 / 0.9247 | 0.9289 ± 0.0071 | 0.2786 / 0.2582 / 0.2734 |
| deep-time + 座標ジッタ0.5 + 核ノルム0.01 | 0.9242 / 0.9276 / 0.9176 | 0.9231 ± 0.0051 | 0.2663 / 0.2647 / 0.2739 |

対策を入れた6 runとも、出力ほぼ0での停滞は起きなかった。

## 判断

- 事前登録した手順の結果として、上表を比較用の結果（`comparison-only`）として記録する。
- 出力ほぼ0での停滞は再実行せず、記録された値で採点した（[decisions.md](decisions.md)）。

## 実行手順

```bash
P=runs/study_060_forge_fair_grid
G=studies/study_061_forge_nersi_profile/grid.yaml
# 1. study_060の準備runを、入力を変えずにこのstudyへ封印し直す（各1回）
.venv/bin/python scripts/forge_derive_preparation.py \
  --parent $P/20260927T133848730415Z_25c5193588b2_validation \
  --study studies/study_061_forge_nersi_profile --method nersi_profile
.venv/bin/python scripts/forge_derive_preparation.py \
  --parent $P/20260927T133829534335Z_25c5193588b2 \
  --study studies/study_061_forge_nersi_profile --method nersi_profile
# 2. S1 → S2 → S3 → E → R を順に生成し、stage.json の commands を実行する
V=runs/study_061_forge_nersi_profile/<validation run>
F=runs/study_061_forge_nersi_profile/<final run>
.venv/bin/python scripts/forge_fair_grid.py --grid $G plan --method nersi_profile --stage S1 \
  --validation-preparation $V --final-preparation $F
.venv/bin/python scripts/forge_fair_grid.py --grid $G status
# 3. 選択と最終3 seedのyaml生成、実行後に最終評価
.venv/bin/python scripts/forge_fair_grid.py --grid $G select --method nersi_profile \
  --validation-preparation $V --final-preparation $F
.venv/bin/python scripts/forge_fair_grid_final_evaluation.py --config <evaluation.yaml>
```
