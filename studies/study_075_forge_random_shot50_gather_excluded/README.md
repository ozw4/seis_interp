# FORGE random whole-shot50 で学習文脈を test 文脈に揃える

Status: `completed` — 予測・採点を完了（失敗 0）。clean_target は **+1.3029 dB / relMSE −0.2514** の改善で、同一 mask の study_069 の5手法中 ReGSI に次ぐ2位（CCNet5D・DRR・POCS の上）。事前の予測（改善するが古典手法帯まで）を上回った。

## 目的

study_072 からの変更は**1行だけ**である（`context.training_exclusion: source`）。学習時に query の
**source gather 全体**を文脈から除外し、学習時に見える近傍を test 時に見える近傍へ揃える。
予測時は観測トレースを一切捨てない。mask・入力・model・近傍数 16・Fourier base 1.05・予算・seed は
study_072 と同一で、[`cinr_real.yaml`](cinr_real.yaml) は study_072 の
[手法設定](../study_072_forge_random_shot50_context_field/cinr_real.yaml) にその1行を加えただけである。

## 動機（幾何だけの測定）

whole-shot mask では、観測 query は自分の shot 全体が観測されているので同一 gather に兄弟があるが、
test query は1本も持たない。study_072 は **test で存在しない情報に頼るよう学習していた**。

| mask | 学習 query の in-gather 近傍 | test query の in-gather 近傍 |
|---|---:|---:|
| study_071 random trace 70% | 2.61 / 16（96 % が1本以上） | 2.63 / 16（96 %） |
| study_072 random shot 50% | **5.52 / 16（100 % が1本以上）** | **0.00 / 16（0 %）** |

random trace mask では学習と test が一致しており、whole-shot mask では完全に不一致である。
study_071 が機能し study_072 が機能しなかった差はここにある。

gather を除外すると、学習文脈は test 文脈に全統計で一致する（study_072 の幾何で測定）。

| | in-gather | 最近傍距離（中央値） | 第16近傍距離（中央値） | 相対 offset 差 |
|---|---:|---:|---:|---:|
| 学習（study_072 のまま） | 5.52 / 16 | 0.03214 | 0.09684 | 0.1135 |
| 学習（gather 除外） | **0.00 / 16** | 0.03224 | **0.11632** | **0.1411** |
| test（参照） | **0.00 / 16** | 0.03229 | **0.11636** | **0.1391** |

## 事前の予測

**改善を予測する。** study_072 の後退は「情報が無い」ことと「無い情報に依存するよう訓練された」ことの
両方からなり、本変更は後者だけを取り除く。前者は残るので、study_071 の random trace mask での水準
（1.4558 dB / relMSE 0.7679）には届かないと予測する。到達目標は study_069 の古典手法帯
（POCS 0.7105、DRR 1.0033 dB）である。

**反証条件**: 改善しない場合、学習と test の文脈不一致は study_072 の後退の原因ではなく、
原因は gather 内近傍の欠落そのもの（情報量）だけである。

## 結果

生成元 [run](../../runs/study_075_forge_random_shot50_gather_excluded/20261110T020859184225Z_aecab8752d88)、
[採用図表・CSV](../../results/study_075_forge_random_shot50_gather_excluded/20261110T023513Z_aecab8752d88_analysis)。
`dirty_worktree: false`、`total_runs: 1`、`test_scoring_calls: 1`、runtime 1467.8621 s。
run は `context_training_exclusion: source` を記録している。入力10成果物は study_069 / study_072 と byte 一致。

### A/B（study_072 との差は学習時の文脈除外だけ）

| | 学習時の除外 | clean_target SNR (dB) | clean_target relMSE | all_eligible SNR (dB) | all_eligible relMSE | 学習 loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| study_072 | query | 0.3596 | 1.0005 | 0.3800 | 0.9984 | 0.2043 |
| study_075 | source | **1.6625** | **0.7490** | **1.6804** | **0.7479** | 0.2444 |
| 差 | | **+1.3029** | **−0.2514** | **+1.3004** | **−0.2505** | +0.0401 |

**学習 loss は上がり（0.2043 → 0.2444）、test は大きく改善した。** 学習と test の文脈不一致を
取り除いたときに期待される符号であり、model は本来の課題で学習するようになった。

### shot 支持別（clean_target）

| shot_support | shot 数 | study_072 SNR | study_075 SNR | study_072 relMSE | study_075 relMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| internal | 85 | +0.4193 | **+1.7245** | 0.9919 | **0.7397** |
| edge | 11 | −0.1023 | **+1.1832** | 1.0667 | **0.8212** |

edge shot は符号が負から正へ変わった（+1.2855 dB）。

### オフセット別（clean_target）

| 平均 offset | trace | study_072 相関 | study_075 相関 | study_072 relMSE | study_075 relMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 331.00 m | 4,880 | 0.2077 | **0.3559** | 1.1208 | **0.8968** |
| 641.88 m | 4,879 | 0.4029 | **0.5480** | 1.0040 | **0.7499** |
| 891.83 m | 4,879 | 0.4111 | **0.5678** | 0.9828 | **0.7279** |
| 1151.26 m | 4,879 | 0.4094 | **0.5778** | 0.9693 | **0.7086** |
| 1520.67 m | 4,880 | 0.4581 | **0.6208** | 0.9254 | **0.6619** |

相関中央値 0.3818 → 0.5431、global NMSE 1.2705 → 1.1854。全帯で改善し、最短オフセット帯が
依然として最も弱い（relMSE 0.8968、NMSE 1.1867）。

### 同一 mask・同一 test 集合での順位（study_069 の記録）

| 手法 | mean trace SNR (dB) | relMSE |
| --- | ---: | ---: |
| ReGSI | 2.5702 | 0.6191 |
| **cinr_real（study_075）** | **1.6625** | **0.7490** |
| CCNet5D | 1.2157 | 0.7898 |
| DRR | 1.0033 | 0.8192 |
| POCS | 0.7105 | 0.8901 |
| profile NeRSI | approximately 0 | 1.0000 |
| cinr_real（study_072） | 0.3596 | 1.0005 |

### 採否

whole-shot mask でも `cinr_real` を neural-implicit の代表手法として採用し、
`context.training_exclusion: source` を whole-shot mask の宣言条件とする。
判断理由と留保は [decisions.md](decisions.md) に置く。

## 条件

条件の正本は [`config.yaml`](config.yaml)、入力と選択元 hash は [`inputs.yaml`](inputs.yaml)。
mask は `mask_study_id` で study_069 の名前空間を引き継ぎ、波形は study_067 FORGE run の sealed 入力を
`physical_reference` で再利用する。`inputs.yaml` は study_069 と study_072 の記録も対比対象として pin する。

1 mask・1 model seed・1手法の記述的比較であり、study_072 / study_069 との対比は事後的・探索的である。

## 実行

```bash
PY=.venv/bin/python
"$PY" scripts/forge_trace_experiment.py prepare --study studies/study_075_forge_random_shot50_gather_excluded
"$PY" scripts/forge_trace_experiment.py launch  --experiment runs/study_075_forge_random_shot50_gather_excluded/<run-id>
"$PY" scripts/forge_trace_experiment.py score   --experiment runs/study_075_forge_random_shot50_gather_excluded/<run-id>
"$PY" scripts/forge_trace_experiment.py figures --experiment runs/study_075_forge_random_shot50_gather_excluded/<run-id> \
  --output results/study_075_forge_random_shot50_gather_excluded/<result-id>
```

prepare 後、launch の前に `input/partition.parquet` の sha256 が study_069 / study_072 と一致することを
確認する。1手法なので `paired_shot_differences.csv` は空になる。
