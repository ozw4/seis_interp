# FORGE random whole-shot50 で学習文脈を test 文脈に揃える

Status: `prepared` — 実行仕様を凍結した段階。本 study の結果は未確定。

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
