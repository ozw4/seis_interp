# study_070: C3 / FORGE 共通 25 候補による implicit trace field

Status: `completed` — S1 25 候補と 3 seed 評価を両 dataset で完了（失敗 0）。FORGE の事前登録した床 0.4 dB は未達。事前登録の規則により追加調整・第3形式・救済 arm は行わない。

## 目的

neural-implicit NN の代表手法を、新規実装のトレース出力 implicit field `inr_real`
（`src/seis_interp/models/trace_coordinate_field.py`、4連続座標 → Fourier 特徴 → GELU MLP →
線形ヘッドで全時間サンプル）に置き、study_067 と同一の入力・mask・validation holdout・予算・seed・
選択規則で両 dataset を測る。

既存の neural-implicit 実装はどちらも時間を conv decoder の出力軸にしている。断面単位 NeRSI は
study_067 FORGE test で 0.3560 dB（事前登録の床 0.4 dB 未達）、study_068 / study_069 の whole-shot
50% ではゼロ予測（relMSE 1.0000）だった。トレース単位 NeRSI も 3段の 2× pixel shuffle で 8 列 canvas を
作り列0だけを返すため、decoder 出力の 7/8 がトレースへ届かない。`inr_real` は canvas を持たず、
出力サンプルを1つも捨てない。

study_067/068/069 の test 結果を見た後に決めた追加実験である。手法内の設定は S1 validation だけで選ぶが、
study_067 の5 arm との対比は事前登録の対比群の外にある探索的な比較として扱う。

## 条件

実行条件の正本は次のファイルとする。

- 探索の正本（両 dataset 共通、1手法 5×5）: [`grid.yaml`](grid.yaml)
- dataset 別: [C3 config](c3/config.yaml) / [inputs](c3/inputs.yaml) / [method](c3/inr_real.yaml)、
  [FORGE config](forge/config.yaml) / [inputs](forge/inputs.yaml) / [method](forge/inr_real.yaml)
- 固定値の例外と理由: [`dataset_differences.yaml`](dataset_differences.yaml)
- 評価・事前登録規則: [`evaluation.yaml`](evaluation.yaml)
- 採用理由と未決事項: [`decisions.md`](decisions.md)

config・inputs のいずれにも書けない値は次のとおり。

| 値 | C3 | FORGE |
|---|---:|---:|
| 整列 canvas のサンプル数 | 384 | 2172 (= 1001 + max_delay 1171) |
| 採用 `hidden_width` | 249 | 126 |
| 25 候補が共有するパラメータ数 | 362,679 | 364,296 |
| 目標 363,269 比 | −0.1624 % | +0.2827 % |

25 候補は単一のパラメータ数を共有する（`training.weight_decay` と `model.frequency_base` は
どちらもパラメータ数を変えない）。`model.time_sample_count` は実行時に整列 canvas 長へ上書きされる。

## 結果

生成元: [C3 run](../../runs/study_070_forge_c3_trace_field/20261109T024714931416Z_63414473e4b2_traces) /
[FORGE run](../../runs/study_070_forge_c3_trace_field/20261109T024720614852Z_63414473e4b2_forge)。
両 run で `dirty_worktree: false`、`total_runs: 28`、`target_waveforms_read: 0`、
S1 完了後に seed ごと 1 回だけ test を採点（`test_scoring_calls: 1`）。
8 partition はすべて study_067 の sealed hash と一致。

### S1 選択（validation `masked_trace_relative_mse` 最小、同値は candidate ID）

| | 採用 cell | wd | base | validation relMSE | validation mean trace SNR | best-of-25 SNR | 床 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C3 | `a05_b04` | 0.1 | 1.15 | 0.0950 | 11.3812 dB | 11.3812 dB | > 10 dB | **達成** |
| FORGE | `a05_b01` | 0.1 | 1.05 | 1.0033 | 0.1201 dB | 0.1201 dB | > 0.4 dB | **未達** |

25 候補の範囲は C3 validation relMSE 0.0950–0.1093（mean trace SNR 10.8108–11.3812 dB）、
FORGE 1.0033–1.3394（−1.0945–0.1201 dB）。**FORGE は 25 候補すべてが validation relMSE > 1**。
両 dataset で採用 cell が grid の角（weight decay 最大 × Fourier base 最小）にあり、
事前登録した探索範囲は最適点を挟んでいない。

ゼロ予測 gate（更新 901–1000、記述のみ）: C3 は 25/25 が `passed`、FORGE は 25/25 が `failed`。
FORGE の学習 loss（最終 1000 更新平均）は 0.2855–0.3297 で、ゼロ予測の loss 0.4609 を全候補が下回る。

### test（3 seed 7321 / 7322 / 7323、採用 cell）

| | trace 数 | mean trace SNR (dB) | relMSE | global NMSE | 相関中央値 |
|---|---:|---:|---:|---:|---:|
| C3（all_eligible） | 104,858 | **11.3884 ± 0.0140** | 0.0941 ± 0.0003 | 0.0915 | 0.9683 |
| FORGE（clean_target） | 34,153 | **0.1011 ± 0.0073** | 1.0071 ± 0.0012 | 1.0640 | 0.2185 |

seed 別 mean trace SNR: C3 11.3779 / 11.3830 / 11.4042 dB、FORGE 0.0933 / 0.1078 / 0.1021 dB。

### 採否

study_067 の記録（[seed_summary.csv](../../results/study_067_forge_c3_common_grid/20261103T231944Z_ba805bd26f6a_analysis/seed_summary.csv)、
同じ test domain・population）の `nersi_profile` は C3 12.5611 dB / relMSE 0.0681、
FORGE 0.3126 dB / relMSE 0.9416。`inr_real` は**両 dataset でこれを下回る**。

neural-implicit の代表手法を `inr_real` へ置き換える根拠は得られなかった。事前登録どおり、
FORGE の床未達をもって本 study の調整を終了する。事前登録した study_067 との対比
（共有 bootstrap 重み、Holm 族）は [`evaluation.yaml`](evaluation.yaml) の規則で別途算出する。

## 実行

```bash
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

C3=$(PYTHONPATH=src "$PY" scripts/run_c3_region_experiment.py prepare \
       --study studies/study_070_forge_c3_trace_field/c3)
FG=$(PYTHONPATH=src "$PY" scripts/run_forge_grid_experiment.py prepare \
       --study studies/study_070_forge_c3_trace_field/forge)
"$PY" scripts/launch_grid_experiment.py forge --experiment "$FG"
"$PY" scripts/launch_grid_experiment.py c3    --experiment "$C3"
```

prepare 後、launch の前に各 experiment の `input/seed_*/partition.parquet` の sha256 が study_067 の
[C3 run](../../runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces) /
[FORGE run](../../runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge) の
`experiment.json` の記録と一致することを確認する。

全 seed 終了後に採点する。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 nice -n 19 "$PY" scripts/evaluate_study070.py evaluate \
  --evaluation studies/study_070_forge_c3_trace_field/evaluation.yaml \
  --c3 "$C3" --forge "$FG" \
  --output results/study_070_forge_c3_trace_field/<result-id>
```

config に書けない規則は [`evaluation.yaml`](evaluation.yaml) に置く（S1 床、ゼロ予測 gate、
Holm 族、study_067 の共有 bootstrap 重みと pin した sha256、partition 同一性）。
