# study_071: C3 / FORGE 共通 25 候補による条件付き implicit trace field

Status: `completed` — S1 25 候補と 3 seed 評価を両 dataset で完了（失敗 0）。事前登録した床は両 dataset で達成（C3 14.4023 dB > 10、FORGE 1.4861 dB > 0.4）。

## 目的

study_070 の測定は、neural-implicit の不足が出力単位でもハイパーパラメータでもなく**機構**にあることを
示した。同一の FORGE clean_target 34,153 trace 上で、予測時に近傍の観測トレースを読む4手法は相関中央値
0.4544–0.7275、波場を重みだけに符号化する2手法は 0.2159–0.2600 だった。

`cinr_real`（[`models/trace_context_field.py`](../../src/seis_interp/models/trace_context_field.py)）は
field を4座標の連続関数のまま保ちつつ、query が `context.neighbor_count` 本の観測トレースを読む。近傍は
field 自身が読む4正規化座標の距離だけで決まり、波形では決まらない。query は自分自身を文脈から除外する。

study_070 と同一の入力・mask・validation holdout・予算・seed・選択規則を使う。study_067 / study_070 の
test 結果を見た後に決めた追加実験であり、対比は両 study の事前登録対比群の外にある探索的な比較とする。

**公平性の宣言**: この baseline は予測時に観測データを読むため、機構が提案法（ReGSI）に近づく。
重みのみに符号化する形式（`nersi_profile`、`inr_real`）とは異なるクラスであることを明示して報告する。
時間 codec は提案法と同じ `models/trace_codec.py` を共有するので、対比は空間機構に絞られる。

## 条件

- 探索の正本（両 dataset 共通、1手法 5×5）: [`grid.yaml`](grid.yaml)
- dataset 別: [C3 config](c3/config.yaml) / [inputs](c3/inputs.yaml) / [method](c3/cinr_real.yaml)、
  [FORGE config](forge/config.yaml) / [inputs](forge/inputs.yaml) / [method](forge/cinr_real.yaml)
- 固定値の例外と理由: [`dataset_differences.yaml`](dataset_differences.yaml)
- 評価・事前登録規則: [`evaluation.yaml`](evaluation.yaml)
- 採用理由と未決事項: [`decisions.md`](decisions.md)

config・inputs のいずれにも書けない値は次のとおり。

| 値 | C3 | FORGE |
|---|---:|---:|
| 整列 canvas のサンプル数 | 384 | 2172 |
| width | 120 | 120 |
| パラメータ数（25 候補で共通） | **362,642** | **362,642** |
| 目標 363,269 比 | −0.1726 % | −0.1726 % |

codec が畳み込みなのでパラメータ数は時間長に依存せず、**両 dataset で同一**。宣言した
`model.time_sample_count` は実行時に整列 canvas 長へ上書きされるため推論には入らない。

本番形状での実測（H100 NVL、40 更新 + 1 予測 batch、batch 64）:

| k | C3 ms/update | C3 peak reserved | FORGE ms/update | FORGE peak reserved | admission C3 / FORGE |
|---:|---:|---:|---:|---:|---|
| 1 | 2.571 | 0.242 GiB | 4.176 | 0.951 GiB | 3 / 3 GiB |
| 2 | 2.604 | 0.342 GiB | 5.392 | 1.686 GiB | 3 / 4 GiB |
| 4 | 2.629 | 0.523 GiB | 8.019 | 2.646 GiB | 3 / 5 GiB |
| 8 | 3.354 | 0.879 GiB | 14.515 | 4.578 GiB | 3 / 7 GiB |
| 16 | 5.166 | 1.598 GiB | 25.638 | 8.578 GiB | 4 / 11 GiB |

admission は study_067 の C-0 規則 `max(ceil(最大 reserved GiB) + 2, 3)`。S1 と 3 seed の合計は
C3 約 1.4 GPU-h、FORGE 約 5.1 GPU-h。

## 結果

生成元: [C3 run](../../runs/study_071_forge_c3_context_field/20261109T042809568788Z_766f4f10a666_traces) /
[FORGE run](../../runs/study_071_forge_c3_context_field/20261109T042814452404Z_766f4f10a666_forge)。
両 run で `dirty_worktree: false`、`total_runs: 28`、`target_waveforms_read: 0`、
S1 完了後に seed ごと 1 回だけ test を採点。8 partition はすべて study_067 / study_070 と一致。
sealed parameter 数は両 dataset とも 362,642。

### S1 選択

| | 採用 cell | k | base | validation relMSE | validation SNR | best-of-25 SNR | 床 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C3 | `a03_b04` | 4 | 1.15 | 0.0510 | 14.4023 dB | 14.4023 dB | > 10 dB | **達成** |
| FORGE | `a05_b01` | 16 | 1.05 | 0.7652 | 1.4861 dB | 1.4861 dB | > 0.4 dB | **達成** |

ゼロ予測 gate は両 dataset で **25/25 が `passed`**（study_070 の FORGE は 25/25 が `failed`）。

近傍数ごとの validation（5 base の平均）:

| k | C3 relMSE | C3 SNR (dB) | C3 相関 | FORGE relMSE | FORGE SNR (dB) | FORGE 相関 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0689 | 13.0678 | 0.9781 | 0.9031 | 0.6631 | 0.3722 |
| 2 | 0.0624 | 13.5677 | 0.9807 | 0.8505 | 1.0037 | 0.4350 |
| 4 | 0.0619 | 13.7495 | 0.9815 | 0.8169 | 1.2078 | 0.4743 |
| 8 | 0.0613 | 13.7942 | 0.9819 | 0.7978 | 1.3255 | 0.4938 |
| 16 | 0.0640 | 13.4920 | 0.9808 | 0.7900 | 1.3636 | 0.5035 |

FORGE は k に対して単調、C3 は k = 4–8 で最良。

### test（3 seed 7321 / 7322 / 7323、採用 cell）

| | trace 数 | mean trace SNR (dB) | relMSE | global NMSE | 相関中央値 |
|---|---:|---:|---:|---:|---:|
| C3（all_eligible） | 104,858 | **14.2516 ± 0.1536** | 0.0540 ± 0.0026 | 0.0523 | 0.9836 |
| FORGE（clean_target） | 34,153 | **1.4558 ± 0.0168** | 0.7679 ± 0.0029 | 1.4894 | 0.5169 |

seed 別 mean trace SNR: C3 14.0743 / 14.3372 / 14.3434 dB、FORGE 1.4692 / 1.4613 / 1.4369 dB。

### 採否

同じ test domain・population における study_067 / study_070 の記録との対比。

| FORGE test | mean trace SNR (dB) | relMSE | 相関中央値 | global NMSE |
|---|---:|---:|---:|---:|
| regsi_real | 3.0342 | 0.5777 | 0.7301 | 1.0628 |
| ccnet5d_grid | 1.6010 | 0.7368 | 0.5743 | 1.1016 |
| **cinr_real** | **1.4558** | **0.7679** | **0.5169** | **1.4894** |
| drr_grid | 1.0910 | 0.8033 | 0.4561 | 1.0681 |
| pocs_grid | 1.0510 | 0.8130 | 0.4532 | 1.0565 |
| nersi_profile | 0.3126 | 0.9416 | 0.2646 | 1.0444 |
| inr_real | 0.1011 | 1.0071 | 0.2185 | 1.0640 |

| C3 test | mean trace SNR (dB) | relMSE | 相関中央値 |
|---|---:|---:|---:|
| regsi_real | 19.9918 | 0.0126 | 0.9951 |
| ccnet5d_grid | 18.8614 | 0.0164 | 0.9942 |
| drr_grid | 17.7448 | 0.0215 | 0.9921 |
| **cinr_real** | **14.2516** | **0.0540** | **0.9836** |
| pocs_grid | 13.8043 | 0.0587 | 0.9822 |
| nersi_profile | 12.5611 | 0.0681 | 0.9743 |
| inr_real | 11.3884 | 0.0941 | — |

`cinr_real` を neural-implicit の代表手法として採用する。FORGE では両古典手法を上回り、
C3 では POCS と両 weight-only field を上回る。事前登録した対比（共有 bootstrap 重み、Holm 族）は
[`evaluation.yaml`](evaluation.yaml) の規則で別途算出する。

FORGE の global NMSE 1.4894 は本比較で最大である（他手法は 1.0444–1.1016）。
この population では全手法が 1 を超えており、`cinr_real` の誤差は高エネルギー trace に
より強く集中している。study_070 の診断が示した短オフセット側の難しさと整合する。

## 実行

```bash
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

C3=$(PYTHONPATH=src "$PY" scripts/run_c3_region_experiment.py prepare \
       --study studies/study_071_forge_c3_context_field/c3)
FG=$(PYTHONPATH=src "$PY" scripts/run_forge_grid_experiment.py prepare \
       --study studies/study_071_forge_c3_context_field/forge)
"$PY" scripts/launch_grid_experiment.py forge --experiment "$FG"
"$PY" scripts/launch_grid_experiment.py c3    --experiment "$C3"
```

prepare 後、launch の前に各 experiment の `input/seed_*/partition.parquet` の sha256 が study_067 の
[C3 run](../../runs/study_067_forge_c3_common_grid/20261024T110740093298Z_5ad36bdb74fd_traces) /
[FORGE run](../../runs/study_067_forge_c3_common_grid/20261024T110745841828Z_5ad36bdb74fd_forge) の
`experiment.json` の記録と一致することを確認する。study_070 と同じ8 partition を使う。
