# study_071: C3 / FORGE 共通 25 候補による条件付き implicit trace field

Status: `prepared` — 実行仕様を凍結した段階。本 study の比較結果は未確定。

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
