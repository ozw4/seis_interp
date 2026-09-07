# Masked gather inputs

`MaskedGatherInputs` は、C3 の selected volume から target shot の観測済み
trace と周辺 shot の context gather を組み立てる、model-independent な runtime
input である。`random_trace` の部分観測 target と `random_whole_ffid` の完全欠損
target を同じ contract で表す。

## Fields

`B` を batch size、`K` を context shot 数、`R_x` / `R_y` を selected volume の
receiver crop、`T` を time sample 数とすると、6 field の shape は次のとおりである。

| Field | Shape | Meaning |
|---|---:|---|
| `target_observed` | `(B, R_x, R_y, T)` | target shot のうちモデルへ見せる観測値。非観測 cell は exact zero。 |
| `target_observation_mask` | `(B, R_x, R_y)` | target observation が利用可能な cell。 |
| `context_gathers` | `(B, K, R_x, R_y, T)` | 周辺 shot の観測値。利用不能 cell は exact zero。 |
| `context_availability` | `(B, K, R_x, R_y)` | context gather ごとの利用可能 cell。 |
| `source_deltas_m` | `(B, K, 2)` | `context source - target source` の x/y 差（m）。 |
| `target_coordinates` | `(B, 2)` | selected volume 全体の source x/y 最小値・最大値で `[0, 1]` に正規化した target 座標。 |

`random_trace` では、target observation に observed trace と exact-zero の
evaluation target cell が混在する。`random_whole_ffid` では target observation mask
がすべて `False` になり、`target_observed` 全体が zero になる。

Receiver dimensions は volume metadata が示す任意の正の dense crop を引き継ぐ。
たとえば full C3 grid の `8 x 68` に加え、benchmark例の `8 x 32` や `3 x 68`
もこのmodel-independent input contractで扱う。固定receiver shapeが必要な既存model
（`TraceGraphInterpolator`など）は、その制約をmodel boundaryで別途検証する。

## Target and context selection

Target は、少なくとも1つの evaluation target cell を持つ source cell である。
Context 候補は、少なくとも1つの observed cell を持つ source cellに限る。各 target
自身を除外し、source x/y の Euclidean distanceで近い順に選び、同距離の場合は
volume source axes の lexicographic flat indexで順序を決める。

Target coordinate の volume-minmax scaling は、mask roleではなく selected volume
内の全 source positionから計算する。このため、同じ volume geometryであれば mask
seedが変わっても座標表現は変わらない。

## Leakage and memory boundary

この input は target observation と context gather だけを含み、evaluation target
amplitude、training target、loss mask、labelは含まない。High-level loaderは既存の
verified volume bindingと `ObservedC3Volume` を使用し、evaluation target amplitudeを
input assemblyへ持ち出さない。

Source objectは observed volumeへのviewと小さいgeometry indexだけを保持する。Full
volumeを別のshot-major contiguous arrayやTorch tensorへ複製せず、要求された
target/context batchだけをmaterializeして指定deviceへ送る。

これは既存の `WholeShotBatch` と `WholeShotTensorSource`、およびStudy 020/021の
whole-shot training pathを置き換えない。既存pathはtrain-only source gathersから
whole-shot targetを予測する契約であり、このruntime inputとは独立している。

## Usage

```python
import numpy as np

from seis_interp.data.c3_masked_gather_source import load_masked_c3_gather_source

source = load_masked_c3_gather_source(
    interim_dir=interim_dir,
    processed_dir=processed_dir,
    mask_dir=mask_dir,
    case_dir=case_dir,
    volume_dir=volume_dir,
    context_gather_count=2,
    device="cuda",
)
batch = source.inputs(np.asarray([0, 1], dtype=np.int64))
prediction = model(
    batch.target_observed,
    batch.target_observation_mask,
    batch.context_gathers,
    batch.context_availability,
    batch.source_deltas_m,
    batch.target_coordinates,
)
```

ここで `model` は、このcontractを将来利用するconsumerを示す概念例であり、対応する
model自体はまだ提供していない。
