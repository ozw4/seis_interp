# Grid-free relational trace graph

一トレースを一ノードとして、任意の絶対source/receiver座標から欠損波形を直接予測するPython API。
時間系列は共有codecで潜在系列へ変換し、有向の観測依存グラフ上で同期的に更新する。
固定receiver格子やFFIDの等間隔性は要求しない。物理座標・特徴順序は
[座標規約](coordinate_conventions.md#grid-free-trace-graph-features)に従う。

## 入力と固定前処理

`data/trace_graph_domain.py`の`load_benchmark_trace_graph_domain()`は、既存interim・partition・mask・caseの
hash、role、canonicalizationを検証する。安定IDはcanonicalな`array_row`であり、振幅は保持しない。
時間は`time_samples=(start, stop)`のhalf-open範囲で選ぶ。`volume_dir`は省略でき、指定時はcase bindingを
確認して観測・queryの両方をcrop内へ制限する。volumeの出力行順は
`inputs_lock["benchmark_volume"]["array_rows"]`に保持する。

`load_training_trace_graph_domain()`は`pool`を必須とする。

| pool | 固定訓練プールO0 |
|---|---|
| `all_train_traces` | canonicalなtrain行すべて |
| `mask_observed` | train caseのobserved行だけ |

`processing/trace_graph_preprocessing.py`の`fit_trace_graph_preprocessing()`は、O0の選択時間範囲を
float64でstream集計し、全サンプルのglobal RMSとCMPの算術平均を固定する。
`TraceGraphPreprocessing`は振幅尺度、座標原点、特徴尺度、方位しきい値、実際の`time_s`、fit domainを持つ。
人工的に隠す訓練行もO0の固定fitには寄与する。episodeやvalidation/testに適用するときは再fitしない。
全ゼロの訓練プールは、正のRMSを定義できないためエラーとなる。

`data/masked_trace_source.py`の`MaskedTraceSource`はmemory mapと行対応を持ち、`inputs(plan)`で必要な
観測supportだけを読む。元の物理振幅を変更せず、固定RMSで一度だけ正規化する。
query波形は最初からexact zeroである。`MaskedTraceGraphInputs`にはwaveform、9/15次元特徴、
edge index/type、observed mask、query index、coverage、依存範囲の`dependency_rounds`を渡す。
ラベルやarray_rowは含めない。

任意座標では`build_trace_graph_domain()`を使える。queryのarray_rowは省略可能で、混在する行対応には
`-1`を使える。`assemble_masked_trace_graph_inputs()`はplan、明示的な観測ID・物理波形、time grid、
固定前処理から直接tensorを作るため、caseや元ファイルの行番号を必要としない。
queryと観測の同一物理source/receiver pairを別IDで入力することは拒否する。

## 正確な近傍と依存範囲

`processing/trace_graph_neighbors.py`の`select_trace_graph_neighbors()`は、可視・許可domain・自己ID除外を
先に適用し、relation別radius内のtop-kを選ぶ。距離はfloat64、同距離はstable ID昇順で決定する。
`relation_scales_m`は`[4,2]`であり、特徴の位置・offset尺度とは独立する。

| relation ID | 距離の第1成分 | 距離の第2成分 |
|---|---|---|
| source = 0 | source差 / scales[0,0] | receiver差 / scales[0,1] |
| receiver = 1 | source差 / scales[1,0] | receiver差 / scales[1,1] |
| cmp = 2 | CMP差 / scales[2,0] | full offset vector差 / scales[2,1] |
| offset_azimuth = 3 | CMP差 / scales[3,0] | full offset vector差 / scales[3,1] |

距離Dは両成分の二乗ノルム和の平方根。既定値は`radius=1`、`neighbors_per_relation=8`で、物理尺度は
呼出側が必ず指定する。検索は候補をchunkで走査するexact searchであり、距離の一時配列は
O(chunk size + k)、返すedgeはO(destination数 × 4k)。全surveyのN×N配列は作らない。
chunk sizeは結果を変えない。実surveyでの速度・メモリ使用量は未測定である。

`processing/trace_graph_subgraphs.py`の`build_trace_graph_subgraph()`はqueryをdepth=0として、
depth<Lのdestinationだけを元の観測domainへ問い合わせる。L-hop葉のincoming edgeは追加しない。
`TraceGraphPlan`のnode順はstable ID順、`query_indices`は元のquery順を保持する。
`edge_index`は`[sender, destination]`で、すべてのsenderは可視観測である。
異なるrelationの同一ペアは保持し、対称化・radius拡大・fallbackは行わない。

coverageは`[N,4,2]`のdegree/kと最小Dであり、空relationと未展開葉では両方0。
planにはquery数、support数、typed edge数、unique pair数、最大depthの診断も含む。
`dependency_rounds`は構築時に要求したLを保持し、そのままモデル入力へ引き継ぐ。
探索が早く完了した場合も、実際の最大depthからLを求め直さない。
モデルは`dependency_rounds >= model.message_passing_rounds`を予測前に検証する。
呼出側は`model.message_passing_rounds`からLを導出し、query分割間で可視集合・前処理・検索条件を固定する。

## 波形モデルとrelation融合

`models/relational_trace_graph.py`の`RelationalTraceGraphMessageBlock`は、ノード内の時間処理、
destination×relation内のattention、共有value変換、channel gateを順に計算する。
すべてのメッセージは更新前の同じラウンド状態に依存する。正規化はノード内に限定する。

`RelationalTraceGraphInterpolator`は既存`TraceNodeEncoder`・`TraceNodeDecoder`を使い、
`forward(inputs)`からquery順の正規化波形`[Q,T]`と`has_observed_context[Q]`を返す。
既定はwidth=64、rounds=2、downsample factor=2、stem kernel=7、temporal kernel=5、
dilations=(1,2)、attention width=32、relation embedding dim=8。
右側をfactorの倍数へzero-padし、復元後に元Tへcropする。width=8かつT=1にも対応するため、
codec境界では最低2サンプルまでpadする。paddingは返す波形に含めない。

`relation_fusion="mean"`は利用可能なrelationだけの等重み、`"learned_gate"`は潜在特徴・relation message・
embedding・coverageによる学習重みである。gate最終層はzero initなので等重みから始まる。
空relationの重みは0、全relation空のqueryは予測0・context=Falseを返す。
decoderも既存のzero initを使うため、未学習モデルの直接予測は0から始まる。

`constructor_config()`は独立した純粋な構成値を返す。
`model(inputs, diagnostics=summary)`は通常と同じ戻り値を保ち、渡したdictへdetach済みの
query診断だけを記録する。途中の観測supportや未展開葉は集計対象に含めない。

| キー | shape | 内容 |
|---|---|---|
| `gate_sum` | `[rounds,4]` | contextのあるqueryのrelation別gate合計 |
| `available_query_count` | `[rounds,4]` | 各relationにincoming edgeがあるquery数 |
| `context_query_count` | `[rounds]` | いずれかのrelationにincoming edgeがある集計対象query数 |
| `no_context_query_count` | `[rounds]` | incoming edgeのないquery数 |

batchごとに各合計・件数を足し、`gate_sum / context_query_count[:, None]`で全体のgate平均を得る。
集計対象queryが0件の場合は合計も0となり、平均は計算しない。
message block単体で診断する場合は`diagnostic_query_indices`に検証済みのquery indexを明示する。
gate値だけから物理的重要度や因果関係を判断しない。

## 任意座標queryの例

```python
import numpy as np
import torch

from seis_interp.data.masked_trace_source import MaskedTraceSource
from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_geometry import compute_trace_graph_geometry
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_subgraphs import build_trace_graph_subgraph

time_s = np.arange(7, dtype=np.float64) * 0.01
waveforms = np.sin(2 * np.pi * 10 * time_s[None, :] + [[0.0], [0.4]]).astype(np.float32)
source = np.array([[0.0, 0.0], [1.0, 0.0]])
receiver = source + [0.0, -2.0]
training = build_trace_graph_domain(
    trace_ids=np.array([0, 1]),
    source_xy_m=source,
    receiver_xy_m=receiver,
    ffids=np.array([10, 11]),
    array_rows=np.array([0, 1]),
    observed_mask=np.ones(2, dtype=bool),
    time_s=time_s,
    pool="all_train_traces",
    inputs_lock={"partition": "train", "source": "analytic_toy"},
)
fixed = fit_trace_graph_preprocessing(
    training,
    waveforms,
    position_scale_m=10.0,
    offset_scale_m=2.0,
    azimuth_min_offset_m=0.1,
)
observed_geometry = compute_trace_graph_geometry(source, receiver, azimuth_min_offset_m=0.1)
query_geometry = compute_trace_graph_geometry(
    np.array([[0.5, 0.2]]),
    np.array([[0.5, -1.8]]),
    azimuth_min_offset_m=0.1,
)
model = RelationalTraceGraphInterpolator(relation_fusion="learned_gate").eval()
plan = build_trace_graph_subgraph(
    query_geometry,
    np.array([-1]),
    observed_geometry,
    training.trace_ids,
    training.observed_mask,
    rounds=model.message_passing_rounds,
    relation_scales_m=np.array([[1.0, 4.0], [4.0, 1.0], [1.0, 4.0], [4.0, 1.0]]),
)
inputs = MaskedTraceSource(training, fixed, waveforms).inputs(plan)
with torch.no_grad():
    normalized_prediction, has_observed_context = model(inputs)
physical_prediction = normalized_prediction * fixed.amplitude_scale
```

この例は任意座標への入出力を示す。未学習モデルの精度を実証するものではない。
学習episode、checkpoint保存、学習・推論CLI、benchmark出力保存はこのAPIの範囲に含めていない。
