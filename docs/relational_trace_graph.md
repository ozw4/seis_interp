# Grid-free relational trace graph

一トレースを一ノードとして、任意の絶対source/receiver座標から欠損波形を直接予測するPython APIとCLI。
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

## 学習episodeと固定validation

`training/trace_graph_episodes.py`の`TraceGraphEpisodeGenerator`は、固定O0からepisode全体の
hidden集合Hと可視集合Oを決め、その後でHをquery minibatchへ分ける。
`random_trace`はトレースを、`random_whole_ffid`はFFIDを選び、そのFFIDのO0内の全行を隠す。
欠損数は`round(unit_count * missing_fraction)`で決まり、観測・hiddenの両方が残らない設定は拒否する。
kindは指定確率、missing fractionは指定リストから等確率で選ぶ。
hidden IDを昇順に並べてからlocal RNGでshuffleし、一巡するまでOを変更しない。
`read_trace_graph_training_labels()`だけが、そのbatchのHに属する訓練ラベルを読み、固定RMSで正規化する。

`training/relational_trace_graph_trainer.py`の`train_relational_trace_graph()`は、初期化済みモデル、
train/validation domain、固定前処理、`TraceGraphSettings`、episode/optimizer設定を受け取る。
AdamWでhidden queryのMSEを最適化し、paddingを分母へ入れない。
historyはerror energyとsample数から集計し、最後の小batchとcontextなしqueryも含める。
graphのroundsは常にモデルから導出する。

validationは固定caseの観測だけで予測を完了してから物理振幅のtarget SSEを採点する。
SSE最小のstepをbestとし、同値では早いstepを保持する。最終stepでもvalidationを行う。
戻り値は独立したCPUのbest/final state、選択指標、加重history、query/context件数、episode完了情報を持つ。
任意の`on_best_update(state_dict, step, metrics)`を指定すると、best更新直後に独立したCPU snapshotと
選択指標を通知する。受け取った値は変更せず保存に用い、通知先の例外は呼出元へ伝播する。
`max_steps`でepisodeの途中に停止した場合は`final_episode_interrupted`へ記録する。
trainerはtest domainを受け入れない。

## checkpointと凍結予測

`training/relational_trace_graph_checkpoints.py`の`save_relational_trace_graph_checkpoint()`は、
constructor configとstate dictを別々に受け取り、best状態を後続の更新から独立したCPU snapshotとして保存する。
graph尺度/k/radius、relation・node/edge特徴の順序、固定前処理とfit domain、time_s/T/factor、
人工mask設定、訓練provenance・seed、`best_validation`または`final`のrole、stepと選択指標も保存する。
module objectやoptimizer/resume状態は保存しない。
保存は同じdirectoryの一時ファイルへ完了してから置き換え、途中の書き込み失敗で直前のcheckpointを失わない。
`load_relational_trace_graph_checkpoint()`は保存構成からモデルを再構築し、stateをstrict loadする。
異なるモデル名、特徴順序、無効な尺度、time gridや構成の不整合はエラーとなる。

`training/relational_trace_graph_prediction.py`の`predict_relational_trace_graph(model, domain,
preprocessing, *, graph_settings, ...)`は、通常はdomainのquery全件を予測する。
`query_trace_ids`で対象と順序を指定でき、任意座標にはさらに`query_source_xy_m`と
`query_receiver_xy_m`を渡す。`observed_waveforms`をdomainのobserved順で渡せば、観測側のarray_rowも不要になる。
ファイル入力では各batchのsupportだけを読み、元の観測domain・前処理・探索条件を固定する。
`model.eval()`と`inference_mode`内で実行し、終了時に呼出前のtraining/evalモードへ戻す。

戻り値`TraceGraphPrediction`は、物理振幅`prediction[Q,T]`、query ID・source/receiver座標、
`has_observed_context`、軽量な診断を持つ。gateの合計・query件数はbatch間で累積する。
`processed_support_node_count`等のグラフ件数はbatch処理の延べ数であり、query診断とは区別する。
異なるtime gridへのresamplingは行わない。

`evaluation/trace_graph_metrics.py`の`evaluate_trace_graph_prediction()`は、完成した物理予測を受けて
mapped evaluation targetだけを読む。float64のreference/error energyからglobal S/N、RMSE、relative L2を
計算する。contextあり・なしの指標と件数も返し、contextなしqueryを主指標から除外しない。
S/Nが非有限になる場合はraw energy、`snr_status`、nullableな`snr_db`でstrict JSONへ記録する。

## 任意座標queryの例

```python
from dataclasses import replace

import numpy as np

from seis_interp.data.trace_graph_domain import build_trace_graph_domain
from seis_interp.models.relational_trace_graph import RelationalTraceGraphInterpolator
from seis_interp.processing.trace_graph_preprocessing import fit_trace_graph_preprocessing
from seis_interp.processing.trace_graph_settings import TraceGraphSettings
from seis_interp.training.relational_trace_graph_prediction import predict_relational_trace_graph

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
model = RelationalTraceGraphInterpolator(width=8, relation_fusion="learned_gate")
settings = TraceGraphSettings(
    relation_scales_m=((1.0, 4.0), (4.0, 1.0), (1.0, 4.0), (4.0, 1.0)),
)
result = predict_relational_trace_graph(
    model,
    replace(training, array_rows=None),
    fixed,
    graph_settings=settings,
    observed_waveforms=waveforms,
    query_trace_ids=np.array([-1]),
    query_source_xy_m=np.array([[0.5, 0.2]]),
    query_receiver_xy_m=np.array([[0.5, -1.8]]),
    query_batch_size=1,
)
physical_prediction = result.prediction  # float32 [1, 7]
has_observed_context = result.has_observed_context  # [True]
```

この例は任意座標への入出力を示す。未学習モデルの精度を実証するものではない。
保存モデルでは`load_relational_trace_graph_checkpoint()`の戻り値にある
`model`・`preprocessing`・`graph_settings`を、そのままこの予測関数へ渡す。

## CLIの最小設定と実行

以下は小規模synthetic artifact向けのtoy例で、研究上の推奨パラメータではない。
`train.yaml`へ保存し、dataset ID・partition seed・時間範囲を既存artifactへ合わせる。
訓練とvalidationは同じinterim/partition artifact内のdisjointな行を使う。
独立した設定ファイルとして使い、別手法の未使用sectionを継承しない。

```yaml
project: {random_seed: 42}
data: {dataset_id: synthetic_c3_ccnet5d}
model:
  name: relational_trace_graph
  width: 8
  message_passing_rounds: 2
  time_downsample_factor: 2
  stem_kernel_size: 3
  temporal_kernel_size: 3
  temporal_dilations: [1, 2]
  attention_width: 4
  relation_embedding_dim: 3
  relation_fusion: learned_gate
graph:
  neighbors_per_relation: 2
  max_normalized_distance: 1.0
  candidate_chunk_size: 16
  relations:
    source: {source_scale_m: 2000.0, receiver_scale_m: 5000.0}
    receiver: {source_scale_m: 5000.0, receiver_scale_m: 2000.0}
    cmp: {midpoint_scale_m: 2000.0, offset_vector_scale_m: 5000.0}
    offset_azimuth: {midpoint_scale_m: 5000.0, offset_vector_scale_m: 2000.0}
geometry_features:
  position_scale_m: 1000.0
  offset_scale_m: 1000.0
  azimuth_min_offset_m: 0.1
training_data: {pool: all_train_traces, time_samples: [1, 4]}
training_mask:
  kinds: [random_trace, random_whole_ffid]
  kind_probabilities: [0.5, 0.5]
  missing_fractions: [0.5]
training:
  device: cpu
  random_seed: 7
  loss: masked_mse
  max_steps: 2
  query_batch_size: 4
  validation_interval: 2
  learning_rate: 0.01
  weight_decay: 0.0
  gradient_clip_norm: 1.0
evaluation:
  primary_metric: physical_amplitude_global_snr_db
  domain: evaluation_target
  query_batch_size: 4
```

`project.random_seed`は既存partitionの条件、`training.random_seed`はモデル初期化と人工episodeの条件である。
episodeのlocal RNGはモデル初期化やvalidation頻度から独立する。
validation/test maskのseedは各caseから取得し、訓練seedとの一致は要求しない。
`time_samples: [1, 4]`は元配列のサンプル1・2・3を意味し、選択した実際のtime_sをcheckpointへ保存する。
volume指定時も時間範囲の一致を要求する。

既存artifactの場所を以下のpathへ置き換えて実行する。

```bash
rtg_run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
seis-interp train relational-trace-graph \
  --config train.yaml \
  --interim data/interim/toy --processed data/processed/toy \
  --validation-mask data/processed/toy/masks/validation \
  --validation-case data/processed/toy/cases/validation \
  --output "runs/trace-graph-toy/${rtg_run_id}-train" --device cpu --json
```

`all_train_traces`ではtrain mask/caseは不要であり、追加訓練アクセスをrunへ明示する。
`training_data.pool: mask_observed`に変更する場合は、同時に
`--train-mask data/processed/toy/masks/train --train-case data/processed/toy/cases/train`を指定する。
`--train-volume`はtrain mask/caseが両方ある場合に使え、O0の観測をそのcropへ制限する。
`--validation-volume`はvalidationの観測・queryをそのvolumeへ制限する。

凍結推論用の`predict.yaml`は、checkpointの構成を再定義しない。
次の`benchmark_case.id`は既存test caseのIDへ合わせる。

```yaml
project: {random_seed: 42}
data: {dataset_id: synthetic_c3_ccnet5d}
benchmark_case: {id: test}
prediction: {device: cpu, query_batch_size: 3}
evaluation:
  primary_metric: physical_amplitude_global_snr_db
  domain: evaluation_target
```

```bash
seis-interp interpolate relational-trace-graph \
  --checkpoint "runs/trace-graph-toy/${rtg_run_id}-train/artifacts/best.pt" \
  --config predict.yaml \
  --interim data/interim/toy --processed data/processed/toy \
  --mask data/processed/toy/masks/test --case data/processed/toy/cases/test \
  --output "runs/trace-graph-toy/${rtg_run_id}-test" --device cpu --json
```

`--volume`を追加すると、同じcaseにbindingされた既存volumeを選択する。
モデル・graph・前処理・time gridはcheckpointを正本とし、推論configでの上書きやtime resamplingは拒否する。
benchmark入力はcheckpointの訓練元と同じinterim/processed hashを要求する。
`--json`のstdoutはstrict JSON、進捗はstderrへ出力する。

## run出力

出力先は既存pathの再使用を拒否する。各runへ`config.resolved.yaml`、`inputs.lock.json`、
`run.json`、`metrics.json`を保存し、実際の入力hash、seedの役割、訓練pool、time、モデル・graph・固定尺度、
method variant、device、件数とlayoutを記録する。
訓練runの`artifacts/best.pt`はvalidation SSE最小の状態、`artifacts/final.pt`は最終stepの状態である。
訓練runに保存するpredictionはbestを再ロードしたvalidation予測であり、testの採点は凍結推論runで行う。

訓練では入力・設定・人工maskの成立条件と固定前処理を検証してから出力先を作り、
学習前にresolved config、input lock、開始時の`run.json`と`metrics.json`を保存する。
`run.json`は`status: running`、`phase: training`で始まり、best更新ごとにpipelineが`best.pt`、
bestのstep・指標、記録時点の進捗を保存する。最初のvalidation前はbest関連の指標・checkpointは存在しない。
configとinput lockは開始後に書き換えない。

trainer正常終了後、`final.pt`と全学習指標を先に保存し、`phase: validation_prediction`でbestの予測を出力する。
予測の保存まで完了すると`status: success`、`phase: complete`と終了時刻を確定する。
例外時は`failed`、KeyboardInterrupt時は`interrupted`として、失敗したphase・例外・終了時刻を記録する。
強制終了など終了処理を実行できない場合は、最後に保存された`running`の記録とcheckpointが残る。
更新するJSONは各ファイルを一時ファイルから置き換える。checkpointとJSON全体を一括更新する契約ではないため、
更新途中で終了した場合、そのcheckpointのstep・選択指標はcheckpoint自身の内容で確認する。

| layout | `artifacts/prediction.npy` | 出力対応 |
|---|---|---|
| `native_trace_list` | query順の物理振幅`[Q,T]` | `artifacts/query_index.parquet`にtrace ID、座標、context flag、存在する場合のarray_row |
| `dense_volume` | `(time, source_line, shot_in_line, rx, ry)` | 既存volume index順へ復元し、観測は元の物理振幅をexact copy。query対応表も保存 |

モデルはこの出力layoutを認識しない。予測・行対応保存は`data/trace_graph_prediction_store.py`、
labelを使う採点は`evaluation/trace_graph_metrics.py`、runの接続は二つの専用pipelineが担当する。
追加診断・IDW、比較モデルとablation、study preflightや本実験を扱うTask14以降は未実装である。
