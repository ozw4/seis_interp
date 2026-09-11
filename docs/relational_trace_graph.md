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

`training_data.max_abs_amplitude`を指定すると、通常学習と訓練preflightは
許可されたtrain行・時間内の物理振幅をRMS集計前に再検査する。正の有限値を指定し、
絶対値がその値を超えたsampleはtrace ID・array row・元のsample位置とともにエラーとなる。
この検査はクリッピングや行の自動除外を行わない。QCによる除外は別のprepared partitionへ
明示的に記録する。未指定時のRMS計算とcheckpoint形式は従来どおりである。

`processing/normalization_qc.py`の`summarize_amplitude_normalization()`は、明示した行と
時間だけについて、固定尺度でfloat32へ正規化した後の非有限値と二乗エネルギーの消失を
調べる。正当なゼロトレースは消失として数えず、尺度の再fitやデータの変更も行わない。

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
chunk sizeは結果を変えない。

`TraceGraphSettings.neighbor_search`（CLIでは`graph.neighbor_search`）の既定は`brute_force`。
任意の`exact_index`は`FixedTraceGraphNeighborIndex`を使い、座標軸のsorted indexと保守的な範囲で
候補を絞ってから、同じfloat64距離・radius判定・距離/ID順で選ぶ。可視・許可domain、自己ID除外、
relation別top-kは共通であり、近似探索やquery batch内への候補制限は行わない。

`processing/trace_graph_subgraphs.py`の`build_trace_graph_subgraph()`はqueryをdepth=0として、
depth<Lのdestinationだけを元の観測domainへ問い合わせる。L-hop葉のincoming edgeは追加しない。
`TraceGraphPlan`のnode順はstable ID順、`query_indices`は元のquery順を保持する。
`edge_index`は`[sender, destination]`で、すべてのsenderは可視観測である。
異なるrelationの同一ペアは保持し、対称化・radius拡大・fallbackは行わない。

`exact_index`の`FixedTraceGraphSubgraphBuilder`はgeometry・ID・可視/許可mask・検索設定を所有し、
固定domain内でindexとID対応を再利用する。可視senderのincoming edgesだけをIDごとに遅延cacheし、
初期queryの検索は毎回行う。各hopの検索domainは元の可視候補全体のままである。
cacheのedge数は可視・許可sender数 × 有効relation数 × kが上限で、返すplanの変更はcacheへ伝わらない。
`observed_neighbor_cache_info()`で件数と数値配列のbytesを取得でき、bytesにはPython容器の領域を含めない。
trainerはepisodeごとに新しいbuilderを作り、凍結予測は一つの固定観測domainでquery batch間に再利用する。

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

`model.amplitude_mode`の既定は`train_global_rms`で、固定global RMSによる既存の入力・出力単位を保つ。
任意の`observed_trace_rms`は、モデル内で可視波形の選択時間範囲ごとのtrace RMSを求め、
非ゼロの各traceをunit RMSにしてencoderへ渡す。正当なゼロtraceは波形・RMSとも0のまま扱う。
queryの倍率は、直接incoming edgesにあるuniqueな観測senderのRMSを
`1 / max(D0, 1e-6)²`の正規化重みで平均する。同じpairのrelation重複は一度だけ数え、
2-hop supportへのedgeや他query専用の近傍は混ぜない。

このmodeは`graph.common_distance_scales_m: [midpoint_scale_m, offset_vector_scale_m]`を必須とし、
`D0 = sqrt(||dm||² / midpoint_scale_m² + ||do||² / offset_vector_scale_m²)`を用いる。
倍率はdecoder出力へ一度だけ掛ける。入力RMSと倍率は固定global RMSで割った単位なので、
予測関数が最後にglobal RMSを掛けると物理振幅へ戻る。この倍率には訓練・予測ともqueryの教師RMSを使わない。
既定lossは固定global RMSで正規化した物理波形のMSEであり、下記の相対MSEは損失の重みだけを変更する。

`model.max_edge_time_shift_samples`の既定は0で、非ゼロの整数はrelational model内で
学習するedgeごとの時間ずれの最大値をraw sample単位で指定する。sender−destinationの
source/receiver差4成分（既存の位置尺度で正規化）と、全round・relationで共有する
ゼロ初期化の4係数から `lag = (max / factor) × tanh(delta @ weights)` を求める。
外部推定係数やquery教師をforwardへ渡さず、選択した訓練lossの勾配で係数を学習する。

各edgeのvalue系列を `value[f + lag]` で線形補間し、既存attentionとgammaで集約する。
正lagは早い出力時刻への移動、範囲外はゼロで、末尾から先頭へのwrapは行わない。
小数lagの線形補間は高周波を減衰させ、半frameでは振幅応答が`abs(cos(omega/2))`になる。
振幅を保存する厳密なdelayとは異なる。`observed_trace_rms`と併用でき、
非ゼロ設定だけが4係数とconstructor設定をcheckpointへ追加する。

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

## 比較モデルとablation

同じ`RelationalTraceGraphInterpolator`、codec、trainer、checkpointで以下を指定できる。

| 設定 | 集約・接続 |
|---|---|
| `model.method_variant: relational`（既定） | 四relationを保持し、`relation_fusion: mean`または`learned_gate`で融合 |
| `model.method_variant: plain_gcn_row_normalized` | unique incoming観測とlocal selfの和を`1 + in_degree`で割り、共有linear変換と残差更新 |
| `model.method_variant: untyped_edge_conditioned` | unique incoming pairへ一組のgeometry attention/gammaを適用。relation embedding/gateは使わない |
| `graph.topology: single_4d` | midpoint/full-offsetの単一距離によるexact top-k。untypedモデルとの組合せに限定 |
| `graph.excluded_relation: source / receiver / cmp / offset_azimuth` | multi_relationの一種類だけを除外し、残るrelation IDは保持 |
| `model.explicit_azimuth_features: false` | nodeのsin/cos/valid、edgeのcos差/sin差/validを0にする |

比較モデルの`relation_fusion`は`mean`を指定する。GCNはこのmasked directed graphのrow-normalized適用であり、
対称GCNの完全再現ではない。local selfはqueryから他ノードへの送信を追加しない。
azimuth列を消す場合もoffset vector/長さとtopologyは保たれるため、「方位情報なし」という比較ではない。
比較モデルのquery診断は`gate_relation_names: [untyped]`の1列で、関係選択を学習するgateではない。

untypedモデルは`graph.common_distance_scales_m: [midpoint_scale_m, offset_vector_scale_m]`を必須とし、
距離列を`D0 = sqrt(||dm||² / lg² + ||do||² / lo²)`へ統一する。relation別距離や重複回数を特徴にしない。
`single_4d_neighbors`の既定は32であり、multi_relationの既定4×8と比較できる。
radiusやrelation重複によって実際のunique pair数は変わる。runにはparameter数、typed/unique edge数と実測時間を記録し、
同情報量・同計算量を仮定しない。single_4dの内部edge typeは0で、名称は`untyped`である。
これらのgraph/model/特徴設定はcheckpointに固定し、凍結推論で変更しない。

## 学習episodeと固定validation

`training/trace_graph_episodes.py`の`TraceGraphEpisodeGenerator`は、固定O0からepisode全体の
hidden集合Hと可視集合Oを決め、その後でHをquery minibatchへ分ける。
`random_trace`はトレースを、`random_whole_ffid`はFFIDを選び、そのFFIDのO0内の全行を隠す。
欠損数は`round(unit_count * missing_fraction)`で決まり、観測・hiddenの両方が残らない設定は拒否する。
kindは指定確率、missing fractionは指定リストから等確率で選ぶ。
hidden IDを昇順に並べてからlocal RNGでshuffleし、一巡するまでOを変更しない。
訓練ラベルは`read_trace_graph_training_labels()`が、そのbatchのHに属する行だけを読み、固定RMSで正規化する。
`exact_index`では`TraceGraphEpisodeLabelReader`がepisodeの行対応・hidden/visible metadataを所有して再利用し、
同じ許可判定・行順・正規化でラベルを読む。波形の読み込みはbatch要求時に行い、ラベルをモデル入力へ渡さない。
builderとreaderはepisodeの切り替え時に作り直し、mask生成やquery順のRNGは変更しない。

`training/relational_trace_graph_trainer.py`の`train_relational_trace_graph()`は、初期化済みモデル、
train/validation domain、固定前処理、`TraceGraphSettings`、episode/optimizer設定を受け取る。
AdamWでhidden queryの選択したlossを最適化し、paddingを分母へ入れない。
historyはerror energyとsample数から集計し、最後の小batchとcontextなしqueryも含める。
graphのroundsは常にモデルから導出する。

`training.cudnn_benchmark`は任意のboolで、既定は`true`。
`false`を指定した場合だけ、モデル初期化のseed設定後にcuDNNのbenchmark探索を無効にし、
trainerと訓練preflightへ同じ設定を渡す。省略時と`true`指定時は従来の設定・RNGを保ち、
既定keyをtrainer引数やcheckpointへ追加しない。native runの`resources.cudnn_benchmark`は
実際に有効な値を記録する。アルゴリズム選択に伴う速度・一時メモリ・丸め差は実測で確認する。

`training.loss`の既定は`masked_mse`で、global正規化単位のMSEを使う。
任意の`masked_trace_relative_mse`は、各訓練教師traceのRMSを`r_q`として
`mean(((prediction_q - teacher_q) / r_q) ** 2)`を最適化する。
RMSはfloat64で計算してdetachし、lossの重み付けだけに使う。モデル入力や予測倍率には渡さず、
validation/testから重みをfitしない。ゼロRMSだけはglobal正規化単位の除数1を使い、
小さい正のRMSにfloorやクリッピングを加えない。したがって小振幅traceの逆重みは大きくなり得る。
このmodeだけhistoryに`batch_objective_loss`とsample数で加重した`objective_loss`を追加する。
既存の`batch_loss`・`train_loss`・`normalized_error_energy`は物理換算可能なglobal正規化MSEのままで、
preflightも同じlossを使い、目的関数とそのMSEを分けて記録する。

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
非既定の`amplitude_mode`と`neighbor_search`も構成値に保存し、復元時にD0尺度との組合せを検証する。
既定値は新しい構成keyを追加せず、従来のcheckpointは既定modeとして復元する。
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

## PoC CLI

`interpolate relational-trace-graph`は固定C3 random-80 volume内の観測集合Oだけで学習し、
最終stateから欠損集合T全体を予測する。`--volume`は必須で、外部checkpointは受け取らない。
上記のpartition学習Python APIではなく、
`training/relational_trace_graph_poc_trainer.py`のfixed-step trainerを使用する。

設定には`project`、`data`、`benchmark_case`、`benchmark_volume`、`interpolation_mask`、
`model`、`graph`、`geometry_features`、`training`、`prediction`、`evaluation`を指定する。
datasetは`seg_c3_na`、volume selectionは全5軸のresolved範囲を宣言する。
`project.random_seed`は外側mask、`training.random_seed`はモデル初期化とepisodeのlocal RNGに使用する。
modelはrelational・global RMS modeとし、次のtraining項目をすべて指定する。

```yaml
training:
  device: cuda
  random_seed: 7
  optimizer: adamw
  loss: masked_trace_relative_mse
  amplitude_scaling: observed_volume_global_rms
  max_steps: 5000
  query_batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  inner_mask_fraction: 0.5
  report_interval: 100
prediction: {query_batch_size: 64}
evaluation:
  primary_metric: physical_amplitude_global_snr_db
  domain: evaluation_target
```

数値は設定形式の例であり、精度や実行予算の推奨値ではない。
Global RMSはO全体から一度だけ計算し、座標原点はD全体のmidpoint bounding box中心に固定する。
各episodeではOの一部Hを全trace単位で隠し、context readerはO−Hの振幅だけを保持する。
Hだけに共通lossを適用し、no-context queryも除外しない。validationやbest選択は行わない。

```bash
seis-interp interpolate relational-trace-graph \
  --config poc.yaml \
  --interim data/interim/c3 --processed data/processed/c3 \
  --mask data/processed/c3/masks/random80 \
  --case data/processed/c3/cases/random80 \
  --volume data/processed/c3/volumes/random80 \
  --output runs/gnn-poc/run-id --device cuda --json
```

## PoC run出力

既存の出力directoryは再使用しない。`artifacts/final.pt`には最終重み、constructor、
graph設定、共通RMS、固定座標bounds、inner mask率、seed、完了step、inputs lockを保存する。
推論にはO全体をcontextとして渡し、TのIDが重複・欠落なく一致することをscatter前に確認する。
物理振幅への復元は予測関数内で一度だけ行い、Oを再挿入したdense volumeを共通C3 evaluatorで採点する。

`artifacts/prediction.npy`、`target_coverage.npy`、`query_trace_ids.npy`と、
`config.resolved.yaml`、`inputs.lock.json`、`metrics.json`、`run.json`を保存する。
metadataは学習step・episode数・loss・query/no-context件数、parameter数、実行時間と取得可能なmemoryを含む。
推論・評価の失敗はrun全体の失敗として記録し、推論前に保存したfinal checkpointを保持する。
`--json`のstdoutはstrict JSON、進捗はstderrへ出力する。

## 診断・独立baseline・preflight

`processing/trace_graph_diagnostics.py`の`summarize_trace_graph_queries(plan)`は完全に展開したqueryだけの
relation degree、空率、最小距離、source/receiver/CMPのx/y幅、relation間Jaccard、typed/unique edge数を返す。
未展開葉を空近傍へ数えず、両集合が空のJaccardは未定義と件数で記録する。

`processing/trace_graph_idw.py`の`predict_trace_graph_idw()`はqueryの直接観測近傍をunique pairへまとめ、
`1 / max(D0, 1e-6)²`を正規化して元の物理波形を加重平均する。contextなしは0である。
`evaluation/trace_graph_diagnostic_metrics.py`の`evaluate_trace_graph_baselines()`はzeroとIDWを、
学習モデルと同じquery/domain/timeの物理targetだけで採点する。IDWは直接予測モデルのskip connectionには入らない。
IDWのD0尺度は明示引数、または固定前処理のposition/offset尺度を使い、結果へ記録する。

train/frozen設定には次の任意sectionを追加できる。境界は実行前に固定し、test性能で調整しない。

```yaml
diagnostics:
  time_s: [0.5, 1.0, 1.5]
  offset_m: [1000.0, 2000.0, 3000.0]
  azimuth_deg: [90.0, 180.0, 270.0]
```

`TraceGraphDiagnosticBands`はこの切断点を保持する。各軸の範囲外も端の帯へ含め、azimuth未定義は別集計する。
帯ごとにsample数とreference/error energy、offset/azimuth帯にはquery数も記録し、主指標の評価対象は変更しない。
trainerの`diagnostic_bands`指定時は学習batchとvalidationに帯別集計を残し、runの保存時にzero/IDW採点も加える。
runのfrozen predictionは`measure_resources=True`でgraph構築・観測入力読み込み・forwardの時間、
処理したsupport/edge数、利用可能なmemory計測値を記録する。Python APIでは計測を省略できる。
GPU未使用時のCUDA memoryは未測定であり、0の実測値として扱わない。

`training/trace_graph_preflight.py`の`run_trace_graph_preflight()`は明示した少数queryを元の観測domain上で調べる。
実測時間やmemoryが指定上限を超えた場合はblockerとして返す。radius拡張、近似探索、support切り捨てや本学習開始は行わない。
artifact検証を含む実行例と固定比較条件は[study_026](../studies/study_026_grid_free_multi_relation_gnn/README.md)を参照する。
CPUの解析的toyは任意座標で波形を再評価する入出力・学習試験であり、数値伝播やfield dataではない。
固定C3 validationでの現行採用条件は[Study032](../studies/study_032_c3_proposed_gnn_10db/README.md)に記載する。
QCで437本を除外し、observed trace RMSと相対MSEを用いた最終5,000更新モデルは、固定shearなしで
全58,999 targetの物理SNR 11.9354 dBを記録し、保存予測再採点と固定CPU復元監査を通過した。
複数seedの本実験とwide-azimuth/fieldへの汎化は未検証である。
