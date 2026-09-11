# GNN training runtime ablation — 2026-09-11

## Environment and fixed conditions

- Commit: `5829a11482edd8adafd29b4be25f910fe5545425`（worktreeの実装変更を含む）。4 runの実行開始時の`src/seis_interp/`配下の全Python source hashは一致。
- GPU: NVIDIA H100 NVL、`cuda:1`、UUID `GPU-5343d08b-0556-2774-6d44-7e8dc8455690`。PyTorch `2.5.0a0+b465a5843b.nv24.09`、CUDA `12.6`。
- Source: `studies/study_037_c3_neural_mse_loss_ablation/smoke/gnn.yaml`。既存のformal config・runは編集せず、resolved内容をscratchへコピーした。
- 各200 updates、query batch 64、warm-upは最初の20 step。予測batch 256。OMP/MKL/OPENBLAS各1 thread。別processでA→B→C→Dを逐次実行。
- Model: width64、2 message-passing rounds、time downsample2、learned gate。初期化seed101、episode seed201。MSE、observed global RMS、AdamW、learning rate0.001、weight decay0、gradient clip1、inner mask0.8。
- Candidate K=2を固定。ユーザーの明示指定により、C/DはZIP例のF=3をF=1に置き換えた。edge seed4201。dtypeはH100の対応を確認してbf16に固定。
- 同一C3 random-80 case/volume/maskを使用。入力のexact lockは全runおよび再読み出しで一致。
- 実行前から両GPUに他processの負荷があった。各run前後のGPU状態を保存した。これは共有GPU環境での一回の測定であり、速度差を混雑から分離した因果的な効果量とはしない。

## Results

各値は全200 stepの集計。prep/optimはstepあたり秒、peakは学習終了直後・予測開始前のCUDA allocated/reserved GiB。機械可読JSONは丸めていない。

| Variant | Precision / F | updates/s | Training s | Prep s | Optim s | Mean / max nodes | Mean / max edges | Peak allocated / reserved GiB | Status / finite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | off / full K | 2.7048 | 73.9434 | 0.1036 | 0.2623 | 1651.3150 / 1788 | 3494.6000 / 3736 | 59.3027 / 59.3809 | success / yes |
| B | bf16 / full K | 6.1111 | 32.7274 | 0.0836 | 0.0766 | 1651.3150 / 1788 | 3494.6000 / 3736 | 3.2915 / 4.7793 | success / yes |
| C | bf16 / F=1 | 7.8134 | 25.5970 | 0.0796 | 0.0450 | 836.8600 / 907 | 1112.9900 / 1188 | 1.6126 / 3.0234 | success / yes |
| D | off / F=1 | 6.5180 | 30.6844 | 0.0799 | 0.0699 | 836.8600 / 907 | 1112.9900 / 1188 | 31.1088 / 31.1641 | success / yes |

| Variant | Mean step s | Median step s | Step21–200 mean s | Step21–200 median s | Initial loss | Final loss | Prediction coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.3659 | 0.3940 | 0.3386 | 0.3889 | 0.9702 | 0.7580 | 104710 / 104710 |
| B | 0.1602 | 0.1387 | 0.1409 | 0.1364 | 0.9702 | 0.7563 | 104710 / 104710 |
| C | 0.1246 | 0.1119 | 0.1100 | 0.0966 | 0.9702 | 0.8314 | 104710 / 104710 |
| D | 0.1499 | 0.1197 | 0.1332 | 0.1053 | 0.9702 | 0.8311 | 104710 / 104710 |

A/DのFP32 peakは大きいが、表にはそのまま示した。既存の`cudnn.benchmark=True`を全条件で保持しており、peakの区間には初回処理も含む。steady-stateのactivationだけのmemory値としては解釈しない。

## Correctness

- 全runが200 optimizer updates、12800 query presentationsで成功し、全stepのlossと保存model stateが有限。gradient clipのnon-finite error検査を有効のまま実行した。
- A/BおよびC/Dの各stepでepisode/query件数と実node/edge/depthが完全一致。sampling無効時のgraph-plan exact parity、および旧HEADとのquery IDs・loss・全model state・RNG一致は別途確認済み。
- 全予測のshapeは`[384, 16, 32, 8, 32]`、全要素有限。観測値のexact reinsertion、target ID順、coverage maskを保存配列と再読み出した入力から確認した。104710 targetをすべて予測し、境界targetも含む。
- 全runのprediction topology診断は完全一致。410 batches、2089784 typed/message edges、339231 processed support nodes。checkpointのgraph K=2を保持し、training-only samplingはgraph/model constructorに入っていない。
- AMP/fanoutによる実行errorはなかった。学習lossやtarget指標によるdtype・fanoutの選択は行っていない。

## Interpretation

Aではoptimization (0.2623 s)がpreparation (0.1036 s)より大きい。Bではoptimizationが0.0766 sへ下がり、Cでは0.0450 sとなった。C/Dで実際のnode数・edge数が減ることも実測で確認した。
Cではpreparationが0.0796 sでoptimizationより大きい。B→Cでpreparationの変化は小さいため、fanoutによるclosure縮小がbatch準備全体を同じ割合で短縮するとはいえない。query検索・geometry/plan/input assemblyの内訳が次の計測候補であり、本検証では新しい最適化やdisk cacheは追加していない。
geometry-only indexのrun内再利用は省略時の数値契約を保持するruntime変更として使用できる。AMPとfanoutは明示設定のまま提供し、とくにfanoutは学習グラフを変えるため精度の採否は別のvalidationで判断する。この測定の最終loss・test target指標をモデル選択や論文性能値として用いない。

## Artifacts

- Scratch root: [20260911T110728Z_5829a11482ed](../runs/gnn_training_speed_scratch/20260911T110728Z_5829a11482ed)
- Fixed configs, commands, environment: [execution_plan.json](../runs/gnn_training_speed_scratch/20260911T110728Z_5829a11482ed/execution_plan.json)
- Full precision measurements and checks: [comparison.json](../runs/gnn_training_speed_scratch/20260911T110728Z_5829a11482ed/comparison.json)
- Postprocessing and artifact assertions: [summarize.py](../runs/gnn_training_speed_scratch/20260911T110728Z_5829a11482ed/summarize.py)
- 各A/B/C/D directoryにresolved config・input lock・全history・metadata・checkpoint・prediction配列を保存。各`*.execution.json`にprocessの開始/終了時刻・return code・GPU状態・source hashを保存。
