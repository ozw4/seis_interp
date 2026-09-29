# Decisions

## 2026-09-27 — 公平グリッド探索を事前登録する

**Status:** active

**Decision:**
GeRSI（ReGSI real/grid）・NeRSI・CCNet5D・POCS・DRRを、全手法で同じ段階・同じ選択規則・同じ検証集合Vで調整し、
新しいマスクBのtestで1回だけ評価する。探索グリッドは [grid.yaml](grid.yaml) に、計画書（公平比較グリッドサーチ最終案）の§3を転記した。

study_059から変えない条件:

- 学習率0.001一定、1更新64 lossトレース、20,000更新
- パラメータ数363,269±5%
- 500 m/sの幾何整列キャンバス、`trace_oracle` のRMS振幅

主testはマスクB（mask seed 600701）である。
マスクA（study_059、seed 510101）は、各手法の中心設定をそのtestで選んできた集合である。この研究ではマスクAの副次評価を行わない（下の「計画書からの逸脱」）。
Bのtestの約70%（34,406本中24,036本）はAのtestと同じトレースなので、同じsurveyを使う限りデータ水準のtuningの影響は除けない。

検証集合Vは、Bの観測からSHA256順位で10%を選んだ1,474本（clean 1,471本）で、全手法で共通である。
探索時の学習観測は残りの13,271本で、global RMSもその13,271本から計算し直した。
Vの実RMSは、testと同じく振幅の復元にだけ使う。

**計画書の解釈と補完:**
計画書に数値や分岐が書かれていない箇所は、次のとおり決めた。grid.yamlでは `fill:` の注記で示す。

- **Eの中点の軸:** 中点は軸1で試し、その中点が定義されていない場合は軸2で試す。CCNetのE行が「[4,8,4,8]@0.35 または d3」と書いているのに合わせた。
- **Eの最終分岐:** どの分岐も当てはまらない場合は、wdを1段上げる。こうすると全手法でEが必ず1 runになり、探索量が揃う。
- **ReGSIのtopology軸:** 順序のない軸として扱う。延長も中点も定義しない。
- **ReGSIの内部マスクの中点:** 名目値0.075と0.175を、同じ丸め規則 `round(64m/N, 5)` で定めた。探索時→再学習時で0.07716→0.07379（隠し本数1,024→1,088）、0.17361→0.17362（2,304→2,560）となる。
- **CCNetのpatch軸の順序:** 1パッチあたりの可視観測（入力密度 1−f）の小さい順に、[4,8,4,8]@0.4、@0.3、[6,6,8,6]@0.2と並べる。計画書の延長値（上端 [6,8,8,8]@0.1）と中点（@0.3と@0.4の間の@0.35）はこの順序と整合する。下端の延長と、@0.3と[6,6,8,6]@0.2の中点は定義しない。CCNetのdilationも、下端の延長と、d1とd2の中点は定義しない。
- **CCNetのwd延長:** 計画書に値がない。wd校正の分岐で必要になるため、ReGSIの0.2→0.5と同じ2.5倍として5e-4とした。
- **NeRSIの中点:** 算術中点とした（midpoint 1.06 / 1.085、azimuth 1.075 / 1.125）。
- **wd校正の「最終train loss」:** 最後の1,000更新の平均学習loss（`final_training_loss_window_mean`）を使う。比較するのは、S3の土台にしたrunの、wdの最小水準と最大水準のrunである。
- **Eの土台と軸上の位置:**
  - 軸1・軸2上の位置はS1の勝者で決める。軸1の近傍の比較には、軸2を勝者の値に固定したS1のrunを使う（軸2も同様）。
  - Eの土台は、S3までで最良の (run, EMA) のrunとする。
  - S2でその軸の値が変わっている場合、その軸にはEを適用しない。
- **POCSの窓の重なり:** 時間方向は窓の半分とする。空間方向は [1,8,1,8] に [0,6,0,6]、[2,8,2,8] に [1,6,1,6] とし、study_057の探索に揃えた。DRRの空間窓は [1,8,1,8]、重なりは [0,6,0,6] に固定する。
- **ReGSIのS2でのwidth:** S1の勝者が (k8, r1, W152) の場合、Fourier 0/8ではwidthを160にする（366,947 / 377,187）。W152のままではパラメータ数が帯の外に出る。
- **失敗run（計画書§1原則5）:** grid.yamlの `selection.diverged_runs` で定める。
  - tuningの `failure_reason` が `non-?finite` に一致する失敗（学習lossの非有限、ReGSIの勾配ノルムの非有限、非有限の予測）を「発散run」とする。
  - 発散runは終了したrunとして予算の16 runに数える。同じyamlを再実行しない（再実行があれば次の段階を生成しない）。
  - 選択規則では全EMAで +inf として扱い、完了したどのrunよりも後に並ぶ。S1〜E・Rの勝者、Rの上位2 run、3 seed平均の最小には選ばれない。
  - wd校正の比較で、wdの最小・最大水準のどちらかが発散runなら、Eの分岐1（wdの校正）は当てはまらないものとする。
  - 表と探索量の曲線（Dodge）には、全EMAでスコア1.0（定数0予測のスコア）として残す。各段階の `stage.json` の `rule_inputs` には `status: diverged` と記録する。
  - それ以外の失敗（メモリ不足、中断）は、同じyamlで再実行する。
- **最終runのEMA:** 最終runはRで選んだEMAだけを主checkpointとして予測する（`training.ema_eval_decays` を外す）。B testを見た後に別のEMAを選べないようにするためである。
- **重みの保存:** 探索runでは、主EMA以外の各shadowの最終重み（`<candidate>/ema_<decay>/final.pt`）と、全EMAの10,000更新時点の重み（`<candidate>/update_10000/ema_<decay>.pt`、grid.yamlの `ema_snapshot_updates`）を保存し、tuning manifestでhashを記録する。学習の数値は変わらない。
- **最終評価:** bootstrapは10,000回、RNG seed 600703とし、grid.yamlの `evaluation.bootstrap` に固定した。`scripts/forge_fair_grid_final_evaluation.py` は次を満たさない入力を、test波形を読む前に拒否する。
  - bootstrapの設定がgrid.yamlと一致する。
  - 各手法の `fair_grid/<method>/final/stage.json` が現在のgrid.yamlのsha256と最終準備runを記録している。
  - そこに並ぶ最終runが1本ずつ、すべて指定されている。各runは、記録されたyamlと同じ `search.yaml` を持ち、`purpose: final_evaluation`・`evaluate: false` で、metricsを持たず、選んだEMAを主checkpointとしている。
  - 探索量の曲線は、各 `final/stage.json` の `rule_inputs` のS1〜E（POCS・DRRはS1）から作る。tuning manifestのsha256も照合する。
- **最終testの保護:** 検証holdoutを宣言したstudyでは、tunerは最終準備run上で `purpose: final_evaluation`・`evaluate: false` 以外の探索を拒否する。

2026-09-28、ユーザーが上記の解釈と補完（grid.yamlの `fill:`）をこの内容で承認した。

**計画書からの逸脱:**

- **IDW（計画書§3.4、§5、§6.9）:** 行わない。FORGEの整合パイプラインにはIDWの波形補間手法がなく、比較契約の6手法（`forge_mvp_contract.METHODS`）にも含まれない。IDWは下限の目安で、事前宣言した6対比には入らない。
- **マスクAの副次評価（計画書§2.3）:** 行わない。計画書§4で公平性を保ったまま削れる項目として挙げたものである。
- **密度差の影響（計画書§2.3）:** 探索時の勝者（seed 7301）の重みは上記のとおり全shadowで保存する。この重みからB testを予測する経路はまだないので、主結果の評価の後に補足として実装して報告する。選択には使わない。10,000更新時点の重みによる更新数の感度（計画書§4）も同じ扱いとする。

**事前登録したファイル（sha256）:**

| ファイル | sha256 |
|---|---|
| [grid.yaml](grid.yaml) | `f62e0129552e7d53eb9e70a78f0abb860bda0afa1e72a8c88f10e0f421c4f76b` |
| [config.yaml](config.yaml) | `759439cf7b98bfc56d7165f1af86ff8ecfcd110cab99625c139aa7dda7e7f940` |
| [inputs.yaml](inputs.yaml) | `cf34d5a71ec97d9620d61e396f2bdc62bca08df10a95c555272053cb49f3a199` |
| [ccnet5d_grid.yaml](ccnet5d_grid.yaml) | `c2fc6e09b34839cb6465184ab82eaa2ae45edc64e7493c55f71162b3f6e7bbb3` |
| [drr_grid.yaml](drr_grid.yaml) | `d2270c5482caeb0fca87b9b8f81aeb81cd0d45a7fd622b9fb4039b488a051e91` |
| [nersi_real.yaml](nersi_real.yaml) | `96b29db439732cc7a53cd73a5ca0f1b7435d93c4dbdd72a06ba20eef371dc9c8` |
| [pocs_grid.yaml](pocs_grid.yaml) | `664dbfce5c83c6ee7ee0e0e657ed554a165df8a4ff1f9305e56e20357c3358e6` |
| [regsi_grid.yaml](regsi_grid.yaml) | `feaac8f686f5af1b009d2db88959aefef16c32746d8ab8400d48a02df7dedeb9` |
| [regsi_real.yaml](regsi_real.yaml) | `005b181a192a860db5cb184d8ad4959c2548174b2ab5aa3103ede139d8ea2002` |

各段階の `stage.json` はgrid.yamlのsha256を記録する。grid.yamlを変えると、それより後の段階は生成できない。

**Evidence:**
- マスクBのmask hash: `4eaa2494eb9ea71b8de392b18d82ee5e9a05f44c5b13d30ad689077d5176701e`
- 検証準備runのmask hash: `a1e15422431f7aeac429c4c9e01d2218af3a43a9ab4aeefd81b6ab6ccd494531`
- Vのセル一覧（int64昇順）のsha256: `092815f4ac87641e9f0d0a8dd82931eef035ef056782d3d433bb62e33f2e01b6`
- 準備run: `runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2`（最終）、`runs/study_060_forge_fair_grid/20260927T133848730415Z_25c5193588b2_validation`（探索）
- S1生成時の実現性の確認値は、各 `runs/study_060_forge_fair_grid/fair_grid/<method>/S1/stage.json` の `feasibility` に記録した。
  - ReGSIの隠し本数（探索時→再学習時）: 640→768 / 1,344→1,472 / 3,328→3,712。すべて64の倍数である。
  - CCNetのパッチ1枚あたりの最小隠し本数（マスクB、N=13,271 / 14,745）: [4,8,4,8]@0.4で92 / 104、@0.3で69 / 78、[6,6,8,6]@0.2で82 / 92。

## 2026-09-28 — CCNet5Dの後半の段階をGPU 1にも振り分ける

**Status:** active

**Decision:**
ユーザーの承認により、ReGSIの全runが最終再学習まで終わった後に生成するCCNet5Dの段階は、
`scripts/forge_fair_grid.py plan/select --devices cuda:0,cuda:1` でrunを2枚のGPUに交互に割り当てる。
実行先はrunのyamlと `stage.json` の `devices` に記録する。

grid.yaml・候補・選択規則・予算は変えない。変わるのは実行先だけである。
GPU上の計算はGPUが違うとビット単位では一致しないが、どのGPUでも同じ条件で学習する。
それ以前に生成した段階（S1など）は元のcuda:0のままとする。

**Evidence:**
- CCNet5DのS1は3本並列でGPU 0を使い切り、1 runに約10〜14時間かかった。ReGSIの終了後はGPU 1が空く。
