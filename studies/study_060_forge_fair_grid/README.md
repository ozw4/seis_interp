# study_060_forge_fair_grid

Status: `pre-registered` — 探索グリッド・選択規則・検証holdoutを固定した。S1の探索yamlを生成済みで、学習runはまだ行っていない。

## 目的

- FORGE M1（欠損70%）で、GeRSI（ReGSI real/grid）・NeRSI・CCNet5D・POCS・DRRを、事前に登録した同じ探索手順で調整し、新しいマスクBのtest集合で1回だけ評価して比較する。
- 手法の選択には、マスクBの観測トレースから分けた検証集合Vだけを使う。

## 条件

実行条件の正本は次のファイルとする。

- 共通条件: [`config.yaml`](config.yaml)
- 入力: [`inputs.yaml`](inputs.yaml)
- 探索グリッドとseed: [`grid.yaml`](grid.yaml)
- 各手法の中心設定: [POCS](pocs_grid.yaml)、[DRR](drr_grid.yaml)、[CCNet5D](ccnet5d_grid.yaml)、[NeRSI](nersi_real.yaml)、[ReGSI-real](regsi_real.yaml)、[ReGSI-grid](regsi_grid.yaml)

判断理由・計画書の解釈・事前登録したファイルのsha256は [decisions.md](decisions.md) に置く。

準備run（どちらも `scripts/forge_m1_mvp.py prepare` で作成）:

| 準備run | 役割 | 観測 | 採点対象 | 除外 |
|---|---|---:|---:|---:|
| [`20260927T133829534335Z_25c5193588b2`](../../runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2/preparation_manifest.json) | 最終（マスクB） | 14,745 | test 34,406（clean 34,145） | 0 |
| [`20260927T133848730415Z_25c5193588b2_validation`](../../runs/study_060_forge_fair_grid/20260927T133848730415Z_25c5193588b2_validation/preparation_manifest.json) | 探索（`--validation-holdout`） | 13,271 | V 1,474（clean 1,471） | 34,406 |

探索用の準備runはマスクBのtest波形を読んでいない（`test_waveforms_read: 0`）。

configに書けない選択規則は次のとおり。すべて `pipelines/fair_grid_stage.py` が機械的に適用し、
各段階の判断と入力を `runs/study_060_forge_fair_grid/fair_grid/<method>/<stage>/stage.json` に記録する。

- スコアはVのcleanトレースでの平均 `masked_trace_relative_mse`。1 runのEMA 3水準をそれぞれ1構成として採点する。
- ニューラル手法は1手法16 run（S1 9、S2 3、S3 3、E 1）で、48構成になる。POCSとDRRはS1の16構成だけとする。
- S1は軸1×軸2の全組合せ。S2はS1の勝者に副次設定を3つ、S3はS2までの最良にweight decayを3水準加える。
- Eは次の順に、最初に当てはまる1 runを行う。
  1. wdの最大水準の最終学習lossが最小水準より5%未満しか高くない場合、wdを1段上げる。
  2. 勝者が軸1・軸2・wdの端にあり、その先の値が定義されている場合、その値へ延ばす。
  3. 軸1（未定義なら軸2）で、勝者とその良い方の隣の中点を試す。
  4. どれも当てはまらなければwdを1段上げる。
- Rでは、各runの最良EMAで上位2 runをseed 7302/7303で再学習する。2 run×EMA 3水準から、3 seedの平均Vが最小のものを選ぶ。
- ReGSIのS1〜Eは `regsi_real` だけで行う。Rでは `regsi_grid` を3 seedで追加し、realとgridそれぞれの3 seed平均をさらに平均した値で選ぶ。
- 同点は特別扱いしない。段階順、run名、EMAの宣言順で先に来たものを選ぶ。
- 発散run（非有限のloss・勾配ノルム・予測で失敗したrun）は予算に数え、再実行しない。選択では最下位、表と探索量の曲線ではスコア1.0とする。メモリ不足などそれ以外の失敗は同じyamlで再実行する。
- 選んだ設定をseed 7311/7312/7313でマスクBの観測14,745本から学習し直し、testを1回だけ評価する。
  最終runは選んだEMAだけを `evaluate: false` で予測・保存し、`scripts/forge_fair_grid_final_evaluation.py` でまとめて採点する。
  評価は、grid.yamlのbootstrap（10,000回、seed 600703）と各手法の `final/stage.json` に一致しない入力を、test波形を読む前に拒否する。
- IDWの下限、マスクAの副次評価は行わない。探索時の勝者によるB testの予測（密度差）と10,000更新時点の重みによる感度は、重みを保存しておき、主結果の後に補足として扱う（[decisions.md](decisions.md)）。

生成と記録の手順:

```bash
P=runs/study_060_forge_fair_grid
V=$P/20260927T133848730415Z_25c5193588b2_validation
F=$P/20260927T133829534335Z_25c5193588b2
.venv/bin/python scripts/forge_fair_grid.py plan --method regsi --stage S1 \
  --validation-preparation $V --final-preparation $F
# stage.json の commands に並ぶ scripts/forge_matched_tuning.py を実行してから次の段階を生成する。
.venv/bin/python scripts/forge_fair_grid.py status
.venv/bin/python scripts/forge_fair_grid.py select --method regsi \
  --validation-preparation $V --final-preparation $F
```

`--method` には `regsi`、`ccnet5d_grid`、`nersi_real`、`pocs_grid`、`drr_grid` を指定する。
前の段階のrunが1本でも終わっていなければ（完了も発散もしていなければ）、次の段階は生成しない。

最終評価の設定ファイル（全手法の `select` と最終runの後に作る）:

```yaml
preparation: runs/study_060_forge_fair_grid/20260927T133829534335Z_25c5193588b2
bootstrap: {resamples: 10000, seed: 600703}
final_runs:
  regsi_real:
    - {seed: 7311, tuning_run: runs/study_060_forge_fair_grid/<tuning run>, candidate: <final run name>}
  # pocs_grid・drr_grid（seed: null）、ccnet5d_grid、nersi_real、regsi_grid も同様
```

S1で生成したrun数とパラメータ数（`model_parameter_count`、4,343サンプルのキャンバス）:

| 手法 | S1 run数 | パラメータ数 |
|---|---:|---|
| ReGSI（real） | 9 | 363,269（k2 r2 W128）、370,295（k1 r3 W112）、352,347（k8 r1 W152） |
| CCNet5D | 9 | 353,413 |
| NeRSI | 9 | 369,839 |
| POCS | 16 | — |
| DRR | 16 | — |
