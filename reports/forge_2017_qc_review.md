# FORGE 2017：固定除外後の波形QCレビュー

2026-09-18。全1,106 SEG-Yファイルの波形監査を入力に、全ゼロ35本・非ゼロ定数100本の除外を固定した。残る1,919,596本について分布を集計し、8分類から中央付近と極端な例を各1本、計16本の原波形を再読込して確認した。

**ほぼ定数・DC偏り・振幅異常の追加除外は未採用。** ほぼ定数の指標が小さい波形には、変動がほとんど消失した例と、大きなDCに波形が重なった例が混在する。DCや生RMSの閾値だけで一括除外する根拠は不足している。

[実行条件](../studies/study_050_forge_qc_review/config.yaml) / [集計JSON](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/summary.json) / [閾値別集計CSV](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/threshold_sensitivity.csv) / [固定除外135本](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/fixed_exclusions.csv)。

## 固定したマスク

| 項目 | 本数 |
|---|---:|
| 全記録（補助・deadを含む） | 1,932,182 |
| ヘッダー上の候補 | 1,919,731 |
| 今回固定：全ゼロ | 35 |
| 今回固定：非ゼロ定数 | 100 |
| 固定除外後の候補 | 1,919,596 |
| 残存候補の要確認（従来の衝撃・極値フラグも含む） | 113,466 |

[fixed_qc_mask.parquet](../data/processed/forge_2017/20260918T060947568159Z_5d660f87b9b4/fixed_qc_mask.parquet) は全記録の行を保持する。キーは `source_file, trace_index`（0始まり）。`fixed_qc_excluded` は今回の135本、`eligible_after_fixed_qc` は既存のヘッダー除外も反映した候補、`review_pending` は追加判断が残る波形を表す。採用候補であることは良質な反射波であることを意味しない。元SEG-Yは変更していない。

## 指標と全体分布

全4,001サンプル、0–4秒を対象に、`q = std / RMS`、`d = abs(mean) / RMS` を用いた。stdは平均除去後のRMS。両者には `q² + d² ≈ 1` の関係があり、ほぼ定数とDC偏りは独立した異常ではない。丸め誤差を避けるため、qはstdから直接計算する。

相対振幅 `a` は同一ショット・500 m offset帯の候補RMS中央値に対する比。参考トレース20本未満の帯は未判定。平均除去後の比 `a_AC` では、分子・参照中央値ともstdに置き換える。振幅の判定可能数は1,914,165本、未判定は5,431本。振幅はファイル内の保存値で、物理単位への校正はしていない。

![全体分布](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/distributions.png)

| 指標 | 中央値 | 99パーセンタイル | 最小–最大 |
|---|---:|---:|---:|
| q | 0.999960 | ほぼ1 | 1.55e-9–約1 |
| d | 0.008925 | 0.2671 | 2.80e-9–1 |
| a | 1 | 10.264 | 6.76e-5–6,746.61 |
| a_AC | 1 | 10.095 | 8.07e-7–6,839.18 |

以下の割合は、未判定も含む固定除外後1,919,596本を分母にする。閾値は診断用で、除外条件ではない。

## ほぼ定数の波形

| qの上限（厳密な未満） | 本数 |
|---|---:|
| 1e-8 | 1 |
| 1e-6 | 1 |
| 1e-4 | 4 |
| 0.001 | 12 |
| 0.01 | 40 |
| 0.05 | 163 |
| 0.1 | 293 |

診断上の `q < 0.01` は40本（0.00208%）で、すべて受振点101/579、FFID 790–1055の一部に集中する。[40本の一覧](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/near_constant_candidates.csv)。非ゼロ定数100本も同じ受振点に集中している。

- **FFID 831 / R101/579**：q=1.55e-9。4,001サンプル中の値は2種類のみ、peak-to-peakは1.862645e-9（この値域のfloat32の1段階分）。末尾1サンプルの違いで厳密な定数判定を免れていた。平均除去後の相対振幅は8.07e-7で、波形変動は実質的に消失している。
- **FFID 982 / R101/579**：同じ40本の中央例だが、q=0.004719、生振幅比361.95に対して平均除去後は1.709。DCの上に振動が残り、FFID 831と同じ状態とは扱えない。
- 40本中、平均除去後の振幅比が0.01未満なのは3本、0.01–100に入るのは37本。範囲内でも近隣との波形整合やS/Nが確認できたことにはならない。

![受振点101/579の推移](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/receiver_101_579_sequence.png)

横軸FFIDはショット識別子で、等間隔の経過時間ではない。FFID後半でDC水準が変わり、固定定数除外とほぼ定数候補が集中する。

![ほぼ定数の極端例831](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/near_constant_extreme_ffid831.png)

![DC上に振動が残る982](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/near_constant_central_ffid982.png)

## DC偏り

| dの下限（厳密な超過） | 本数 |
|---|---:|
| 0.1 | 112,542 |
| 0.2 | 32,474 |
| 0.5 | 6,713 |
| 0.9 | 2,435 |
| 0.99 | 411 |
| 0.999 | 149 |

既存フラグ `d > 0.1` は5.8628%。前回の112,642本から非ゼロ定数100本を除いたため、112,542本になった。80,068本は0.1–0.2の比較的軽い範囲にある。DC比0.1はDCのエネルギー比では1%に対応する。

DCフラグが多い受振点は185/531（942/1,106本、85.17%）、181/574（891本、80.56%）、173/505（750本、67.81%）。受振点による反復性があり、測線組合せでも割合が異なる。

代表例のFFID 687（d=0.130）、533（0.264）、73（0.810）、530（0.995）では、平均除去後に近隣波形と似た振動・到来が目視できる。特にFFID 530 / R185/531は強いDCにもかかわらず平均除去後の振幅比0.790で、DC閾値だけによる除外は波形を失う可能性がある。これは選んだ例の所見であり、全DC対象を補正可能と判定したものではない。

![DC偏りの代表例73](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_above_0p5_central_ffid73.png)

## 振幅異常

| 診断 | 生RMSでの本数 | 平均除去後RMSで再集計した本数 |
|---|---:|---:|
| 参照中央値の100倍超 | 1,034 | 896 |
| 参照中央値の0.01倍未満 | 387 | 463 |

生振幅フラグは計1,421本（0.07403%）。生RMSで大振幅の1,034本の内訳は、平均除去後も100倍超が881本、0.01–100が151本、0.01未満が2本。平均除去後に新たに100倍を超える15本もあるため、右列は881本より多い。参照中央値も変わる点に注意する。生RMSで小振幅の387本は、平均除去後も全本0.01未満だった。

大振幅が多い受振点は185/503（591本）、101/579（112本）、105/584（68本）。小振幅は161/564（101本）、129/521（82本）、181/516（73本）。

- **大振幅・周期振動：FFID 139と635 / R185/503**。生振幅比253.4と6,746.6で、ほぼ全時間に持続する約60 Hzの振動と鋭いスペクトルピークがある。平均除去でも大振幅は残る。狭帯域ノイズを疑う例だが、発生原因は確定していない。近隣のフラグなし波形にも60 Hzピークが見られ、フラグなしを無雑音の正解とは扱えない。
- **大振幅・DC優勢：FFID 786 / R101/579**。生振幅比2,157.2から平均除去後43.81へ下がる。ただし近隣より変動が大きく、単に平均を引けば良質になるとは言えない。中央例929も同様に339.8から7.424へ下がる。
- **小振幅：FFID 83 / R181/516と370 / R129/521**。比は0.004028と6.76e-5。拡大すると非ゼロの振動はあるが、近隣の主要な到来と対応する構造が目視しにくく、広い帯域の細かな変動が目立つ。信号消失・低S/Nを疑う確認対象として残す。全387本について故障や欠損を確定したわけではない。

![大振幅の周期振動635](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_high_ac_extreme_ffid635.png)

![小振幅の極端例370](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_low_extreme_ffid370.png)

## 空間分布

![受振点ごとの要確認率](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/station_rates.png)

![震源測線×受振測線ごとの要確認率](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/line_pair_rates.png)

各地点・測線組合せの固定除外後候補数を分母にする。振幅未判定は振幅フラグの分子に入らない。[受振点別CSV](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/receiver_rates.csv)、[測線組合せ別CSV](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/line_pair_counts.csv)。測線図のDCと振幅では色軸の範囲が異なる。

## 代表波形16本の一覧

各分類の指標を昇順に並べ、中央の実測トレースと異常方向の極端値を選んだ。同値はファイル名・trace index順。分類間の件数は重複するため合算しない。`high_ac` は生振幅比>100かつd≤0.9という表示用分類で、平均除去後>100という定義ではない。

各図は赤が対象、青・緑が同じショット・受振測線で確認フラグのない近隣2本。近さは受振点番号差で選ぶため、図中の地点名と[対照一覧](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/representative_controls.csv)を併せて読む。例えばFFID 562では直近が要確認のため、9点・12点離れた対照になる。

左上は原振幅、右上は平均除去後の共通振幅軸、左下は各自のstdで割った表示、右下は平均除去・Hann窓後のスペクトルを各自の最大値で正規化した表示。正規化図は微小な数値残差も増幅するため、原振幅と併読する。平均除去・正規化は図のためだけに実施した。

| 分類 | 選択 | FFID / 受振点 | q | d | 生振幅比 | 平均除去後比 | 波形 |
|---|---|---|---:|---:|---:|---:|---|
| near_constant | central | 982 / 101/579 | 0.004719 | 0.999989 | 361.95 | 1.7091 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/near_constant_central_ffid982.png) |
| near_constant | extreme | 831 / 101/579 | 1.547e-09 | 1 | 521.64 | 8.0706e-07 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/near_constant_extreme_ffid831.png) |
| low_variation | central | 10 / 185/531 | 0.0533 | 0.998578 | 7.2666 | 0.3911 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/low_variation_central_ffid10.png) |
| low_variation | extreme | 793 / 101/579 | 0.01022 | 0.999948 | 795.24 | 8.1256 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/low_variation_extreme_ffid793.png) |
| dc_0p1_0p2 | central | 687 / 189/570 | 0.9915 | 0.129826 | 0.93543 | 0.9416 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_0p1_0p2_central_ffid687.png) |
| dc_0p1_0p2 | extreme | 895 / 121/551 | 0.9798 | 0.2 | 1.2063 | 1.1844 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_0p1_0p2_extreme_ffid895.png) |
| dc_0p2_0p5 | central | 533 / 161/522 | 0.9646 | 0.263842 | 0.45054 | 0.43459 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_0p2_0p5_central_ffid533.png) |
| dc_0p2_0p5 | extreme | 562 / 181/578 | 0.8661 | 0.499944 | 0.24031 | 0.21686 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_0p2_0p5_extreme_ffid562.png) |
| dc_above_0p5 | central | 73 / 181/574 | 0.5864 | 0.810013 | 1.3979 | 0.82413 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_above_0p5_central_ffid73.png) |
| dc_above_0p5 | extreme | 530 / 185/531 | 0.1 | 0.994983 | 7.5985 | 0.79004 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/dc_above_0p5_extreme_ffid530.png) |
| amplitude_high_dc | central | 929 / 101/579 | 0.02184 | 0.999761 | 339.77 | 7.4244 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_high_dc_central_ffid929.png) |
| amplitude_high_dc | extreme | 786 / 101/579 | 0.02022 | 0.999795 | 2157.2 | 43.81 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_high_dc_extreme_ffid786.png) |
| amplitude_high_ac | central | 139 / 185/503 | 1 | 0.000456138 | 253.43 | 253.46 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_high_ac_central_ffid139.png) |
| amplitude_high_ac | extreme | 635 / 185/503 | 1 | 2.30055e-05 | 6746.6 | 6839.2 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_high_ac_extreme_ffid635.png) |
| amplitude_low | central | 83 / 181/516 | 0.8257 | 0.564063 | 0.0040279 | 0.003326 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_low_central_ffid83.png) |
| amplitude_low | extreme | 370 / 129/521 | 0.9856 | 0.169182 | 6.7591e-05 | 6.6617e-05 | [図](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/figures/examples/amplitude_low_extreme_ffid370.png) |

丸め表示でd=1や0.2に見える場合も、分類は丸め前の数値で行う。[選択元の全指標](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/representative_traces.csv)、[再読込した波形の測定値](../runs/study_050_forge_qc_review/20260918T060947568159Z_5d660f87b9b4/representative_measurements.csv)、[全記録のレビュー指標](../data/interim/forge_2017/20260918T060947568159Z_5d660f87b9b4/qc_review.parquet)。

## 次の判断に向けて

今回固定した135本の除外に加え、次は「数値的に変動が消失した波形」「平均除去で有用な波形が残るDC偏り」「60 Hzの持続振動」「微小振幅で到来が見えない波形」を分けて規則を検討する。ほぼ定数はq単独ではなく平均除去後の振幅・量子化状態、振幅異常は近隣との時間窓別整合を併用するのが今回の所見に合う。追加の除外閾値、平均除去の実験用適用、notch、実験領域はまだ確定していない。

## 再現・検証

実行: `.venv/bin/python scripts/review_forge_qc.py`。

入力の波形Parquetと選択した16 SEG-YのSHA256を前回監査の記録と照合。実行runに入力ハッシュ、条件、コードsnapshot、出力Parquetハッシュ、環境情報を保存した。全数の分布は既存の全サンプル測定値を使い、原波形の再読込と目視は16例とその対照に限定した。

以下を実行し、静的検査と書式検査は成功、関連テスト15件が成功した。固定除外、追加レビューを除外にしないこと、閾値境界、参照母集団、決定的な例選択、1 ULPの残差、入出力ハッシュ照合を検証している。

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest tests/unit/test_forge_qc_review.py tests/integration/test_forge_qc_review.py tests/unit/test_forge_waveform_qc.py tests/integration/test_forge_waveforms.py
```
