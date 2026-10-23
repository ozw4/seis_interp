# 分割の運用定義

ユーザー指定のstudy_060と同じ観測holdout方式を採用する。
[study_060/config.yaml](../study_060_forge_fair_grid/config.yaml) の
`validation_holdout.fraction: 0.1` と、
[実際の検証準備記録](../../runs/study_060_forge_fair_grid/20260927T133848730415Z_25c5193588b2_validation/preparation_manifest.json)
を照合した。観測集合だけをSHA-256で順位付けし、`round(0.1 * N_observed)`
本を検証へ割り当てる既存関数 `make_validation_holdout` をそのまま使う。
C3は元の欠損率80%を維持するため、FORGEの欠損率70%は引き継がない。

S1の元観測マスクはstudy_062と同一になるよう、mask namespace・ID・seedを
維持する。追加3 runはユーザー指定どおり、同じ全領域内でトレース割当を
変更する。各seedで元観測26,214本のマスクを作り、その10%を検証に隠す。
inputに渡すのは残り23,593本のみである。Global RMSもこの入力だけでfitする。

S1と追加3 runの両方で同じ90/10規則を適用し、検証とtestを互いに分離する。
study_060の最終段階で検証トレースを観測へ戻す工程は、今回の25+3フローには
採用しない。追加3 run後に別の全観測再学習工程を設けないという依頼を維持する。
S1でtestは採点せず、全手法の設定確定後に各seedのtestを1回採点する。

## POCSのForge条件との一致

2026-10-22のユーザー指定により、POCSの再探索はstudy066 Forgeの
16反復・終端しきい値0.01の25構成を、その保存済み候補から取り込む。
空間窓（sl, shot）は[2,8]・[2,16]・[4,8]・[4,16]・[6,16]へ揃える。
C3の従来候補とは異なるため、元のgridと完了結果を上書きせず、
`pocs_forge_matched.yaml`を使う別runとする。C3の入力・分割・8 msの
時間軸・observed global RMSは維持する。今回の対象は検証だけの25構成であり、
追加seed評価やtest採点はこのrunに含めない。

16反復の完了結果との比較後、ユーザー指定により反復数だけを32へ変更する。
25構成の窓・終端しきい値0.01・C3の入力と検証割当・POCS実装は16反復runと
同一とし、新しいrunへ保存する。
