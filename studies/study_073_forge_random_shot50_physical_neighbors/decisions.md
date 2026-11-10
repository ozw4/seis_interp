# study_073 decisions

## 2026-11-10 — 物理スケールの順位付けは採用しない

**Status:** 採用（不採用の判断）。`context.distance` の既定は `index` のまま維持する。

**Decision:** 近傍の順位付けを取得幾何の物理座標（m）へ変えても shot50 は改善しない。
clean_target で −0.1684 dB / relMSE +0.0648 の劣化であり、`physical` は採用しない。
宣言可能なオプションとしてはコードに残す（既定は `index`）。

**Evidence:** study_072 との差は手法設定の1行（`context.distance: physical`）だけで、
mask・入力10成果物・model・近傍数 16・Fourier base 1.05・予算・seed は同一である。

| | clean_target SNR (dB) | clean_target relMSE | 学習 loss |
|---|---:|---:|---:|
| index（study_072） | 0.3596 | 1.0005 | 0.2043 |
| physical（study_073） | 0.1912 | 1.0652 | 0.2038 |

学習 loss がほぼ同一なので、観測トレースへの適合能力は変わらず、差は汎化だけに現れる。
劣化は edge shot に集中する（−0.1023 → −0.8687 dB、relMSE 1.0667 → 1.3476）。
両順位付けが選ぶ近傍の不一致率も edge shot で最大である（集合が完全一致する query は
internal 83.88 % に対し edge 56.46 %）。

**事前の予測との関係:** 本 study は「改善を予想しない」と launch 前に README へ記録していた。
根拠は、FORGE では index 正規化が偶然物理異方性に近いこと（source 比 6.20 対 7.45、
receiver 比 4.43 対 4.00）と、test query の 16 近傍が平均 15.72 本一致し最近傍は 100 % 一致すること、
そして同一 shot gather 内の近傍が双方ゼロで study_072 の後退の構造的原因が変わらないことだった。
予測は当たり、加えて小幅な劣化であることが確定した。

**理由（設計上の含意）:** field が読む座標と近傍の順位付けが同じ空間であるほうが良い。
`cinr_real` の入力は4つの正規化格子座標なので、同じ空間で選んだ近傍は model の関数自身に対して
局所的である。物理座標で選ぶと、crooked な測線と edge clamping により格子局所性からずれ、
格子上の近さと物理上の近さが最も乖離する survey 端で損をする。異方性の重みが「軸の index 本数由来」
であるという指摘自体は正しいが、FORGE ではその重みが物理異方性の 11–17 % 以内にあり、
格子局所性の利得がそれを上回る。

**帰結と未決事項:**

- `context.distance: physical` はコードに残すが既定にしない。他の取得幾何（index 比と物理比が
  大きく乖離する survey）では結論が変わりうるので、オプションとして保持する。
- 異方性を直したいなら、物理座標へ丸ごと移すのではなく、**格子座標のまま軸ごとの重みを宣言する**形が
  本測定と整合する。これは別 study の対象とする。
- study_072 の後退（gather 内近傍ゼロ）は本 study では一切改善していない。whole-shot 欠損に対しては
  近傍の定義を関係別（common receiver / common offset など）に変える必要があり、
  これも別 study の対象である。
- 1 mask・1 model seed の記述的比較であり、study_072 / study_069 との対比は事後的・探索的である。
