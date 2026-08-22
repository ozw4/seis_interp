# `seis_interp` リポジトリ フォルダ構成規約

> POC向け / Version 0.3 / 2026-08-22

## 目的

SEG C3 Narrow-Azimuthを用いる5D seismic interpolation POCで、コード、研究条件、データ、実行履歴、採用成果を混在させない。必要になったディレクトリだけを追加し、論文の完全再現よりも実証と再現性を優先する。

## 1. 配置の原則

| 対象 | 配置先 | 原則 |
|---|---|---|
| 再利用する実装 | `src/seis_interp/` | studyに依存しない処理を置く。結果へ影響するロジックはテスト可能にする。 |
| 薄い実行補助 | `scripts/` | 環境変数とCLI呼び出しだけを扱う。データ処理ロジックは置かない。 |
| 研究条件と判断記録 | `studies/<study>/` | 研究質問ごとに`README.md`、`config.yaml`、`inputs.yaml`を置く。 |
| データ | `data/` | 由来と処理段階で`external`、`interim`、`processed`を分ける。 |
| 実行履歴 | `runs/` | runごとの設定、入力、指標、ログ、成果物を機械生成し、手編集しない。 |
| 採用した成果 | `results/` | 採用した図表・モデルのみを置き、生成元runを記録する。 |
| 文書 | `docs/`、`reports/` | 規約は`docs/`、人が読む技術報告は`reports/`へ置く。 |

## 2. POCの標準構成

以下は開始時の標準形である。完成形を先回りして空ディレクトリを作らず、実装または成果物が発生した時点で追加する。

```text
seis_interp/
├── README.md
├── pyproject.toml
├── <lock-file>
├── .gitignore
├── .env.example
│
├── src/
│   └── seis_interp/
│       ├── __init__.py
│       ├── cli.py
│       ├── data/
│       ├── processing/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── visualization/
│       └── pipelines/
│
├── scripts/
│   ├── download_seg_c3_na.sh
│   ├── verify_seg_c3_na.sh
│   └── inspect_seg_c3_na.sh
│
├── configs/
│   └── default.yaml
│
├── studies/
│   ├── _template/
│   │   ├── README.md
│   │   ├── config.yaml
│   │   └── inputs.yaml
│   └── study_001_c3_na_baseline/
│       ├── README.md
│       ├── config.yaml
│       └── inputs.yaml
│
├── data/
│   ├── README.md
│   ├── external/
│   │   └── seg_c3_na/
│   │       ├── README.md
│   │       ├── manifest.yaml
│   │       └── <local SEG-Y files; ignored by Git>
│   ├── interim/
│   └── processed/
│
├── runs/
│   └── README.md
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── docs/
    └── repository_layout.md
```

`results/`と`reports/`は、採用成果または報告書が発生した時点で追加する。

Dev Container内では`SEIS_INTERP_DATA_ROOT=/workspace/data`とし、SEG C3 NAの実体は`/workspace/data/external/seg_c3_na/`へ置く。実SEG-Yと生成された`download.lock.yaml`はGit管理対象外とする。

## 3. ディレクトリ境界

| 場所 | 置くもの／置かないもの |
|---|---|
| `src/` | SEG-Y I/O、座標計算、mask、正規化、SIREN、学習、評価、可視化、pipelineを置く。Notebook専用コードやstudy固有条件は置かない。 |
| `scripts/` | CLIを呼ぶ薄いshell wrapperや環境セットアップ補助だけを置く。manifest解析、download、checksum、SEG-Y QCなどの主要ロジックは`src/`へ置く。 |
| `studies/` | 一つの研究質問を管理する。同じ問いでseedやepochだけを変える場合は別studyではなく別runとする。checkpointや全実行結果は置かない。 |
| `data/` | C3 NAは外部公開データなので`external/`へ置く。manifestと説明文書はGit管理し、実SEG-Y、大容量配列、download lockはGitへ入れない。 |
| `runs/` | run IDにUTC時刻とGit SHAを含める。resolved config、input lock、metrics、logs、checkpoint、figuresを保存する。 |
| `results/` | 採用判断後に追加する。全runのコピーではなく、正式採用した図表・モデル・評価結果だけを保持する。 |
| `notebooks/` | 必要な場合だけstudy配下に置き、geometry QC、探索、結果レビューに限定する。主要ロジックは`src/`からimportする。 |

## 4. 設定・命名・再現性

設定の優先順位は次とする。

```text
configs/default.yaml
    < studies/<study>/config.yaml
    < CLIでの明示指定
```

正式な実行で重要な値をCLIだけに残さず、継承後の設定を次へ保存する。

```text
runs/<study>/<run-id>/config.resolved.yaml
```

名前は小文字ASCIIの`snake_case`を基本とし、数値には可能な限り意味と単位を含める。

```yaml
time_window_s: 1.5
distance_m: 500
learning_rate: 1.0e-5
```

study名は、研究目的が分かる名称にする。

```text
study_001_c3_na_baseline
study_002_mask_sensitivity
study_003_omega0_sensitivity
```

各runには、少なくとも次を記録する。

- random seed
- Git commit
- 入力データのversionとchecksum
- 実行環境
- 開始・終了時刻
- 評価指標

テストでは実SEG-Yを使用せず、小さなsynthetic fixtureで座標計算、data split、model、metrics、tiny pipelineを検証する。

## 5. 禁止事項

| 避けるもの | 代替 |
|---|---|
| ルート直下の`train.py`、`plot.py`、`analysis.py` | 再利用コードは`src/`、単発条件はstudy、実行入口はCLIまたは薄いscriptへ分ける。 |
| `utils.py`、`misc.py`、`common.py` | `coordinates.py`、`trace_masks.py`、`metrics.py`のように責務を名前で示す。 |
| `latest`、`final2`、`temp`などの名称 | study ID、run ID、result IDで識別する。 |
| 絶対パス、秘密情報、大容量データのcommit | `.env`、manifest、Git ignore、checksumを使用する。 |
| run結果の手編集 | 条件を修正して再実行する。 |
| Notebookだけにある主要ロジック | `src/`へ移し、Notebookからimportする。 |
| 空の将来用ディレクトリ | 実際の責務または成果物が発生した時点で追加する。 |

## 6. 変更ルール

新しいディレクトリは、実際の責務または成果物が発生した場合だけ追加する。

フォルダ境界や命名規則を変更するPRでは、同時にこの文書を更新する。

POCの進行中も、次の境界を維持する。

```text
src/      = 再利用する実装
scripts/  = 薄い実行補助
studies/  = 研究条件と判断記録
runs/     = 実行履歴
results/  = 採用した成果
```
