# Coordinate conventions

> POC向け / SEG C3 Narrow-Azimuth

SEG-Yを固定形状の5D配列へ変換せず、各traceを1行とするtableとして扱う。この文書はそのときの座標規約を定める。実装は`src/seis_interp/processing/geometry.py`と`src/seis_interp/data/segy_index.py`にある。

## SEG-Y coordinate scalar

trace headerの`SourceGroupScalar`をtraceごとに適用する。

```text
scalar > 0  : value * scalar
scalar < 0  : value / abs(scalar)
scalar == 0 : value
```

## Source / receiver header fields

| 値 | SEG-Y trace header |
|---|---|
| FFID | `FieldRecord` |
| coordinate scalar | `SourceGroupScalar` |
| source座標 | `SourceX`、`SourceY` |
| receiver座標 | `GroupX`、`GroupY` |
| 座標単位コード | `CoordinateUnits` |

`CoordinateUnits == 1`（length）だけをmeterとして受け入れ、それ以外は`ValueError`とする。scalar適用後の座標はmeterとして扱う。

## CMP

```python
cmp_x_m = 0.5 * (source_x_m + receiver_x_m)
cmp_y_m = 0.5 * (source_y_m + receiver_y_m)
```

## Offset

```python
dx_m = source_x_m - receiver_x_m
dy_m = source_y_m - receiver_y_m

offset_m = np.hypot(dx_m, dy_m)
```

offsetが0のtraceはerrorにしない。

## Azimuth

```python
azimuth_deg = np.mod(np.degrees(np.arctan2(dx_m, dy_m)), 360.0)
```

vectorは`source - receiver`、argument順は`atan2(dx, dy)`、範囲は`[0, 360)`とする。0～180°へのfoldは行わない。interim datasetでは監査可能な物理値として`azimuth_deg`をそのまま保持し、`sin`/`cos`への変換は保存時には行わない。`dataset.json`には次の文字列で記録する。

```text
degrees(atan2(source_x-receiver_x, source_y-receiver_y)) wrapped to [0, 360)
```

## 実装の正本

`apply_coordinate_scalar()`と`compute_trace_geometry()`（`src/seis_interp/processing/geometry.py`）を唯一の実装とする。QC・inspection・pipelineはこの2関数をimportして使い、再実装しない。

## Time axis

```python
time_s = np.arange(sample_count, dtype=np.float64) * sample_interval_s
```

- recording delayはこのPOCでは使用せず、time originは0秒とする（`dataset.json`の`time_origin_s`は`0.0`）。
- `sample_count`はSEG-Y fileのsample数から取得する。
- `sample_interval_s`はSEG-Y binary headerのsample interval（microsecond）から取得し、秒へ変換する。625 samplesや8 msのようなsurvey固有値をhard-codeしない。

## Stored physical coordinates

interim datasetに保存する物理coordinateの順序と単位は、`dataset.json`に次の形で記録する。
コード上の正本は`src/seis_interp/data/trace_schema.py`のphysical schema定数とし、保存と読み込みから参照する。

```json
{
  "coordinate_order": ["time_s", "cmp_x_m", "cmp_y_m", "offset_m", "azimuth_deg"],
  "coordinate_units": {
    "time_s": "s",
    "cmp_x_m": "m",
    "cmp_y_m": "m",
    "offset_m": "m",
    "azimuth_deg": "deg"
  }
}
```

`time_s`は`time_s.npy`、残り4つは`traces.parquet`の列である。`traces.parquet`の1行と`amplitudes.npy`の1行は`array_row`で対応する。

これは論文と同じ物理的な5D写像

```text
(time_s, cmp_x_m, cmp_y_m, offset_m, azimuth_deg) -> amplitude
```

を表す保存契約である。

## Numerical model features

modelと単純baselineへ渡すときだけ、物理azimuthから次をオンデマンドで導出する。

```python
azimuth_rad = np.deg2rad(azimuth_deg)
azimuth_sin = np.sin(azimuth_rad)
azimuth_cos = np.cos(azimuth_rad)
```

`azimuth_sin`と`azimuth_cos`はdimensionlessで、数値誤差の範囲で
`azimuth_sin**2 + azimuth_cos**2 == 1`となる単位円上の表現である。これにより0°と360°が同じ近傍へ連続的に写り、degreeを線形axisとして扱う場合の境界不連続を避ける。derived feature全体は保存せず、必要な行に対して都度生成する。
コード上では`build_spatial_model_coordinates()`（`src/seis_interp/processing/model_coordinates.py`）をこの変換の唯一の公開実装とし、出力順序は`trace_schema.py`のmodel feature schemaで固定する。

したがって、物理問題は5Dのままだが、SIRENへ渡す数値入力は次の6 featuresになる。

```text
time_s, cmp_x_m, cmp_y_m, offset_m, azimuth_sin, azimuth_cos
```

NN/IDW baselineはtimeを除いた5つのnumerical spatial featuresを使う。time、CMP、offsetにはtraining traceからfitしたmin-max変換を適用し、azimuthの2 featuresは単位円表現をそのまま使う。

論文は5D coordinateのazimuthを線形値として直接使用しており、sin/cos embeddingは規定していない。この周期表現は論文再現の要件ではなく、0°/360°境界を連続に扱うための`seis_interp` POC固有の設計判断である。

## 数値精度の境界

```text
物理座標と時間軸はinterim段階ではfloat64で保持する。
振幅はfloat32で保持する。
正規化およびazimuth encoding後の数値入力はfloat64で組み立てる。
モデル入力と学習targetはtraining境界でfloat32へ変換する。
Siren.forward()はdtype変換を行わない。
```

この変換は`to_model_tensors()`（`src/seis_interp/training/model_inputs.py`）を唯一の実装とする。

## Regeneration boundary

interim datasetは物理値の`azimuth_deg`だけを保存し、正規化契約を持たない。正規化はprocessed dataset側の契約であり、`normalization.json`、`trace_split.parquet`、`preparation.json`は`prepare-baseline`が一体で生成する。この3ファイルは手編集せず、座標規約または正規化契約を変えたときはprocessed datasetごと再生成する。

## Selectionの記録

どのtraceを、どの条件で抽出したかは`dataset.json`の`selection`に記録する。

```json
{
  "selection": {
    "ffid": 20,
    "expected_trace_count": 544
  }
}
```

`expected_trace_count`はcomplete shotの判定に使った値である。後続stepが`dataset.json`だけを見て抽出条件を再現できるよう、CLIやpipelineの戻り値にだけ存在する項目を作らない。`prepare_c3_complete_shot()`の戻り値は`dataset.json`の内容と完全に一致する。

544はSEG C3 NA固有の値なので、`C3_COMPLETE_SHOT_TRACE_COUNT`として`seis_interp.pipelines.prepare_c3`に置く。survey非依存の`select_ffid()`・`annotate_ffid_quality()`は`expected_trace_count`を必須引数とし、既定値を持たない。
