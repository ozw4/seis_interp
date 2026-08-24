# Coordinate conventions

> POC向け / Step 1 / SEG C3 Narrow-Azimuth

Step 1では、SEG-Yを固定形状の5D配列へ変換せず、各traceを1行とするtableとして扱う。ここではその際に採用した座標規約を記録する。実装は`src/seis_interp/processing/geometry.py`と`src/seis_interp/data/segy_index.py`にある。

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

vectorは`source - receiver`、argument順は`atan2(dx, dy)`、範囲は`[0, 360)`とする。0～180°へのfoldや`sin`/`cos`表現はStep 1では行わない。`dataset.json`には次の文字列で記録する。

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

## Model coordinates

model inputとして使うcoordinateの順序と単位は、`dataset.json`に次の形で記録する。

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

## 数値精度の境界

```text
物理座標と時間軸はinterim段階ではfloat64で保持する。
振幅はfloat32で保持する。
正規化後のモデル入力と学習targetはtraining境界でfloat32へ変換する。
Siren.forward()はdtype変換を行わない。
```

この変換は`to_model_tensors()`（`src/seis_interp/training/model_inputs.py`）を唯一の実装とする。

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
