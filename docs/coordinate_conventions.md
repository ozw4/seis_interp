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

argument順は論文式に合わせて`atan2(dx, dy)`とし、`[0, 360)`へwrapする。SEG C3 NAの既存QCで使っている90～270°の表現と揃えるためのPOC上の決定である。`dataset.json`には次の文字列で記録する。

```text
degrees(atan2(source_x-receiver_x, source_y-receiver_y)) wrapped to [0, 360)
```

## Time axis

```python
time_s = np.arange(sample_count, dtype=np.float64) * sample_interval_s
```

- recording delayはこのPOCでは使用せず、time originは0秒とする（`dataset.json`の`time_origin_s`は`0.0`）。
- `sample_count`はSEG-Y fileのsample数から取得する。
- `sample_interval_s`はSEG-Y binary headerのsample interval（microsecond）から取得し、秒へ変換する。625 samplesや8 msのようなsurvey固有値をhard-codeしない。

## Interim datasetのcoordinate columns

`dataset.json`の`coordinate_columns`は次の順序で記録する。

```text
time_s
cmp_x_m
cmp_y_m
offset_m
azimuth_deg
```

`traces.parquet`の1行と`amplitudes.npy`の1行は`array_row`で対応する。
