"""Waveform measurements and review flags; no automatic noise removal."""

import numpy as np
import pandas as pd


def waveform_metrics(
    values: np.ndarray, dt_s: float, config: dict, aggregate_mask: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict]:
    """Measure all samples, with undefined statistics retained as NaN.

    Spectra use demeaned, Hann-windowed traces and one-sided power. Clipping
    flags detect repeated exact extrema, not instrument saturation itself.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("expected traces x samples (at least four), positive finite dt")
    n = x.shape[1]
    aggregate = (
        np.ones(len(x), dtype=bool)
        if aggregate_mask is None
        else np.asarray(aggregate_mask, dtype=bool)
    )
    if aggregate.shape != (len(x),):
        raise ValueError("aggregate mask must have one entry per trace")
    finite = np.isfinite(x).all(axis=1)
    safe = np.where(np.isfinite(x), x, 0)
    mean = safe.mean(axis=1)
    rms = np.sqrt(np.mean(safe**2, axis=1))
    peak = np.abs(safe).max(axis=1)
    centered = safe - mean[:, None]
    std = np.sqrt(np.mean(centered**2, axis=1))
    constant = finite & (np.ptp(safe, axis=1) == 0)
    zero = finite & (peak == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = peak / rms
        dc_ratio = np.abs(mean) / rms
    extreme = (safe == safe.max(axis=1)[:, None]) | (safe == safe.min(axis=1)[:, None])
    run_length = config["extreme_plateau_samples"]
    if not 2 <= run_length <= n:
        raise ValueError("extreme plateau length must be between two and sample count")
    width = n - run_length + 1
    runs = extreme[:, :width] & (safe[:, :width] != 0)
    for shift in range(1, run_length):
        runs &= safe[:, :width] == safe[:, shift : shift + width]
    plateau = runs.any(axis=1)
    table = pd.DataFrame(
        {
            "nonfinite_count": (~np.isfinite(x)).sum(axis=1),
            "all_zero": zero,
            "constant": constant,
            "zero_fraction": (safe == 0).mean(axis=1),
            "rms": rms,
            "mean": mean,
            "std": std,
            "max_abs": peak,
            "crest_factor": crest,
            "dc_to_rms": dc_ratio,
            "extreme_fraction": extreme.mean(axis=1),
            "extreme_plateau_review": plateau & finite & ~constant,
            "impulsive_review": finite & (crest > config["crest_factor_review"]),
            "dc_review": finite & ~zero & (dc_ratio > config["dc_to_rms_review"]),
        }
    )
    for column in [
        "zero_fraction",
        "rms",
        "mean",
        "std",
        "max_abs",
        "crest_factor",
        "dc_to_rms",
        "extreme_fraction",
    ]:
        table.loc[~finite, column] = np.nan
    for index, (start, stop) in enumerate(config["time_windows_s"]):
        times = np.arange(n) * dt_s
        selected = (times >= start) & (times < stop)
        if not selected.any():
            raise ValueError("time window contains no samples")
        energy = np.mean(safe[:, selected] ** 2, axis=1)
        table[f"window_{index}_rms"] = np.where(finite, np.sqrt(energy), np.nan)
    f = np.fft.rfftfreq(n, dt_s)
    power = np.abs(np.fft.rfft(centered * np.hanning(n), axis=1)) ** 2
    power[:, 1 : (-1 if n % 2 == 0 else None)] *= 2
    total = power.sum(axis=1)
    valid_spectrum = finite & ~constant & (total > 0)
    normalized = np.divide(
        power, total[:, None], out=np.zeros_like(power), where=valid_spectrum[:, None]
    )
    for name, (low, high) in config["frequency_bands_hz"].items():
        mask = (f >= low) & (f < high)
        table["power_fraction_" + name] = np.where(
            valid_spectrum, normalized[:, mask].sum(axis=1), np.nan
        )
    table["peak_frequency_hz"] = np.where(valid_spectrum, f[power.argmax(axis=1)], np.nan)
    table["numerically_usable"] = finite & ~constant
    return table, {
        "frequency_hz": f,
        "power_sum": power[valid_spectrum & aggregate].sum(axis=0),
        "normalized_power_sum": normalized[valid_spectrum & aggregate].sum(axis=0),
        "spectrum_count": int((valid_spectrum & aggregate).sum()),
        "time_energy_sum": (safe[finite & aggregate] ** 2).sum(axis=0),
        "finite_trace_count": int((finite & aggregate).sum()),
    }


def flag_relative_amplitude(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare RMS within each shot and offset bin; flags never imply rejection."""
    out = table.copy()
    out["offset_bin"] = np.floor(out.offset_m / config["offset_bin_width_m"])
    valid = out.header_eligible & out.numerically_usable
    baseline = out.rms.where(valid)
    groups = [out.source_file, out.offset_bin]
    median = baseline.groupby(groups).transform("median")
    count = baseline.groupby(groups).transform("count")
    out["rms_to_shot_offset_median"] = (out.rms / median).where(
        valid & count.ge(config["minimum_offset_bin_traces"])
    )
    ratio = out.rms_to_shot_offset_median
    out["amplitude_review"] = ratio.gt(config["rms_ratio_review"]) | ratio.lt(
        1 / config["rms_ratio_review"]
    )
    out["waveform_status"] = np.select(
        [
            ~out.header_eligible,
            out.nonfinite_count.gt(0),
            out.all_zero,
            out.constant,
            out[
                ["amplitude_review", "impulsive_review", "dc_review", "extreme_plateau_review"]
            ].any(axis=1),
        ],
        ["header_excluded", "nonfinite", "all_zero", "constant", "review"],
        default="passed_numeric_checks",
    )
    return out


def select_representative_shots(headers: pd.DataFrame) -> list[int]:
    """First/middle/last source point on each source line, independent of signal."""
    shots = headers[["ffid", "source_line", "source_point"]].drop_duplicates()
    selected = []
    for _, group in shots.groupby("source_line", sort=True):
        group = group.sort_values(["source_point", "ffid"])
        selected.extend(group.iloc[sorted({0, len(group) // 2, len(group) - 1})].ffid.tolist())
    return list(dict.fromkeys(map(int, selected)))


def summarize_waveform_qc(table: pd.DataFrame) -> dict:
    """Separate hard numeric failures, diagnostic flags, and component uncertainty."""
    candidate = table[table.header_eligible]
    columns = [c for c in table if c.startswith(("power_fraction_", "window_"))]
    columns += [
        "rms",
        "max_abs",
        "crest_factor",
        "dc_to_rms",
        "peak_frequency_hz",
        "rms_to_shot_offset_median",
    ]
    return {
        "trace_count": len(table),
        "candidate_count": len(candidate),
        "candidate_status_counts": candidate.waveform_status.value_counts().to_dict(),
        "candidate_review_counts": {
            c: int(candidate[c].sum())
            for c in ["amplitude_review", "impulsive_review", "dc_review", "extreme_plateau_review"]
        },
        "candidate_nonfinite_samples": int(candidate.nonfinite_count.sum()),
        "candidate_quantiles": {
            c: {
                str(q): (float(v) if pd.notna(v) else None)
                for q, v in candidate[c].quantile([0, 0.01, 0.5, 0.99, 1]).items()
            }
            for c in columns
        },
        "component_status": "single_recorded_channel_per_station; orientation_unconfirmed",
        "amplitude_units": "stored SEG-Y amplitude; physical calibration unconfirmed",
        "scope": "numeric checks on all traces; visual inspection of selected gathers only",
    }
