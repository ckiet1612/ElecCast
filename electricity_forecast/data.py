from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

from .types import LOCAL_TIMEZONE, DataPaths, ParsedName

RAW_VALUE_COLUMN = "original_value_float"
LOCAL_TZ = LOCAL_TIMEZONE
NAME_RE = re.compile(r"^(?:System1:)?PMS_(?P<core>.+?)\.value\.PVLAST$")
TELEMETRY_METRICS = [
    "KWH",
    "P",
    "Q",
    "S",
    "PF",
    "IA",
    "IB",
    "IC",
    "IAVG",
    "%V",
    "%A",
    "VAVG",
    "VAB",
    "VBC",
    "VCA",
    "THD-R-I1",
    "THD-R-I2",
    "THD-R-I3",
    "THD-R-U1",
    "THD-R-U2",
    "THD-R-U3",
]


def parse_meter_name(raw_name: str) -> ParsedName:
    """Parse SCADA names like System1:PMS_FB2_MSB01_KWH.value.PVLAST."""
    match = NAME_RE.match(str(raw_name).strip())
    core = match.group("core") if match else str(raw_name).strip()
    if "_" not in core:
        meter, metric = core, ""
    else:
        meter, metric = core.rsplit("_", 1)
    if "_MSB" in meter:
        area = meter.split("_MSB", 1)[0]
    else:
        area = meter.split("_", 1)[0]
    return ParsedName(
        raw_name=str(raw_name), meter=meter, area=area, metric=metric.upper()
    )


def parse_names_series(series):
    """Vectorized-ish helper returning meter/area/metric columns for a pandas Series."""
    parsed = series.map(parse_meter_name)
    return (
        parsed.map(lambda item: item.meter),
        parsed.map(lambda item: item.area),
        parsed.map(lambda item: item.metric),
    )


def read_raw_metric_hourly(
    csv_path: str | Path,
    metrics: Iterable[str],
    chunksize: int = 250_000,
    aggfunc: str = "mean",
):
    """Read large raw CSV files and aggregate selected metrics to hourly means."""
    import pandas as pd

    path = Path(csv_path)
    wanted = {metric.upper() for metric in metrics}
    chunks = []
    usecols = ["time", "name", RAW_VALUE_COLUMN]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk["meter"], chunk["area"], chunk["metric"] = parse_names_series(
            chunk["name"]
        )
        chunk = chunk[chunk["metric"].isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["timestamp_local"] = (
            pd.to_datetime(chunk["time"], utc=True, errors="coerce")
            .dt.tz_convert(LOCAL_TZ)
            .dt.floor("h")
        )
        chunk["value"] = pd.to_numeric(chunk[RAW_VALUE_COLUMN], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp_local", "meter", "value"])
        grouped = _aggregate_metric_chunk(chunk, aggfunc)
        chunks.append(grouped)

    if not chunks:
        return pd.DataFrame(
            columns=["timestamp_local", "meter", "area", "metric", "value"]
        )

    data = pd.concat(chunks, ignore_index=True)
    grouped = data.groupby(
        ["timestamp_local", "meter", "area", "metric"], as_index=False
    )["value"]
    if aggfunc == "max":
        return grouped.max()
    return grouped.mean()


def read_cumulative_kwh_hourly(
    csv_path: str | Path,
    chunksize: int = 250_000,
):
    """Read cumulative KWH telemetry and convert it to hourly consumption deltas."""
    import pandas as pd

    raw = read_raw_metric_hourly(csv_path, ["KWH"], chunksize=chunksize, aggfunc="max")
    if raw.empty:
        return pd.DataFrame(columns=_TELEMETRY_KWH_COLUMNS)

    raw = raw.rename(columns={"value": "kwh_cumulative"})
    raw = raw[["timestamp_local", "meter", "area", "kwh_cumulative"]].copy()
    return _cumulative_kwh_to_hourly(raw)


def read_guest_counts(guests_csv: str | Path):
    """Read optional customer/visitor counts and aggregate them hourly."""
    import pandas as pd

    path = Path(guests_csv)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_local", "guest_count"])

    normalized = {_normalize_column_name(column): column for column in df.columns}
    time_col = _first_existing(
        normalized,
        [
            "datetime",
            "timestamp",
            "timestamp_local",
            "time",
            "date_time",
            "ngay_gio",
        ],
    )
    date_col = _first_existing(normalized, ["date", "day", "ngay"])
    hour_col = _first_existing(normalized, ["hour", "gio"])
    value_col = _first_existing(
        normalized,
        [
            "guest_count",
            "guests",
            "visitor_count",
            "visitors",
            "customer_count",
            "customers",
            "khach",
            "so_khach",
            "so_luong_khach",
            "soluongkhach",
            "value",
        ],
    )
    if value_col is None:
        raise ValueError(
            f"{path} missing customer count column. Expected visitors, guest_count, customer_count, or value."
        )
    if time_col is None and date_col is None:
        raise ValueError(
            f"{path} missing datetime column. Expected datetime, timestamp, time, or date/hour."
        )

    guests = pd.DataFrame()
    if time_col is not None:
        guests["timestamp_local"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        guests["timestamp_local"] = pd.to_datetime(df[date_col], errors="coerce")
        if hour_col is not None:
            guests["timestamp_local"] = guests["timestamp_local"] + pd.to_timedelta(
                pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int),
                unit="h",
            )
    guests["timestamp_local"] = _to_local_hour(guests["timestamp_local"])
    guests["guest_count"] = pd.to_numeric(df[value_col], errors="coerce")

    area_col = _first_existing(normalized, ["area", "khu_vuc", "khuvuc"])
    meter_col = _first_existing(
        normalized,
        ["meter", "meter_id", "device", "device_id", "dong_ho", "dongho"],
    )
    group_cols = ["timestamp_local"]
    if area_col is not None:
        guests["area"] = df[area_col].astype(str).str.strip()
        group_cols.append("area")
    if meter_col is not None:
        guests["meter"] = df[meter_col].astype(str).str.strip()
        group_cols.append("meter")

    guests = guests.dropna(subset=["timestamp_local", "guest_count"])
    guests["timestamp_local"] = guests["timestamp_local"].dt.floor("h")
    if guests.empty:
        return pd.DataFrame(columns=[*group_cols, "guest_count"])
    return guests.groupby(group_cols, as_index=False)["guest_count"].mean()


def read_telemetry_hourly_features(
    csv_path: str | Path,
    chunksize: int = 250_000,
):
    """Read data_2026.csv once and build hourly electrical telemetry features."""
    import pandas as pd

    raw = _read_mixed_metric_hourly(csv_path, TELEMETRY_METRICS, chunksize)
    columns = [
        "timestamp_local",
        "meter",
        "area",
        "p",
        "q",
        "s",
        "pf",
        "ia",
        "ib",
        "ic",
        "iavg",
        "voltage_imbalance_pct",
        "current_imbalance_pct",
        "vavg",
        "thd_current",
        "thd_voltage",
        *_TELEMETRY_KWH_COLUMNS[3:],
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    pivot = pivot_metric_features(raw)
    pivot = _derive_telemetry_columns(pivot)
    if "kwh" in pivot:
        pivot = pivot.rename(columns={"kwh": "kwh_cumulative"})
    if "kwh_cumulative" in pivot:
        telemetry_kwh = _cumulative_kwh_to_hourly(
            pivot[["timestamp_local", "meter", "area", "kwh_cumulative"]].dropna(
                subset=["kwh_cumulative"]
            )
        )
        pivot = pivot.drop(columns=["kwh_cumulative"])
        pivot = pivot.merge(
            telemetry_kwh,
            on=["timestamp_local", "meter", "area"],
            how="outer",
        )
    for column in columns:
        if column not in pivot:
            pivot[column] = pd.NA
    return (
        pivot[columns].sort_values(["meter", "timestamp_local"]).reset_index(drop=True)
    )


_TELEMETRY_KWH_COLUMNS = [
    "timestamp_local",
    "meter",
    "area",
    "kwh_cumulative",
    "kwh_telemetry_raw_delta",
    "kwh_telemetry",
    "kwh_telemetry_issue",
]


def _cumulative_kwh_to_hourly(raw):
    import pandas as pd

    if raw.empty:
        return pd.DataFrame(columns=_TELEMETRY_KWH_COLUMNS)

    pieces = []
    for _, group in raw.sort_values(["meter", "timestamp_local"]).groupby("meter"):
        meter_data = group.copy()
        raw_delta = meter_data["kwh_cumulative"].diff()
        issue = pd.Series("", index=meter_data.index, dtype="object")
        issue = issue.mask(raw_delta < 0, "kwh_reset_or_negative_delta")
        high_limit = _large_delta_limit(raw_delta)
        issue = issue.mask(
            raw_delta.notna() & raw_delta.gt(high_limit),
            "kwh_delta_outlier",
        )
        valid_delta = raw_delta.mask(issue.ne("") | raw_delta.isna())
        meter_data["kwh_telemetry_raw_delta"] = raw_delta
        meter_data["kwh_telemetry"] = valid_delta
        meter_data["kwh_telemetry_issue"] = issue
        pieces.append(meter_data)

    return pd.concat(pieces, ignore_index=True)[_TELEMETRY_KWH_COLUMNS]


def _read_mixed_metric_hourly(
    csv_path: str | Path,
    metrics: Iterable[str],
    chunksize: int,
):
    import pandas as pd

    path = Path(csv_path)
    wanted = {metric.upper() for metric in metrics}
    chunks = []
    usecols = ["time", "name", RAW_VALUE_COLUMN]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk["meter"], chunk["area"], chunk["metric"] = parse_names_series(
            chunk["name"]
        )
        chunk = chunk[chunk["metric"].isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["timestamp_local"] = (
            pd.to_datetime(chunk["time"], utc=True, errors="coerce")
            .dt.tz_convert(LOCAL_TZ)
            .dt.floor("h")
        )
        chunk["value"] = pd.to_numeric(chunk[RAW_VALUE_COLUMN], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp_local", "meter", "value"])
        if chunk.empty:
            continue
        kwh = chunk[chunk["metric"].eq("KWH")]
        other = chunk[~chunk["metric"].eq("KWH")]
        if not kwh.empty:
            chunks.append(_aggregate_metric_chunk(kwh, "max"))
        if not other.empty:
            chunks.append(_aggregate_metric_chunk(other, "mean"))

    if not chunks:
        return pd.DataFrame(
            columns=["timestamp_local", "meter", "area", "metric", "value"]
        )

    data = pd.concat(chunks, ignore_index=True)
    kwh_data = data[data["metric"].eq("KWH")]
    other_data = data[~data["metric"].eq("KWH")]
    final_chunks = []
    if not kwh_data.empty:
        final_chunks.append(_aggregate_metric_chunk(kwh_data, "max"))
    if not other_data.empty:
        final_chunks.append(_aggregate_metric_chunk(other_data, "mean"))
    return pd.concat(final_chunks, ignore_index=True)


def _aggregate_metric_chunk(chunk, aggfunc: str):
    grouped = chunk.groupby(
        ["timestamp_local", "meter", "area", "metric"], as_index=False
    )["value"]
    if aggfunc == "max":
        return grouped.max()
    return grouped.mean()


def _normalize_column_name(column) -> str:
    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    return normalized.strip("_")


def _first_existing(normalized_columns: dict[str, str], candidates: list[str]):
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def _to_local_hour(values):
    if values.dt.tz is None:
        return values.dt.tz_localize(
            LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT"
        )
    return values.dt.tz_convert(LOCAL_TZ)


def _large_delta_limit(delta) -> float:
    positive = delta[delta > 0].dropna()
    if len(positive) < 4:
        return math.inf
    q1 = positive.quantile(0.25)
    q3 = positive.quantile(0.75)
    iqr = q3 - q1
    median = positive.median()
    if iqr > 0:
        return float(max(q3 + 6 * iqr, median * 8, q3))
    if median > 0:
        return float(median * 8)
    return math.inf


def pivot_metric_features(raw_hourly):
    import pandas as pd

    if raw_hourly.empty:
        return pd.DataFrame(columns=["timestamp_local", "meter", "area"])
    pivot = raw_hourly.pivot_table(
        index=["timestamp_local", "meter", "area"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    rename = {column: _metric_column_name(column) for column in pivot.columns}
    return pivot.rename(columns=rename)


def _metric_column_name(column) -> str:
    name = str(column)
    mapping = {
        "%V": "voltage_imbalance_pct",
        "%A": "current_imbalance_pct",
    }
    if name in mapping:
        return mapping[name]
    return name.lower().replace("%", "pct_").replace("-", "_")


def _derive_telemetry_columns(pivot):
    """Derive document-level anomaly metrics from raw data_2026 columns."""
    import pandas as pd

    data = pivot.copy()
    phase_current_cols = [col for col in ["ia", "ib", "ic"] if col in data]
    phase_voltage_cols = [col for col in ["vab", "vbc", "vca"] if col in data]
    thd_current_cols = [
        col for col in ["thd_r_i1", "thd_r_i2", "thd_r_i3"] if col in data
    ]
    thd_voltage_cols = [
        col for col in ["thd_r_u1", "thd_r_u2", "thd_r_u3"] if col in data
    ]

    if "p" in data and "q" in data:
        derived_s = (data["p"].pow(2) + data["q"].pow(2)).pow(0.5)
        if "s" in data:
            data["s"] = data["s"].fillna(derived_s)
        else:
            data["s"] = derived_s
    if "p" in data and "s" in data:
        derived_pf = (data["p"].abs() / data["s"]).where(data["s"].ne(0))
        if "pf" in data:
            data["pf"] = data["pf"].fillna(derived_pf)
        else:
            data["pf"] = derived_pf
    if phase_current_cols:
        derived_iavg = data[phase_current_cols].mean(axis=1)
        if "iavg" in data:
            data["iavg"] = data["iavg"].fillna(derived_iavg)
        else:
            data["iavg"] = derived_iavg
    if phase_voltage_cols:
        derived_vavg = data[phase_voltage_cols].mean(axis=1)
        if "vavg" in data:
            data["vavg"] = data["vavg"].fillna(derived_vavg)
        else:
            data["vavg"] = derived_vavg
    if len(phase_current_cols) >= 2:
        phase_current = data[phase_current_cols]
        phase_avg = phase_current.mean(axis=1)
        derived_current_imbalance = (
            phase_current.sub(phase_avg, axis=0).abs().max(axis=1) / phase_avg
        ).where(phase_avg.ne(0)) * 100
        if "current_imbalance_pct" in data:
            data["current_imbalance_pct"] = data["current_imbalance_pct"].fillna(
                derived_current_imbalance
            )
        else:
            data["current_imbalance_pct"] = derived_current_imbalance
    if len(phase_voltage_cols) >= 2:
        phase_voltage = data[phase_voltage_cols]
        phase_avg = phase_voltage.mean(axis=1)
        derived_voltage_imbalance = (
            phase_voltage.sub(phase_avg, axis=0).abs().max(axis=1) / phase_avg
        ).where(phase_avg.ne(0)) * 100
        if "voltage_imbalance_pct" in data:
            data["voltage_imbalance_pct"] = data["voltage_imbalance_pct"].fillna(
                derived_voltage_imbalance
            )
        else:
            data["voltage_imbalance_pct"] = derived_voltage_imbalance
    if thd_current_cols:
        data["thd_current"] = data[thd_current_cols].max(axis=1)
    if thd_voltage_cols:
        data["thd_voltage"] = data[thd_voltage_cols].max(axis=1)

    for column in [
        "iavg",
        "vavg",
        "current_imbalance_pct",
        "voltage_imbalance_pct",
        "thd_current",
        "thd_voltage",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def summarize_csv(
    path: str | Path, sample_rows: int = 50_000, full_threshold_mb: float = 25.0
) -> dict[str, object]:
    import pandas as pd

    csv_path = Path(path)
    size_mb = round(csv_path.stat().st_size / 1024 / 1024, 2)
    read_kwargs = {} if size_mb <= full_threshold_mb else {"nrows": sample_rows}
    df = pd.read_csv(csv_path, **read_kwargs)
    sampled = "nrows" in read_kwargs
    row_count: int | str = len(df) if not sampled else f">={len(df):,} (sample)"
    min_time = None
    max_time = None
    meters: set[str] = set()
    areas: set[str] = set()
    metrics: set[str] = set()
    columns = list(df.columns)
    if "time" in df:
        times = df["time"].dropna()
        if not times.empty:
            min_time = str(times.min())
            max_time = str(times.max())
    if "name" in df:
        parsed = df["name"].dropna().map(parse_meter_name)
        meters.update(parsed.map(lambda item: item.meter))
        areas.update(parsed.map(lambda item: item.area))
        metrics.update(parsed.map(lambda item: item.metric))

    return {
        "path": str(csv_path),
        "size_mb": size_mb,
        "rows": row_count,
        "columns": columns or [],
        "min_time": min_time,
        "max_time": max_time,
        "meters": len(meters),
        "areas": len(areas),
        "metrics": sorted(metric for metric in metrics if metric),
        "sampled": sampled,
    }


def summarize_paths(paths: DataPaths) -> list[dict[str, object]]:
    summaries = []
    summary = summarize_csv(paths.telemetry_csv)
    summary["label"] = "data_2026"
    summaries.append(summary)
    if paths.guests_csv:
        summary = summarize_csv(paths.guests_csv)
        summary["label"] = "customer_list"
        summaries.append(summary)
    return summaries


def default_temperature(timestamp) -> float:
    hour = int(timestamp.hour)
    day_of_year = int(timestamp.dayofyear)
    daily = 3.0 * math.sin((hour - 6) / 24 * 2 * math.pi)
    seasonal = 1.5 * math.sin(day_of_year / 365 * 2 * math.pi)
    return round(27.0 + daily + seasonal, 2)


def default_guest_count(timestamp, area: str) -> float:
    hour = int(timestamp.hour)
    weekend_boost = 1.25 if int(timestamp.dayofweek) >= 5 else 1.0
    evening_boost = 1.15 if 17 <= hour <= 22 else 1.0
    area_seed = (sum(ord(char) for char in area) % 45) + 55
    return round(area_seed * weekend_boost * evening_boost, 2)
