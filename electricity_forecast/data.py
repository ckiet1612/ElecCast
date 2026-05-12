from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

from .types import LOCAL_TIMEZONE, DataPaths, ParsedName

RAW_VALUE_COLUMN = "original_value_float"
LOCAL_TZ = LOCAL_TIMEZONE
NAME_RE = re.compile(r"^(?:System1:)?PMS_(?P<core>.+?)\.value\.PVLAST$")


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


def read_telemetry_hourly_features(
    csv_path: str | Path,
    chunksize: int = 250_000,
):
    """Read data_2026.csv once and build hourly P/PF/current/KWH features."""
    import pandas as pd

    raw = _read_mixed_metric_hourly(csv_path, ["P", "PF", "IAVG", "KWH"], chunksize)
    columns = [
        "timestamp_local",
        "meter",
        "area",
        "p",
        "pf",
        "iavg",
        *_TELEMETRY_KWH_COLUMNS[3:],
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    pivot = pivot_metric_features(raw)
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
    rename = {
        column: column.lower().replace("%", "pct_").replace("-", "_")
        for column in pivot.columns
    }
    return pivot.rename(columns=rename)


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
