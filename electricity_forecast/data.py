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


def read_kwh_target(kwh_csv: str | Path):
    import pandas as pd

    path = Path(kwh_csv)
    df = pd.read_csv(path)
    required = {"name", "time", "hour", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df[["name", "time", "hour", "value"]].copy()
    df["meter"], df["area"], df["metric"] = parse_names_series(df["name"])
    df = df[df["metric"].eq("KWH")].copy()
    df["timestamp_local"] = pd.to_datetime(
        df["time"], errors="coerce"
    ) + pd.to_timedelta(
        pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int), unit="h"
    )
    df["timestamp_local"] = df["timestamp_local"].dt.tz_localize(
        LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    df["kwh"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp_local", "meter", "kwh"])
    df = (
        df.groupby(["timestamp_local", "meter", "area"], as_index=False)["kwh"]
        .mean()
        .sort_values(["meter", "timestamp_local"])
    )
    return df


def read_guest_counts(guests_csv: str | Path):
    import pandas as pd

    path = Path(guests_csv)
    df = pd.read_csv(path)
    required = {"datetime", "visitors"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    guests = df[["datetime", "visitors"]].copy()
    guests["timestamp_local"] = pd.to_datetime(guests["datetime"], errors="coerce")
    guests["timestamp_local"] = guests["timestamp_local"].dt.tz_localize(
        LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    guests["guest_count"] = pd.to_numeric(guests["visitors"], errors="coerce")
    guests = guests.dropna(subset=["timestamp_local", "guest_count"])
    guests["timestamp_local"] = guests["timestamp_local"].dt.floor("h")
    return guests.groupby("timestamp_local", as_index=False)["guest_count"].mean()


def read_raw_metric_hourly(
    csv_path: str | Path,
    metrics: Iterable[str],
    chunksize: int = 250_000,
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
        grouped = chunk.groupby(
            ["timestamp_local", "meter", "area", "metric"], as_index=False
        )["value"].mean()
        chunks.append(grouped)

    if not chunks:
        return pd.DataFrame(
            columns=["timestamp_local", "meter", "area", "metric", "value"]
        )

    data = pd.concat(chunks, ignore_index=True)
    return data.groupby(["timestamp_local", "meter", "area", "metric"], as_index=False)[
        "value"
    ].mean()


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
    for label, value in [
        ("data_kwh", paths.kwh_csv),
        ("guests", paths.guests_csv),
        ("energy_log", paths.energy_log_csv),
        ("data_pf", paths.pf_csv),
        ("data_current", paths.current_csv),
        ("data_2026", paths.telemetry_csv),
    ]:
        if value:
            summary = summarize_csv(value)
            summary["label"] = label
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
