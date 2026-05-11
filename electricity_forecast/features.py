from __future__ import annotations

from pathlib import Path

from .data import (
    default_guest_count,
    default_temperature,
    pivot_metric_features,
    read_cumulative_kwh_hourly,
    read_guest_counts,
    read_kwh_target,
    read_raw_metric_hourly,
)
from .types import DataPaths


NUMERIC_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "p",
    "pf",
    "iavg",
    "temperature_c",
    "guest_count",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_24h",
    "rolling_168h",
]


def build_feature_table(paths: DataPaths):
    target = read_kwh_target(paths.kwh_csv)
    feature_frames = []
    telemetry_kwh = None
    if paths.telemetry_csv:
        telemetry = read_raw_metric_hourly(paths.telemetry_csv, ["P", "PF", "IAVG"])
        feature_frames.append(pivot_metric_features(telemetry))
        telemetry_kwh = read_cumulative_kwh_hourly(paths.telemetry_csv)
    if paths.pf_csv:
        pf = read_raw_metric_hourly(paths.pf_csv, ["PF"])
        feature_frames.append(pivot_metric_features(pf))
    if paths.current_csv:
        current = read_raw_metric_hourly(paths.current_csv, ["IAVG"])
        feature_frames.append(pivot_metric_features(current))

    features = target.copy()
    for frame in feature_frames:
        if frame.empty:
            continue
        merge_cols = ["timestamp_local", "meter", "area"]
        features = features.merge(
            frame, on=merge_cols, how="left", suffixes=("", "_new")
        )
        for col in ["p", "pf", "iavg"]:
            new_col = f"{col}_new"
            if new_col in features:
                if col in features:
                    features[col] = features[col].combine_first(features[new_col])
                else:
                    features[col] = features[new_col]
                features = features.drop(columns=[new_col])

    for col, default in [("p", 0.0), ("pf", 0.95), ("iavg", 0.0)]:
        if col not in features:
            features[col] = default
        features[col] = features.groupby("meter")[col].transform(
            lambda s: s.ffill().bfill()
        )
        features[col] = features[col].fillna(default)

    if paths.guests_csv:
        guests = read_guest_counts(paths.guests_csv)
        features = features.merge(guests, on="timestamp_local", how="left")

    features = add_time_features(features)
    features = add_detection_kwh(features, telemetry_kwh)
    features["temperature_c"] = features["timestamp_local"].map(default_temperature)
    simulated_guests = features.apply(
        lambda row: default_guest_count(row["timestamp_local"], row["area"]), axis=1
    )
    if "guest_count" in features:
        features["guest_count"] = features["guest_count"].fillna(simulated_guests)
    else:
        features["guest_count"] = simulated_guests
    features = add_lag_features(features)
    return features.sort_values(["meter", "timestamp_local"]).reset_index(drop=True)


def add_detection_kwh(df, telemetry_kwh):
    import pandas as pd

    data = df.copy()
    if telemetry_kwh is not None and not telemetry_kwh.empty:
        data = data.merge(
            telemetry_kwh,
            on=["timestamp_local", "meter", "area"],
            how="left",
        )
    for column in ["kwh_cumulative", "kwh_telemetry_raw_delta", "kwh_telemetry"]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "kwh_telemetry_issue" not in data:
        data["kwh_telemetry_issue"] = ""
    data["kwh_telemetry_issue"] = data["kwh_telemetry_issue"].fillna("")
    data["kwh_detection"] = data["kwh_telemetry"].combine_first(data["kwh"])
    data["kwh_source"] = "missing"
    data.loc[data["kwh"].notna(), "kwh_source"] = "data_kwh"
    data.loc[data["kwh_telemetry"].notna(), "kwh_source"] = "data_2026"
    return data


def build_feature_table_from_files(
    kwh_csv: str | Path,
    guests_csv: str | Path | None = None,
    energy_log_csv: str | Path | None = None,
    pf_csv: str | Path | None = None,
    current_csv: str | Path | None = None,
    telemetry_csv: str | Path | None = None,
):
    paths = DataPaths(
        kwh_csv=Path(kwh_csv),
        guests_csv=Path(guests_csv) if guests_csv else None,
        energy_log_csv=Path(energy_log_csv) if energy_log_csv else None,
        pf_csv=Path(pf_csv) if pf_csv else None,
        current_csv=Path(current_csv) if current_csv else None,
        telemetry_csv=Path(telemetry_csv) if telemetry_csv else None,
    )
    return build_feature_table(paths)


def add_time_features(df):
    data = df.copy()
    ts = data["timestamp_local"]
    data["minute"] = ts.dt.minute
    data["hour"] = ts.dt.hour
    data["day_of_week"] = ts.dt.dayofweek
    data["day_of_month"] = ts.dt.day
    data["month"] = ts.dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    return data


def add_lag_features(df):
    data = df.sort_values(["meter", "timestamp_local"]).copy()
    grouped = data.groupby("meter", group_keys=False)
    data["lag_1h"] = grouped["kwh"].shift(1)
    data["lag_24h"] = grouped["kwh"].shift(24)
    data["lag_168h"] = grouped["kwh"].shift(168)
    data["rolling_24h"] = grouped["kwh"].transform(
        lambda s: s.shift(1).rolling(24, min_periods=1).mean()
    )
    data["rolling_168h"] = grouped["kwh"].transform(
        lambda s: s.shift(1).rolling(168, min_periods=1).mean()
    )

    meter_median = grouped["kwh"].transform("median")
    global_median = data["kwh"].median()
    for col in ["lag_1h", "lag_24h", "lag_168h", "rolling_24h", "rolling_168h"]:
        data[col] = data[col].fillna(meter_median).fillna(global_median).fillna(0.0)
    return data


def clean_training_frame(df):
    data = df.copy()
    data = data.dropna(subset=["kwh", "timestamp_local", "meter"])
    data = data[data["kwh"] >= 0]
    for meter, index in data.groupby("meter").groups.items():
        values = data.loc[index, "kwh"]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        upper = q3 + 6 * iqr
        data.loc[index, "kwh"] = values.clip(lower=0, upper=upper)
    return data


def feature_summary(df) -> dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "meters": 0,
            "min_time": None,
            "max_time": None,
            "missing_kwh": 0,
        }
    return {
        "rows": int(len(df)),
        "meters": int(df["meter"].nunique()),
        "areas": int(df["area"].nunique()),
        "min_time": str(df["timestamp_local"].min()),
        "max_time": str(df["timestamp_local"].max()),
        "missing_kwh": int(df["kwh"].isna().sum()),
        "missing_kwh_detection": int(df["kwh_detection"].isna().sum())
        if "kwh_detection" in df
        else 0,
        "columns": list(df.columns),
    }
