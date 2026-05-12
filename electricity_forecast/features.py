from __future__ import annotations

from pathlib import Path

from .data import (
    default_guest_count,
    default_temperature,
    read_telemetry_hourly_features,
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
    import pandas as pd

    features = read_telemetry_hourly_features(paths.telemetry_csv)
    if features.empty:
        return _empty_feature_table()

    features["kwh"] = features["kwh_telemetry"]
    features["kwh_detection"] = features["kwh_telemetry"]
    features["kwh_source"] = "missing"
    features.loc[features["kwh_telemetry"].notna(), "kwh_source"] = "data_2026"

    for col, default in [("p", 0.0), ("pf", 0.95), ("iavg", 0.0)]:
        if col not in features:
            features[col] = default
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features.groupby("meter")[col].transform(
            lambda s: s.ffill().bfill()
        )
        features[col] = features[col].fillna(default)

    features = add_time_features(features)
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


def build_feature_table_from_files(telemetry_csv: str | Path):
    paths = DataPaths(telemetry_csv=Path(telemetry_csv))
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

    if data["kwh"].notna().any():
        meter_median = grouped["kwh"].transform("median")
        global_median = data["kwh"].median()
    else:
        meter_median = 0.0
        global_median = 0.0
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


def _empty_feature_table():
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "timestamp_local",
            "meter",
            "area",
            "p",
            "pf",
            "iavg",
            "kwh_cumulative",
            "kwh_telemetry_raw_delta",
            "kwh_telemetry",
            "kwh_telemetry_issue",
            "kwh",
            "kwh_detection",
            "kwh_source",
            "minute",
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "temperature_c",
            "guest_count",
            "lag_1h",
            "lag_24h",
            "lag_168h",
            "rolling_24h",
            "rolling_168h",
        ]
    )
