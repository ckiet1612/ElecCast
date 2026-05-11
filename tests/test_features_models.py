from __future__ import annotations

import math

import pandas as pd

from electricity_forecast.features import add_lag_features, add_time_features
from electricity_forecast.models import forecast_dataframe, train_models
from electricity_forecast.types import ForecastRequest, LOCAL_TIMEZONE


def synthetic_features(meters=("FB2_MSB01", "SHOW_MSB01"), periods=240):
    rows = []
    timestamps = pd.date_range("2026-01-01 00:00", periods=periods, freq="h", tz=LOCAL_TIMEZONE)
    for meter in meters:
        area = meter.split("_MSB", 1)[0]
        base = 80 if meter.startswith("FB2") else 120
        for idx, timestamp in enumerate(timestamps):
            value = base + 20 * math.sin(timestamp.hour / 24 * 2 * math.pi) + (idx % 7) * 1.5
            rows.append(
                {
                    "timestamp_local": timestamp,
                    "meter": meter,
                    "area": area,
                    "kwh": max(value, 0),
                    "p": base * 2,
                    "pf": 0.95,
                    "iavg": base / 3,
                    "temperature_c": 28,
                    "guest_count": 100,
                }
            )
    df = pd.DataFrame(rows)
    df = add_time_features(df)
    return add_lag_features(df)


def test_add_lag_features_has_expected_lag_24h():
    df = synthetic_features(meters=("FB2_MSB01",), periods=48)
    meter_df = df[df["meter"].eq("FB2_MSB01")].sort_values("timestamp_local")
    assert meter_df.iloc[24]["lag_24h"] == meter_df.iloc[0]["kwh"]


def test_train_and_forecast_generates_requested_rows():
    features = synthetic_features(periods=240)
    models, metrics = train_models(features, include_arima=False)
    forecast = forecast_dataframe(
        models,
        features,
        ForecastRequest(meters=list(models.keys()), horizon_hours=168, temperature_c=29.0),
    )
    assert not metrics.empty
    assert len(models) == 2
    assert len(forecast) == 2 * 168
    assert forecast["predicted_kwh"].ge(0).all()
