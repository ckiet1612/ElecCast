from __future__ import annotations

import math

import pandas as pd

from electricity_forecast.anomaly import detect_anomalies
from electricity_forecast.features import add_lag_features, add_time_features
from electricity_forecast.features import build_feature_table_from_files
from electricity_forecast.models import forecast_dataframe, train_models
from electricity_forecast.types import AnomalyRequest, ForecastRequest, LOCAL_TIMEZONE


def synthetic_features(meters=("FB2_MSB01", "SHOW_MSB01"), periods=240):
    rows = []
    timestamps = pd.date_range(
        "2026-01-01 00:00", periods=periods, freq="h", tz=LOCAL_TIMEZONE
    )
    for meter in meters:
        area = meter.split("_MSB", 1)[0]
        base = 80 if meter.startswith("FB2") else 120
        for idx, timestamp in enumerate(timestamps):
            value = (
                base
                + 20 * math.sin(timestamp.hour / 24 * 2 * math.pi)
                + (idx % 7) * 1.5
            )
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
        ForecastRequest(
            meters=list(models.keys()), horizon_hours=168, temperature_c=29.0
        ),
    )
    assert not metrics.empty
    assert len(models) == 2
    assert len(forecast) == 2 * 168
    assert forecast["predicted_kwh"].ge(0).all()
    assert forecast["guest_count"].notna().all()


def test_build_feature_table_uses_only_data2026(tmp_path):
    telemetry_path = tmp_path / "data_2026.csv"
    telemetry_path.write_text(
        "time,name,original_value_float\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,100\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_P.value.PVLAST,50\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_PF.value.PVLAST,0.95\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_IAVG.value.PVLAST,20\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,112\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_P.value.PVLAST,55\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_PF.value.PVLAST,0.96\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_IAVG.value.PVLAST,22\n",
        encoding="utf-8",
    )
    df = build_feature_table_from_files(telemetry_path)
    rows = df.sort_values("timestamp_local").reset_index(drop=True)
    assert pd.isna(rows.iloc[0]["kwh"])
    assert rows.iloc[0]["kwh_source"] == "missing"
    assert rows.iloc[0]["p"] == 50
    assert rows.iloc[0]["pf"] == 0.95
    assert rows.iloc[0]["iavg"] == 20
    assert rows.iloc[0]["guest_count"] > 0
    assert rows.iloc[1]["kwh"] == 12
    assert rows.iloc[1]["kwh_detection"] == 12
    assert rows.iloc[1]["kwh_source"] == "data_2026"


def test_detect_anomalies_returns_scores_and_reasons():
    features = synthetic_features(meters=("FB2_MSB01",), periods=80)
    features["kwh_detection"] = features["kwh"]
    features["kwh_source"] = "data_2026"
    features["kwh_telemetry_issue"] = ""
    target_index = features.index[-1]
    features.loc[target_index, "kwh_detection"] = features["kwh"].median() * 12
    features.loc[target_index, "pf"] = 0.5
    features.loc[target_index, "iavg"] = features["iavg"].median() * 8
    features.loc[target_index, "guest_count"] = 1
    features.loc[target_index, "kwh_telemetry_issue"] = "kwh_delta_outlier"
    result = detect_anomalies(
        features,
        AnomalyRequest(meters=["FB2_MSB01"], contamination=0.05),
    )
    assert not result.empty
    assert result["anomaly_score"].notna().all()
    assert result["is_anomaly"].dtype == bool
    assert set(result["severity"]).issubset({"Normal", "Medium", "High"})
    anomalies = result[result["is_anomaly"]]
    assert not anomalies.empty
    assert anomalies["reason"].str.contains("KWH telemetry delta outlier").any()


def test_detect_anomalies_keeps_invalid_data2026_delta(tmp_path):
    telemetry_path = tmp_path / "data_2026.csv"
    telemetry_path.write_text(
        "time,name,original_value_float\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,100\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_P.value.PVLAST,40\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_PF.value.PVLAST,0.94\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_IAVG.value.PVLAST,10\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,90\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_P.value.PVLAST,40\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_PF.value.PVLAST,0.94\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_IAVG.value.PVLAST,10\n",
        encoding="utf-8",
    )
    features = build_feature_table_from_files(telemetry_path)
    result = detect_anomalies(
        features,
        AnomalyRequest(meters=["FB2_MSB01"], contamination=0.05),
    )
    assert not result.empty
    assert result["is_anomaly"].any()
    assert result["reason"].str.contains("KWH telemetry reset/negative delta").any()
