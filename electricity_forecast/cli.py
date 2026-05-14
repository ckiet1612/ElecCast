from __future__ import annotations

import argparse
from pathlib import Path

from .anomaly import detect_anomalies as run_anomaly_detection
from .features import build_feature_table_from_files, feature_summary
from .models import forecast_dataframe, metrics_to_frame, train_models
from .types import AnomalyRequest, ForecastRequest
from .weather import (
    default_weather_location_label,
    default_weather_month,
    monthly_average_temperature,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and forecast hourly electricity consumption."
    )
    parser.add_argument("--telemetry", required=True, help="Path to data_2026.csv")
    parser.add_argument(
        "--guests",
        help="Optional customer/visitor CSV with datetime and guest_count/visitors.",
    )
    parser.add_argument(
        "--meter",
        action="append",
        help="Meter to train/forecast. Repeat for multiple meters.",
    )
    parser.add_argument(
        "--horizon", type=int, default=168, help="Forecast horizon in hours."
    )
    parser.add_argument(
        "--temperature-c", type=float, help="Override simulated temperature."
    )
    parser.add_argument(
        "--weather-location",
        help="Location name or lat,long for Open-Meteo monthly average temperature.",
    )
    parser.add_argument(
        "--weather-month",
        help="Forecast weather month in YYYY-MM format for Open-Meteo.",
    )
    parser.add_argument(
        "--guest-count", type=float, help="Override simulated guest count."
    )
    parser.add_argument(
        "--detect-anomalies",
        action="store_true",
        help="Run Isolation Forest anomaly detection instead of forecast training.",
    )
    parser.add_argument(
        "--anomaly-meter",
        action="append",
        help="Meter to include in anomaly detection. Repeat for multiple meters.",
    )
    parser.add_argument(
        "--anomaly-contamination",
        type=float,
        default=0.05,
        help="Expected anomaly fraction for Isolation Forest.",
    )
    parser.add_argument(
        "--anomaly-output",
        default="exports/anomalies.csv",
        help="Anomaly CSV output path.",
    )
    parser.add_argument(
        "--anomaly-max-rows",
        type=int,
        help="Optional max rows to analyze for faster CLI smoke runs.",
    )
    parser.add_argument(
        "--output", default="exports/forecast.csv", help="Forecast CSV output path."
    )
    parser.add_argument(
        "--metrics-output",
        default="exports/metrics.csv",
        help="Metrics CSV output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = build_feature_table_from_files(
        telemetry_csv=args.telemetry,
        guests_csv=args.guests,
    )
    print("Feature summary:", feature_summary(features))
    if args.detect_anomalies:
        anomalies = run_anomaly_detection(
            features,
            AnomalyRequest(
                meters=args.anomaly_meter or args.meter,
                contamination=args.anomaly_contamination,
                max_rows=args.anomaly_max_rows,
            ),
        )
        anomaly_output = Path(args.anomaly_output)
        anomaly_output.parent.mkdir(parents=True, exist_ok=True)
        anomalies.to_csv(anomaly_output, index=False)
        print(f"Wrote anomalies: {anomaly_output}")
        print(anomalies[anomalies["is_anomaly"]].head(20).to_string(index=False))
        return 0

    trained, all_metrics = train_models(features, meters=args.meter)
    meters = args.meter or list(trained.keys())
    temperature_c = args.temperature_c
    weather_location = None
    weather_month = None
    if temperature_c is None and (args.weather_location or args.weather_month):
        weather = monthly_average_temperature(
            args.weather_location or default_weather_location_label(),
            args.weather_month or default_weather_month(),
        )
        temperature_c = weather.average_c
        weather_location = weather.location_label
        weather_month = weather.month
        print(
            f"Weather temperature: {temperature_c:.1f} C ({weather_location}, {weather_month})"
        )
    forecast = forecast_dataframe(
        trained,
        features,
        ForecastRequest(
            meters=meters,
            horizon_hours=args.horizon,
            temperature_c=temperature_c,
            guest_count=args.guest_count,
            weather_location=weather_location,
            weather_month=weather_month,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output, index=False)
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(metrics_output, index=False)
    print(f"Wrote forecast: {output}")
    print(f"Wrote metrics: {metrics_output}")
    print(metrics_to_frame(trained).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
