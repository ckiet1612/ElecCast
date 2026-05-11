from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_feature_table_from_files, feature_summary
from .models import forecast_dataframe, metrics_to_frame, train_models
from .types import ForecastRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and forecast hourly electricity consumption."
    )
    parser.add_argument("--kwh", required=True, help="Path to data_kwh.csv")
    parser.add_argument("--guests", help="Path to hourly visitors CSV")
    parser.add_argument("--energy-log", help="Path to energy_log.csv")
    parser.add_argument("--pf", help="Path to data_pf.csv")
    parser.add_argument("--current", help="Path to data_current.csv")
    parser.add_argument("--telemetry", help="Path to data_2026.csv")
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
        "--guest-count", type=float, help="Override simulated guest count."
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
        kwh_csv=args.kwh,
        guests_csv=args.guests,
        energy_log_csv=args.energy_log,
        pf_csv=args.pf,
        current_csv=args.current,
        telemetry_csv=args.telemetry,
    )
    print("Feature summary:", feature_summary(features))
    trained, all_metrics = train_models(features, meters=args.meter)
    meters = args.meter or list(trained.keys())
    forecast = forecast_dataframe(
        trained,
        features,
        ForecastRequest(
            meters=meters,
            horizon_hours=args.horizon,
            temperature_c=args.temperature_c,
            guest_count=args.guest_count,
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
