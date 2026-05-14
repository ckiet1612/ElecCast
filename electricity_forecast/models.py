from __future__ import annotations

import warnings
from pathlib import Path

from .data import default_guest_count, default_temperature
from .features import NUMERIC_FEATURE_COLUMNS, clean_training_frame
from .types import ForecastRequest, LOCAL_TIMEZONE, MeterModelMetrics, TrainedMeterModel


def train_models(
    feature_table,
    meters: list[str] | None = None,
    include_arima: bool = False,
    model_policy: str = "linear",
):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = clean_training_frame(feature_table)
    if meters:
        data = data[data["meter"].isin(meters)]
    if data.empty:
        raise ValueError("No training rows available.")

    feature_columns = [col for col in NUMERIC_FEATURE_COLUMNS if col in data.columns]
    trained: dict[str, TrainedMeterModel] = {}
    metric_rows: list[MeterModelMetrics] = []

    for meter, meter_df in data.groupby("meter"):
        meter_df = meter_df.sort_values("timestamp_local").reset_index(drop=True)
        if len(meter_df) < 30:
            continue
        split_idx = max(
            int(len(meter_df) * 0.8),
            len(meter_df) - min(168, max(8, len(meter_df) // 5)),
        )
        split_idx = min(max(split_idx, 8), len(meter_df) - 4)
        train_df = meter_df.iloc[:split_idx]
        test_df = meter_df.iloc[split_idx:]
        X_train = train_df[feature_columns]
        y_train = train_df["kwh"].astype(float)
        X_test = test_df[feature_columns]
        y_test = test_df["kwh"].astype(float)
        fill_values = X_train.median(numeric_only=True).fillna(0.0).to_dict()

        candidates = []
        baseline_pred = _baseline_predict(X_test, y_train.median())
        candidates.append(("SeasonalNaive", "baseline", None, baseline_pred))

        for name, estimator in [
            (
                "LinearRegression",
                make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression()),
            ),
            (
                "RidgeRegression",
                make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
            ),
        ]:
            model = estimator.fit(X_train, y_train)
            candidates.append((name, "sklearn", model, model.predict(X_test)))

        if include_arima:
            arima_model, arima_pred = _fit_arima(y_train, len(y_test))
            if arima_model is not None:
                candidates.append(("ARIMA", "arima", arima_model, arima_pred))

        scored = []
        for name, kind, estimator, pred in candidates:
            pred = np.asarray(pred, dtype=float)
            pred = np.clip(pred, 0.0, None)
            mae = float(mean_absolute_error(y_test, pred))
            rmse = float(mean_squared_error(y_test, pred) ** 0.5)
            r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0
            metrics = MeterModelMetrics(
                meter=meter,
                model_name=name,
                mae=mae,
                rmse=rmse,
                r2=r2,
                rows_train=len(train_df),
                rows_test=len(test_df),
            )
            metric_rows.append(metrics)
            scored.append((rmse, metrics, name, kind, estimator, pred))

        selection_pool = scored
        if model_policy == "linear":
            linear_pool = [item for item in scored if item[2] == "LinearRegression"]
            if linear_pool:
                selection_pool = linear_pool
        selection_pool.sort(key=lambda item: item[0])
        (
            _,
            selected_metrics,
            selected_name,
            selected_kind,
            selected_estimator,
            selected_backtest_pred,
        ) = selection_pool[0]

        if selected_kind == "baseline":
            final_estimator = None
            residual_pred = selected_backtest_pred
        elif selected_kind == "arima":
            final_estimator, residual_pred = _fit_arima(
                meter_df["kwh"].astype(float), len(test_df)
            )
            if final_estimator is None:
                final_estimator = None
                selected_name = "SeasonalNaive"
                selected_kind = "baseline"
                residual_pred = selected_backtest_pred
        else:
            final_estimator = selected_estimator.fit(
                meter_df[feature_columns], meter_df["kwh"].astype(float)
            )
            residual_pred = selected_backtest_pred

        residual_std = (
            float(np.std(np.asarray(y_test) - np.asarray(residual_pred)))
            if len(test_df)
            else 0.0
        )
        area = str(meter_df["area"].iloc[0])
        history_cols = [
            "timestamp_local",
            "meter",
            "area",
            "kwh",
            "p",
            "pf",
            "iavg",
            "vavg",
        ]
        history_cols = [col for col in history_cols if col in meter_df.columns]
        trained[meter] = TrainedMeterModel(
            meter=meter,
            area=area,
            model_name=selected_name,
            kind=selected_kind,
            feature_columns=feature_columns,
            estimator=final_estimator,
            fill_values=fill_values,
            metrics=selected_metrics,
            residual_std=residual_std,
            history=meter_df[history_cols].copy(),
            metadata={
                "last_timestamp": str(meter_df["timestamp_local"].max()),
                "backtest_rows": _backtest_rows(
                    test_df, selected_backtest_pred, selected_name
                ),
            },
        )

    metrics_df = pd.DataFrame([metric.__dict__ for metric in metric_rows])
    return trained, metrics_df


def forecast_dataframe(
    models: dict[str, TrainedMeterModel], feature_table, request: ForecastRequest
):
    import pandas as pd

    if not models:
        raise ValueError("No trained models available.")

    meters = request.meters or list(models.keys())
    start = _resolve_start_time(feature_table, request.start_time)
    timestamps = pd.date_range(start=start, periods=request.horizon_hours, freq="h")
    rows = []
    for meter in meters:
        if meter not in models:
            continue
        model = models[meter]
        rows.extend(_forecast_meter(model, timestamps, request))

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["meter", "timestamp_local"]).reset_index(drop=True)
    return result


def metrics_to_frame(models: dict[str, TrainedMeterModel]):
    import pandas as pd

    return pd.DataFrame([model.metrics.__dict__ for model in models.values()])


def backtest_predictions_dataframe(models: dict[str, TrainedMeterModel]):
    import pandas as pd

    rows = []
    for model in models.values():
        rows.extend(model.metadata.get("backtest_rows", []))
    columns = [
        "timestamp_local",
        "meter",
        "area",
        "actual_kwh",
        "predicted_kwh",
        "residual_kwh",
        "model_name",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["area", "meter", "timestamp_local"]
    )


def save_models(models: dict[str, TrainedMeterModel], path: str | Path) -> None:
    import joblib

    joblib.dump(models, path)


def load_models(path: str | Path) -> dict[str, TrainedMeterModel]:
    import joblib

    return joblib.load(path)


def _baseline_predict(X, fallback: float):
    import numpy as np

    values = None
    for column in ["lag_168h", "lag_24h", "lag_1h", "rolling_24h", "rolling_168h"]:
        if column not in X:
            continue
        candidate = X[column].astype(float).to_numpy()
        values = (
            candidate
            if values is None
            else np.where(np.isnan(values), candidate, values)
        )
    if values is None:
        values = np.full(len(X), float(fallback if fallback == fallback else 0.0))
    values = np.where(
        np.isnan(values), float(fallback if fallback == fallback else 0.0), values
    )
    return np.clip(values, 0.0, None)


def _backtest_rows(test_df, predictions, model_name: str) -> list[dict[str, object]]:
    rows = []
    for (_, row), pred in zip(test_df.iterrows(), predictions):
        actual = float(row["kwh"])
        predicted = max(float(pred), 0.0)
        rows.append(
            {
                "timestamp_local": row["timestamp_local"],
                "meter": row["meter"],
                "area": row["area"],
                "actual_kwh": actual,
                "predicted_kwh": predicted,
                "residual_kwh": actual - predicted,
                "model_name": model_name,
            }
        )
    return rows


def _fit_arima(y, steps: int):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return None, None

    if len(y) < 48:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y.astype(float), order=(1, 0, 1)).fit()
            forecast = model.forecast(steps=max(steps, 1))
        return model, forecast[:steps]
    except Exception:
        return None, None


def _resolve_start_time(feature_table, start_time):
    import pandas as pd

    if start_time is None:
        start = feature_table["timestamp_local"].max() + pd.Timedelta(hours=1)
    else:
        start = pd.Timestamp(start_time)
        if start.tzinfo is None:
            start = start.tz_localize(LOCAL_TIMEZONE)
        else:
            start = start.tz_convert(LOCAL_TIMEZONE)
    return start.floor("h")


def _forecast_meter(model: TrainedMeterModel, timestamps, request: ForecastRequest):
    import numpy as np
    import pandas as pd

    history = model.history.copy().sort_values("timestamp_local")
    rows = []
    if model.kind == "arima" and model.estimator is not None:
        try:
            preds = np.asarray(
                model.estimator.forecast(steps=len(timestamps)), dtype=float
            )
        except Exception:
            preds = None
        if preds is not None:
            for timestamp, pred in zip(timestamps, preds):
                pred = max(float(pred), 0.0)
                rows.append(_forecast_row(model, timestamp, pred, request))
            return rows

    values = list(history["kwh"].astype(float))
    timestamps_seen = list(history["timestamp_local"])
    value_by_ts = {
        pd.Timestamp(ts): float(value) for ts, value in zip(timestamps_seen, values)
    }
    meter_median = float(pd.Series(values).median()) if values else 0.0

    for timestamp in timestamps:
        row = _future_feature_row(
            model, history, value_by_ts, values, timestamp, request, meter_median
        )
        if model.kind == "baseline" or model.estimator is None:
            pred = _baseline_predict(pd.DataFrame([row]), meter_median)[0]
        else:
            X = pd.DataFrame([row], columns=model.feature_columns)
            for col, value in model.fill_values.items():
                if col in X:
                    X[col] = X[col].fillna(value)
            pred = model.estimator.predict(X)[0]
        pred = max(float(pred), 0.0)
        rows.append(
            _forecast_row(
                model,
                timestamp,
                pred,
                request,
                temperature_c=row["temperature_c"],
                guest_count=row["guest_count"],
            )
        )
        values.append(pred)
        value_by_ts[pd.Timestamp(timestamp)] = pred
    return rows


def _future_feature_row(
    model, history, value_by_ts, values, timestamp, request, fallback
):
    import pandas as pd

    p = _recent_median(history, "p", 0.0)
    pf = _recent_median(history, "pf", 0.95)
    iavg = _recent_median(history, "iavg", 0.0)
    vavg = _recent_median(history, "vavg", 400.0)
    lag_1h = value_by_ts.get(pd.Timestamp(timestamp) - pd.Timedelta(hours=1), fallback)
    lag_24h = value_by_ts.get(pd.Timestamp(timestamp) - pd.Timedelta(hours=24), lag_1h)
    lag_168h = value_by_ts.get(
        pd.Timestamp(timestamp) - pd.Timedelta(hours=168), lag_24h
    )
    recent_24 = values[-24:] if values else [fallback]
    recent_168 = values[-168:] if values else [fallback]
    return {
        "hour": int(timestamp.hour),
        "day_of_week": int(timestamp.dayofweek),
        "day_of_month": int(timestamp.day),
        "month": int(timestamp.month),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "p": p,
        "pf": pf,
        "iavg": iavg,
        "vavg": vavg,
        "temperature_c": _forecast_temperature(timestamp, request),
        "guest_count": _forecast_guest_count(model, timestamp, request),
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
        "lag_168h": lag_168h,
        "rolling_24h": float(pd.Series(recent_24).mean()),
        "rolling_168h": float(pd.Series(recent_168).mean()),
    }


def _recent_median(history, column: str, default: float) -> float:
    if column not in history or history[column].dropna().empty:
        return default
    return float(history[column].tail(168).median())


def _forecast_temperature(timestamp, request: ForecastRequest) -> float:
    return (
        float(request.temperature_c)
        if request.temperature_c is not None
        else default_temperature(timestamp)
    )


def _forecast_guest_count(
    model: TrainedMeterModel, timestamp, request: ForecastRequest
) -> float:
    if request.guest_count is not None:
        return float(request.guest_count)
    return default_guest_count(timestamp, model.area)


def _forecast_row(
    model: TrainedMeterModel,
    timestamp,
    pred: float,
    request: ForecastRequest,
    temperature_c: float | None = None,
    guest_count: float | None = None,
):
    residual = max(model.residual_std, 0.0)
    lower = max(pred - 1.96 * residual, 0.0) if residual else None
    upper = pred + 1.96 * residual if residual else None
    return {
        "timestamp_local": timestamp,
        "meter": model.meter,
        "area": model.area,
        "predicted_kwh": pred,
        "model_name": model.model_name,
        "lower_kwh": lower,
        "upper_kwh": upper,
        "temperature_c": (
            _forecast_temperature(timestamp, request)
            if temperature_c is None
            else float(temperature_c)
        ),
        "guest_count": _forecast_guest_count(model, timestamp, request)
        if guest_count is None
        else float(guest_count),
        "weather_location": request.weather_location,
        "weather_month": request.weather_month,
    }
