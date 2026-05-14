"""Electricity consumption optimization using Gradient Descent.

This module implements a projected gradient descent optimizer that finds
optimal controllable parameters (temperature setpoint, guest scheduling)
to minimize predicted kWh consumption while respecting operational
constraints.

Performance strategy:
- Pre-build the full feature matrix once (only the base/fixed columns).
- Extract analytical gradient directly from LinearRegression/Ridge
  coefficients — no finite differences needed.
- Vectorized numpy operations throughout — no per-hour loops during GD.
"""
from __future__ import annotations

from .data import default_guest_count, default_temperature
from .types import LOCAL_TIMEZONE, OptimizationRequest, OptimizationResult, TrainedMeterModel


CONTROLLABLE_PARAMS = ["temperature_c", "guest_count"]
EPSILON = 1e-4


def optimize_consumption(
    models: dict[str, TrainedMeterModel],
    feature_table,
    request: OptimizationRequest | None = None,
):
    """Run gradient descent optimization for each selected meter.

    Returns a tuple of (schedule_dataframe, summary_dataframe, list_of_results).
    """
    import pandas as pd

    request = request or OptimizationRequest()
    if not models:
        raise ValueError("No trained models available. Train models before optimizing.")

    meters = request.meters or list(models.keys())
    start = _resolve_start_time(feature_table, request.start_time)
    timestamps = pd.date_range(start=start, periods=request.horizon_hours, freq="h")

    all_results: list[OptimizationResult] = []
    all_schedule_rows: list[dict] = []

    for meter in meters:
        if meter not in models:
            continue
        model = models[meter]
        if model.kind == "arima" or model.estimator is None:
            continue

        result = _gradient_descent_meter(model, timestamps, request)
        all_results.append(result)
        all_schedule_rows.extend(result.schedule)

    schedule_df = pd.DataFrame(all_schedule_rows)
    if not schedule_df.empty:
        schedule_df = schedule_df.sort_values(
            ["meter", "timestamp_local"]
        ).reset_index(drop=True)

    summary_df = pd.DataFrame([
        {
            "meter": r.meter,
            "area": r.area,
            "total_kwh_before": round(r.total_kwh_before, 3),
            "total_kwh_after": round(r.total_kwh_after, 3),
            "savings_kwh": round(r.savings_kwh, 3),
            "savings_percent": round(r.savings_percent, 2),
            "iterations": r.iterations,
            "converged": r.converged,
        }
        for r in all_results
    ])

    return schedule_df, summary_df, all_results


def _gradient_descent_meter(
    model: TrainedMeterModel,
    timestamps,
    request: OptimizationRequest,
) -> OptimizationResult:
    """Run projected gradient descent for a single meter.

    Uses analytical gradients from model coefficients and vectorized
    numpy operations for fast convergence.
    """
    import numpy as np

    n = len(timestamps)
    guest_max = _resolve_guest_max(model, request)

    default_temps = np.array(
        [default_temperature(ts) for ts in timestamps], dtype=float
    )
    default_guests = np.array(
        [default_guest_count(ts, model.area) for ts in timestamps], dtype=float
    )

    base_X, temp_col_idx, guest_col_idx = _build_base_feature_matrix(
        model, timestamps, default_temps, default_guests
    )

    params = np.column_stack([default_temps.copy(), default_guests.copy()])

    total_kwh_before = _vectorized_predict_total(
        model, base_X, params, temp_col_idx, guest_col_idx
    )

    params = _project_constraints(params, request, guest_max)

    grad_temp, grad_guest = _extract_analytical_gradient(
        model, temp_col_idx, guest_col_idx
    )
    use_analytical = grad_temp is not None

    current_cost = _vectorized_predict_total(
        model, base_X, params, temp_col_idx, guest_col_idx
    )
    cost_history = [float(current_cost)]

    converged = False
    iteration = 0

    for iteration in range(1, request.max_iterations + 1):
        if use_analytical:
            gradient = np.column_stack([
                np.full(n, grad_temp),
                np.full(n, grad_guest),
            ])
        else:
            gradient = _numerical_gradient(
                model, base_X, params, temp_col_idx, guest_col_idx,
                request, guest_max,
            )

        params = params - request.learning_rate * gradient
        params = _project_constraints(params, request, guest_max)

        new_cost = _vectorized_predict_total(
            model, base_X, params, temp_col_idx, guest_col_idx
        )
        cost_history.append(float(new_cost))

        if abs(current_cost - new_cost) < request.convergence_threshold:
            converged = True
            break

        current_cost = new_cost

    total_kwh_after = _vectorized_predict_total(
        model, base_X, params, temp_col_idx, guest_col_idx
    )
    savings_kwh = max(total_kwh_before - total_kwh_after, 0.0)
    savings_pct = (savings_kwh / total_kwh_before * 100) if total_kwh_before > 0 else 0.0

    schedule = _build_schedule_vectorized(
        model, base_X, params, timestamps,
        default_temps, default_guests, temp_col_idx, guest_col_idx,
    )

    return OptimizationResult(
        meter=model.meter,
        area=model.area,
        total_kwh_before=total_kwh_before,
        total_kwh_after=total_kwh_after,
        savings_kwh=savings_kwh,
        savings_percent=savings_pct,
        iterations=iteration,
        converged=converged,
        schedule=schedule,
        cost_history=cost_history,
    )


def _build_base_feature_matrix(model, timestamps, default_temps, default_guests):
    """Pre-build the full feature matrix once, returning column indices
    for the controllable parameters.

    Returns (X_matrix_as_ndarray, temp_col_index, guest_col_index).
    """
    import numpy as np
    import pandas as pd

    history = model.history.copy().sort_values("timestamp_local")
    hist_kwh = history["kwh"].astype(float).values
    hist_timestamps = history["timestamp_local"].values
    meter_median = float(np.nanmedian(hist_kwh)) if len(hist_kwh) > 0 else 0.0

    value_by_ts = {
        pd.Timestamp(ts): float(v) for ts, v in zip(hist_timestamps, hist_kwh)
    }
    values = list(hist_kwh)

    p = _recent_median(history, "p", 0.0)
    pf = _recent_median(history, "pf", 0.95)
    iavg = _recent_median(history, "iavg", 0.0)
    vavg = _recent_median(history, "vavg", 400.0)

    rows = []
    for idx, ts in enumerate(timestamps):
        lag_1h = value_by_ts.get(pd.Timestamp(ts) - pd.Timedelta(hours=1), meter_median)
        lag_24h = value_by_ts.get(pd.Timestamp(ts) - pd.Timedelta(hours=24), lag_1h)
        lag_168h = value_by_ts.get(pd.Timestamp(ts) - pd.Timedelta(hours=168), lag_24h)
        recent_24 = values[-24:] if values else [meter_median]
        recent_168 = values[-168:] if values else [meter_median]

        rows.append({
            "hour": int(ts.hour),
            "day_of_week": int(ts.dayofweek),
            "day_of_month": int(ts.day),
            "month": int(ts.month),
            "is_weekend": int(ts.dayofweek >= 5),
            "p": p,
            "pf": pf,
            "iavg": iavg,
            "vavg": vavg,
            "temperature_c": float(default_temps[idx]),
            "guest_count": float(default_guests[idx]),
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_168h": lag_168h,
            "rolling_24h": float(np.mean(recent_24)),
            "rolling_168h": float(np.mean(recent_168)),
        })

        values.append(meter_median)
        value_by_ts[pd.Timestamp(ts)] = meter_median

    df = pd.DataFrame(rows, columns=model.feature_columns)
    for col, val in model.fill_values.items():
        if col in df:
            df[col] = df[col].fillna(val)

    temp_col_idx = model.feature_columns.index("temperature_c") if "temperature_c" in model.feature_columns else None
    guest_col_idx = model.feature_columns.index("guest_count") if "guest_count" in model.feature_columns else None

    return df.values.astype(float), temp_col_idx, guest_col_idx


def _vectorized_predict_total(model, base_X, params, temp_col_idx, guest_col_idx):
    """Predict total kWh using vectorized numpy operations.

    Copies base_X, overwrites controllable columns, runs batch predict.
    """
    import numpy as np

    X = base_X.copy()
    if temp_col_idx is not None:
        X[:, temp_col_idx] = params[:, 0]
    if guest_col_idx is not None:
        X[:, guest_col_idx] = params[:, 1]

    preds = model.estimator.predict(X)
    preds = np.clip(preds, 0.0, None)
    return float(np.sum(preds))


def _extract_analytical_gradient(model, temp_col_idx, guest_col_idx):
    """Extract gradient directly from linear model coefficients.

    For LinearRegression/Ridge inside a Pipeline, the gradient of
    sum(predictions) w.r.t. a controllable feature is simply
    n * (scaler_scale * coef) mapped back through the pipeline.

    Returns (grad_per_unit_temp, grad_per_unit_guest) or (None, None)
    if the model is not a supported linear type.
    """
    try:
        estimator = model.estimator
        steps = estimator.steps if hasattr(estimator, "steps") else None
        if steps is None:
            return None, None

        final_model = steps[-1][1]
        coefs = getattr(final_model, "coef_", None)
        if coefs is None:
            return None, None

        scaler = None
        for _, step in steps:
            if hasattr(step, "scale_"):
                scaler = step
                break

        grad_temp = None
        grad_guest = None

        if temp_col_idx is not None:
            raw_coef = float(coefs[temp_col_idx])
            if scaler is not None and scaler.scale_ is not None:
                scale = float(scaler.scale_[temp_col_idx])
                grad_temp = raw_coef / scale if scale > 0 else 0.0
            else:
                grad_temp = raw_coef

        if guest_col_idx is not None:
            raw_coef = float(coefs[guest_col_idx])
            if scaler is not None and scaler.scale_ is not None:
                scale = float(scaler.scale_[guest_col_idx])
                grad_guest = raw_coef / scale if scale > 0 else 0.0
            else:
                grad_guest = raw_coef

        return grad_temp, grad_guest
    except Exception:
        return None, None


def _numerical_gradient(model, base_X, params, temp_col_idx, guest_col_idx,
                         request, guest_max):
    """Fallback: compute gradient using central finite differences (vectorized)."""
    import numpy as np

    gradient = np.zeros_like(params)

    for j in range(params.shape[1]):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[:, j] += EPSILON
        params_minus[:, j] -= EPSILON
        params_plus = _project_constraints(params_plus, request, guest_max)
        params_minus = _project_constraints(params_minus, request, guest_max)

        cost_plus = _vectorized_predict_total(
            model, base_X, params_plus, temp_col_idx, guest_col_idx
        )
        cost_minus = _vectorized_predict_total(
            model, base_X, params_minus, temp_col_idx, guest_col_idx
        )
        gradient[:, j] = (cost_plus - cost_minus) / (2 * EPSILON)

    return gradient


def _project_constraints(params, request, guest_max):
    """Project parameters onto the feasible constraint set (clamping)."""
    import numpy as np

    projected = params.copy()
    projected[:, 0] = np.clip(projected[:, 0], request.temp_min, request.temp_max)
    projected[:, 1] = np.clip(projected[:, 1], request.guest_min, guest_max)
    return projected


def _build_schedule_vectorized(
    model, base_X, params, timestamps,
    default_temps, default_guests, temp_col_idx, guest_col_idx,
):
    """Build the detailed hourly schedule using vectorized predictions."""
    import numpy as np

    X_opt = base_X.copy()
    if temp_col_idx is not None:
        X_opt[:, temp_col_idx] = params[:, 0]
    if guest_col_idx is not None:
        X_opt[:, guest_col_idx] = params[:, 1]
    preds_opt = np.clip(model.estimator.predict(X_opt), 0.0, None)

    X_orig = base_X.copy()
    if temp_col_idx is not None:
        X_orig[:, temp_col_idx] = default_temps
    if guest_col_idx is not None:
        X_orig[:, guest_col_idx] = default_guests
    preds_orig = np.clip(model.estimator.predict(X_orig), 0.0, None)

    schedule = []
    for idx, ts in enumerate(timestamps):
        pred_orig = float(preds_orig[idx])
        pred_opt = float(preds_opt[idx])
        schedule.append({
            "timestamp_local": ts,
            "meter": model.meter,
            "area": model.area,
            "temperature_c_before": round(float(default_temps[idx]), 2),
            "temperature_c_after": round(float(params[idx, 0]), 2),
            "guest_count_before": round(float(default_guests[idx]), 1),
            "guest_count_after": round(float(params[idx, 1]), 1),
            "kwh_before": round(pred_orig, 4),
            "kwh_after": round(pred_opt, 4),
            "kwh_saved": round(max(pred_orig - pred_opt, 0.0), 4),
        })

    return schedule


def _resolve_guest_max(model, request):
    """Resolve the guest_count upper bound from request or history."""
    if request.guest_max is not None:
        return request.guest_max
    try:
        guests = model.history.get("guest_count")
        if guests is not None and not guests.dropna().empty:
            return float(guests.max()) * 1.2
    except Exception:
        pass
    return default_guest_count(
        model.history["timestamp_local"].iloc[-1], model.area
    ) * 1.5


def _resolve_start_time(feature_table, start_time):
    """Resolve the optimization start timestamp."""
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


def _recent_median(history, column: str, default: float) -> float:
    """Get the median of the most recent 168 values for a column."""
    if column not in history or history[column].dropna().empty:
        return default
    return float(history[column].tail(168).median())
