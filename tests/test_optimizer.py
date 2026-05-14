"""Tests for the gradient descent electricity consumption optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from electricity_forecast.optimizer import (
    CONTROLLABLE_PARAMS,
    _build_base_feature_matrix,
    _extract_analytical_gradient,
    _project_constraints,
    _vectorized_predict_total,
    optimize_consumption,
)
from electricity_forecast.types import (
    MeterModelMetrics,
    OptimizationRequest,
    OptimizationResult,
    TrainedMeterModel,
)
from electricity_forecast.features import NUMERIC_FEATURE_COLUMNS


def _make_mock_model(meter="METER_A", area="AREA_A"):
    """Create a minimal trained model for testing."""
    feature_columns = list(NUMERIC_FEATURE_COLUMNS)

    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="Asia/Ho_Chi_Minh")
    hours = np.array([ts.hour for ts in timestamps])
    temps = 25.0 + 3 * np.sin((hours - 6) / 24 * 2 * np.pi) + np.random.normal(0, 0.5, n)
    guests = 80.0 + 10 * (hours >= 17).astype(float) + np.random.normal(0, 2, n)
    kwh = 5.0 + 0.3 * temps + 0.05 * guests + np.random.normal(0, 0.5, n)
    kwh = np.clip(kwh, 0, None)

    history = pd.DataFrame({
        "timestamp_local": timestamps,
        "meter": meter,
        "area": area,
        "kwh": kwh,
        "p": np.random.uniform(10, 50, n),
        "pf": np.random.uniform(0.85, 1.0, n),
        "iavg": np.random.uniform(5, 20, n),
        "vavg": np.random.uniform(380, 420, n),
    })

    X_data = pd.DataFrame({
        "hour": hours,
        "day_of_week": [ts.dayofweek for ts in timestamps],
        "day_of_month": [ts.day for ts in timestamps],
        "month": [ts.month for ts in timestamps],
        "is_weekend": (np.array([ts.dayofweek for ts in timestamps]) >= 5).astype(int),
        "p": history["p"],
        "pf": history["pf"],
        "iavg": history["iavg"],
        "vavg": history["vavg"],
        "temperature_c": temps,
        "guest_count": guests,
        "lag_1h": np.roll(kwh, 1),
        "lag_24h": np.roll(kwh, 24),
        "lag_168h": np.roll(kwh, 168),
        "rolling_24h": pd.Series(kwh).rolling(24, min_periods=1).mean().values,
        "rolling_168h": pd.Series(kwh).rolling(168, min_periods=1).mean().values,
    })

    estimator = make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())
    estimator.fit(X_data, kwh)

    fill_values = {col: float(X_data[col].median()) for col in feature_columns}
    metrics = MeterModelMetrics(
        meter=meter,
        model_name="LinearRegression",
        mae=0.5,
        rmse=0.6,
        r2=0.85,
        rows_train=160,
        rows_test=40,
    )

    return TrainedMeterModel(
        meter=meter,
        area=area,
        model_name="LinearRegression",
        kind="sklearn",
        feature_columns=feature_columns,
        estimator=estimator,
        fill_values=fill_values,
        metrics=metrics,
        residual_std=0.5,
        history=history,
        metadata={},
    )


def _make_feature_table():
    """Create a minimal feature table for testing."""
    n = 200
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="Asia/Ho_Chi_Minh")
    return pd.DataFrame({
        "timestamp_local": timestamps,
        "meter": "METER_A",
        "area": "AREA_A",
        "kwh": np.random.uniform(5, 15, n),
    })


class TestProjectConstraints:
    def test_clamps_temperature(self):
        params = np.array([[15.0, 50.0], [35.0, 50.0]])
        request = OptimizationRequest(temp_min=20.0, temp_max=30.0)
        result = _project_constraints(params, request, guest_max=100.0)
        assert result[0, 0] == 20.0
        assert result[1, 0] == 30.0

    def test_clamps_guest_count(self):
        params = np.array([[25.0, -10.0], [25.0, 200.0]])
        request = OptimizationRequest(guest_min=0.0)
        result = _project_constraints(params, request, guest_max=150.0)
        assert result[0, 1] == 0.0
        assert result[1, 1] == 150.0

    def test_within_bounds_unchanged(self):
        params = np.array([[25.0, 80.0]])
        request = OptimizationRequest(temp_min=20.0, temp_max=30.0, guest_min=0.0)
        result = _project_constraints(params, request, guest_max=150.0)
        np.testing.assert_array_almost_equal(result, params)


class TestVectorizedPredict:
    def test_cost_is_positive(self):
        model = _make_mock_model()
        timestamps = pd.date_range(
            "2026-01-10", periods=6, freq="h", tz="Asia/Ho_Chi_Minh"
        )
        from electricity_forecast.data import default_temperature, default_guest_count
        default_temps = np.array([default_temperature(ts) for ts in timestamps])
        default_guests = np.array([default_guest_count(ts, "AREA_A") for ts in timestamps])
        base_X, t_idx, g_idx = _build_base_feature_matrix(
            model, timestamps, default_temps, default_guests
        )
        params = np.column_stack([np.full(6, 27.0), np.full(6, 80.0)])
        cost = _vectorized_predict_total(model, base_X, params, t_idx, g_idx)
        assert cost > 0

    def test_lower_temp_lower_cost(self):
        model = _make_mock_model()
        timestamps = pd.date_range(
            "2026-01-10", periods=6, freq="h", tz="Asia/Ho_Chi_Minh"
        )
        from electricity_forecast.data import default_temperature, default_guest_count
        default_temps = np.array([default_temperature(ts) for ts in timestamps])
        default_guests = np.array([default_guest_count(ts, "AREA_A") for ts in timestamps])
        base_X, t_idx, g_idx = _build_base_feature_matrix(
            model, timestamps, default_temps, default_guests
        )
        params_high = np.column_stack([np.full(6, 30.0), np.full(6, 80.0)])
        params_low = np.column_stack([np.full(6, 22.0), np.full(6, 80.0)])
        cost_high = _vectorized_predict_total(model, base_X, params_high, t_idx, g_idx)
        cost_low = _vectorized_predict_total(model, base_X, params_low, t_idx, g_idx)
        assert cost_low < cost_high


class TestAnalyticalGradient:
    def test_gradient_extracted(self):
        model = _make_mock_model()
        t_idx = model.feature_columns.index("temperature_c")
        g_idx = model.feature_columns.index("guest_count")
        grad_temp, grad_guest = _extract_analytical_gradient(model, t_idx, g_idx)
        assert grad_temp is not None
        assert grad_guest is not None
        assert abs(grad_temp) > 1e-8, "Temperature gradient should be non-zero"
        assert abs(grad_guest) > 1e-8, "Guest gradient should be non-zero"

    def test_temperature_gradient_positive(self):
        """Model was trained with positive temp coefficient (kwh = 0.3*temp + ...)."""
        model = _make_mock_model()
        t_idx = model.feature_columns.index("temperature_c")
        g_idx = model.feature_columns.index("guest_count")
        grad_temp, _ = _extract_analytical_gradient(model, t_idx, g_idx)
        assert grad_temp > 0, "Higher temp should increase kWh"


class TestOptimizeConsumption:
    def test_basic_optimization(self):
        model = _make_mock_model()
        models = {"METER_A": model}
        feature_table = _make_feature_table()
        request = OptimizationRequest(
            meters=["METER_A"],
            horizon_hours=6,
            max_iterations=20,
            learning_rate=0.05,
        )
        schedule_df, summary_df, results = optimize_consumption(
            models, feature_table, request
        )
        assert len(results) == 1
        result = results[0]
        assert result.meter == "METER_A"
        assert result.total_kwh_before > 0
        assert result.total_kwh_after > 0
        assert result.iterations > 0
        assert len(result.cost_history) > 1

    def test_cost_decreases(self):
        model = _make_mock_model()
        models = {"METER_A": model}
        feature_table = _make_feature_table()
        request = OptimizationRequest(
            meters=["METER_A"],
            horizon_hours=6,
            max_iterations=50,
            learning_rate=0.05,
        )
        _, _, results = optimize_consumption(models, feature_table, request)
        cost_h = results[0].cost_history
        assert cost_h[-1] <= cost_h[0], (
            f"Cost should not increase overall: start={cost_h[0]}, end={cost_h[-1]}"
        )

    def test_schedule_has_correct_columns(self):
        model = _make_mock_model()
        models = {"METER_A": model}
        feature_table = _make_feature_table()
        request = OptimizationRequest(
            meters=["METER_A"],
            horizon_hours=6,
            max_iterations=10,
        )
        schedule_df, _, _ = optimize_consumption(models, feature_table, request)
        expected_cols = {
            "timestamp_local", "meter", "area",
            "temperature_c_before", "temperature_c_after",
            "guest_count_before", "guest_count_after",
            "kwh_before", "kwh_after", "kwh_saved",
        }
        assert expected_cols.issubset(set(schedule_df.columns))

    def test_summary_df_structure(self):
        model = _make_mock_model()
        models = {"METER_A": model}
        feature_table = _make_feature_table()
        request = OptimizationRequest(
            meters=["METER_A"],
            horizon_hours=6,
            max_iterations=10,
        )
        _, summary_df, _ = optimize_consumption(models, feature_table, request)
        assert len(summary_df) == 1
        assert "savings_kwh" in summary_df.columns
        assert "savings_percent" in summary_df.columns

    def test_no_models_raises(self):
        feature_table = _make_feature_table()
        with pytest.raises(ValueError, match="No trained models"):
            optimize_consumption({}, feature_table, OptimizationRequest())

    def test_savings_non_negative(self):
        model = _make_mock_model()
        models = {"METER_A": model}
        feature_table = _make_feature_table()
        request = OptimizationRequest(
            meters=["METER_A"],
            horizon_hours=6,
            max_iterations=30,
            learning_rate=0.05,
        )
        _, _, results = optimize_consumption(models, feature_table, request)
        assert results[0].savings_kwh >= 0
