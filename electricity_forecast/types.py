from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"


@dataclass(frozen=True)
class ParsedName:
    raw_name: str
    meter: str
    area: str
    metric: str


@dataclass(frozen=True)
class RawReading:
    time_utc: datetime
    meter: str
    area: str
    metric: str
    value: float


@dataclass(frozen=True)
class DataPaths:
    telemetry_csv: Path
    guests_csv: Path | None = None


@dataclass(frozen=True)
class ForecastRequest:
    meters: list[str]
    horizon_hours: int
    start_time: datetime | None = None
    temperature_c: float | None = None
    guest_count: float | None = None
    weather_location: str | None = None
    weather_month: str | None = None


@dataclass(frozen=True)
class ForecastResult:
    timestamp_local: datetime
    meter: str
    area: str
    predicted_kwh: float
    model_name: str
    lower_kwh: float | None = None
    upper_kwh: float | None = None


@dataclass(frozen=True)
class AnomalyRequest:
    meters: list[str] | None = None
    contamination: float = 0.05
    max_rows: int | None = None
    source_policy: str = "data_2026"
    only_anomalies: bool = False


@dataclass(frozen=True)
class AnomalyResult:
    timestamp_local: datetime
    meter: str
    area: str
    anomaly_score: float
    severity: str
    is_anomaly: bool
    anomaly_type: str
    reason: str
    kwh: float | None
    p: float | None
    q: float | None
    s: float | None
    pf: float | None
    ia: float | None
    ib: float | None
    ic: float | None
    iavg: float | None
    vavg: float | None
    voltage_imbalance_pct: float | None
    current_imbalance_pct: float | None
    thd_current: float | None
    thd_voltage: float | None
    temperature_c: float | None
    kwh_source: str


@dataclass
class MeterModelMetrics:
    meter: str
    model_name: str
    mae: float
    rmse: float
    r2: float
    rows_train: int
    rows_test: int


@dataclass
class TrainedMeterModel:
    meter: str
    area: str
    model_name: str
    kind: str
    feature_columns: list[str]
    estimator: Any
    fill_values: dict[str, float]
    metrics: MeterModelMetrics
    residual_std: float
    history: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationRequest:
    meters: list[str] | None = None
    horizon_hours: int = 24
    start_time: datetime | None = None
    temp_min: float = 22.0
    temp_max: float = 30.0
    guest_min: float = 0.0
    guest_max: float | None = None
    learning_rate: float = 0.01
    max_iterations: int = 500
    convergence_threshold: float = 1e-6
    weather_location: str | None = None
    weather_month: str | None = None


@dataclass
class OptimizationResult:
    meter: str
    area: str
    total_kwh_before: float
    total_kwh_after: float
    savings_kwh: float
    savings_percent: float
    iterations: int
    converged: bool
    schedule: list[dict[str, Any]] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)
