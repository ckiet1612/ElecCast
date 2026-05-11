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
    kwh_csv: Path
    guests_csv: Path | None = None
    energy_log_csv: Path | None = None
    pf_csv: Path | None = None
    current_csv: Path | None = None
    telemetry_csv: Path | None = None


@dataclass(frozen=True)
class ForecastRequest:
    meters: list[str]
    horizon_hours: int
    start_time: datetime | None = None
    temperature_c: float | None = None
    guest_count: float | None = None


@dataclass(frozen=True)
class ForecastResult:
    timestamp_local: datetime
    meter: str
    area: str
    predicted_kwh: float
    model_name: str
    lower_kwh: float | None = None
    upper_kwh: float | None = None


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
