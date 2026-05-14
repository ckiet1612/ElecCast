from __future__ import annotations

from .types import AnomalyRequest

ANOMALY_FEATURE_COLUMNS = [
    "minute",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "kwh_detection",
    "p",
    "pf",
    "iavg",
    "temperature_c",
    "guest_count",
    "lag_1h",
    "lag_24h",
    "rolling_24h",
]

ANOMALY_OUTPUT_COLUMNS = [
    "timestamp_local",
    "meter",
    "area",
    "anomaly_score",
    "severity",
    "is_anomaly",
    "reason",
    "kwh",
    "pf",
    "iavg",
    "p",
    "temperature_c",
    "guest_count",
    "kwh_source",
]

MIN_METER_ROWS = 24


def detect_anomalies(feature_table, request: AnomalyRequest | None = None):
    import numpy as np
    import pandas as pd

    request = request or AnomalyRequest()
    data = _prepare_anomaly_frame(feature_table, request)
    if data.empty:
        return pd.DataFrame(columns=ANOMALY_OUTPUT_COLUMNS)

    if len(data) < 2:
        result = _normal_result(data)
        return _finish_result(result[ANOMALY_OUTPUT_COLUMNS], request.only_anomalies)

    contamination = _clamp_contamination(request.contamination)
    global_model = (
        _fit_model(data, contamination) if len(data) >= MIN_METER_ROWS else None
    )
    result_frames = []
    for _, meter_df in data.groupby("meter", sort=True):
        if len(meter_df) >= MIN_METER_ROWS:
            model = _fit_model(meter_df, contamination)
        else:
            model = global_model
        if model is None:
            result_frames.append(_normal_result(meter_df))
            continue
        X = meter_df[ANOMALY_FEATURE_COLUMNS]
        scores = -model.decision_function(X)
        predictions = model.predict(X)
        result = meter_df.copy()
        result["anomaly_score"] = np.asarray(scores, dtype=float)
        result["is_anomaly"] = predictions == -1
        result_frames.append(result)

    result = pd.concat(result_frames, ignore_index=True)
    issue_mask = result["kwh_telemetry_issue"].fillna("").ne("")
    result["is_anomaly"] = result["is_anomaly"] | issue_mask
    result["severity"] = "Normal"
    for _, index in result.groupby("meter").groups.items():
        anomaly_index = result.loc[index][result.loc[index, "is_anomaly"]].index
        if len(anomaly_index) == 0:
            continue
        high_cutoff = result.loc[anomaly_index, "anomaly_score"].quantile(0.80)
        high_index = anomaly_index[
            result.loc[anomaly_index, "anomaly_score"] >= high_cutoff
        ]
        result.loc[anomaly_index, "severity"] = "Medium"
        result.loc[high_index, "severity"] = "High"
    result.loc[issue_mask, "severity"] = "High"

    thresholds = {
        meter: _meter_thresholds(group)
        for meter, group in data.groupby("meter", sort=False)
    }
    result["reason"] = result.apply(
        lambda row: _reason_for_row(row, thresholds.get(row["meter"], {})),
        axis=1,
    )
    result["kwh"] = result["kwh_detection"]
    result = result.sort_values(["timestamp_local", "meter"]).reset_index(drop=True)
    return _finish_result(result[ANOMALY_OUTPUT_COLUMNS], request.only_anomalies)


def _prepare_anomaly_frame(feature_table, request: AnomalyRequest):
    import pandas as pd

    data = feature_table.copy()
    if request.meters:
        data = data[data["meter"].isin(request.meters)].copy()
    if data.empty:
        return data

    if "minute" not in data:
        data["minute"] = data["timestamp_local"].dt.minute
    if "kwh_telemetry" not in data:
        data["kwh_telemetry"] = pd.NA
    if "kwh_telemetry_issue" not in data:
        data["kwh_telemetry_issue"] = ""
    if "kwh_detection" not in data:
        data["kwh_detection"] = data["kwh"]
    if request.source_policy == "data_kwh":
        data["kwh_detection"] = data["kwh"]
        data["kwh_source"] = "data_kwh"
    elif request.source_policy == "telemetry_only":
        data["kwh_detection"] = data["kwh_telemetry"]
        data["kwh_source"] = "data_2026"
    elif "kwh_source" not in data:
        data["kwh_source"] = "data_kwh"

    for column in ANOMALY_FEATURE_COLUMNS:
        if column not in data:
            data[column] = pd.NA
    for column in ["p", "pf", "iavg", "temperature_c", "guest_count", "kwh_detection"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["timestamp_local", "meter", "kwh_detection"])
    data = data.sort_values(["timestamp_local", "meter"])
    if request.max_rows and len(data) > request.max_rows:
        data = data.tail(int(request.max_rows))
    return data.reset_index(drop=True)


def _fit_model(data, contamination: float):
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        IsolationForest(contamination=contamination, random_state=42),
    )
    return model.fit(data[ANOMALY_FEATURE_COLUMNS])


def _normal_result(data):
    result = data.copy()
    result["anomaly_score"] = 0.0
    result["is_anomaly"] = result["kwh_telemetry_issue"].fillna("").ne("")
    result["severity"] = "High"
    result.loc[~result["is_anomaly"], "severity"] = "Normal"
    result["reason"] = result.apply(lambda row: _reason_for_row(row, {}), axis=1)
    result["kwh"] = result["kwh_detection"]
    return result


def _finish_result(result, only_anomalies: bool):
    if only_anomalies and "is_anomaly" in result:
        result = result[result["is_anomaly"]].copy()
    return result.reset_index(drop=True)


def _clamp_contamination(value: float) -> float:
    return min(max(float(value), 0.001), 0.5)


def _meter_thresholds(group) -> dict[str, float]:
    return {
        "kwh_median": _median(group["kwh_detection"]),
        "kwh_high": _upper_limit(group["kwh_detection"]),
        "kwh_low": _lower_limit(group["kwh_detection"]),
        "iavg_median": _median(group["iavg"]),
        "iavg_high": _upper_limit(group["iavg"]),
        "guest_low": _lower_quantile(group["guest_count"], 0.25),
    }


def _reason_for_row(row, thresholds: dict[str, float]) -> str:
    reasons = []
    issue = str(row.get("kwh_telemetry_issue") or "")
    if issue == "kwh_reset_or_negative_delta":
        reasons.append("KWH telemetry reset/negative delta")
    elif issue == "kwh_delta_outlier":
        reasons.append("KWH telemetry delta outlier")

    kwh = _float_or_none(row.get("kwh_detection"))
    pf = _float_or_none(row.get("pf"))
    iavg = _float_or_none(row.get("iavg"))
    guests = _float_or_none(row.get("guest_count"))
    kwh_median = thresholds.get("kwh_median", 0.0)
    if kwh is not None:
        if kwh > thresholds.get("kwh_high", float("inf")):
            reasons.append("kWh spike")
        if kwh_median > 0 and kwh < max(
            thresholds.get("kwh_low", 0.0), kwh_median * 0.25
        ):
            reasons.append("kWh drop")
    if pf is not None and pf < 0.85:
        reasons.append("Low PF")
    if iavg is not None and iavg > thresholds.get("iavg_high", float("inf")):
        reasons.append("High current")
    if (
        iavg is not None
        and kwh is not None
        and kwh_median > 0
        and iavg <= max(0.1, thresholds.get("iavg_median", 0.0) * 0.05)
        and kwh > kwh_median * 0.5
    ):
        reasons.append("kWh while current near zero")
    if (
        guests is not None
        and kwh is not None
        and kwh > thresholds.get("kwh_high", float("inf")) * 0.8
        and guests <= thresholds.get("guest_low", float("-inf"))
    ):
        reasons.append("High kWh with low guests")

    if not reasons and bool(row.get("is_anomaly")):
        reasons.append("Isolation Forest high anomaly score")
    if not reasons:
        return "Normal"
    return "; ".join(dict.fromkeys(reasons))


def _upper_limit(series) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return float("inf")
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    median = clean.median()
    if iqr > 0:
        return float(max(q3 + 3 * iqr, median * 3, q3))
    if median > 0:
        return float(median * 3)
    return float("inf")


def _lower_limit(series) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    return float(max(q1 - 3 * iqr, 0.0))


def _lower_quantile(series, quantile: float) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return float("-inf")
    return float(clean.quantile(quantile))


def _median(series) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float(clean.median())


def _float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
