from __future__ import annotations

from .types import AnomalyRequest

ANOMALY_FEATURE_COLUMNS = [
    "minute",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "temperature_c",
    "kwh_detection",
    "p",
    "q",
    "s",
    "pf",
    "ia",
    "ib",
    "ic",
    "iavg",
    "voltage_imbalance_pct",
    "current_imbalance_pct",
    "vavg",
    "thd_current",
    "thd_voltage",
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
    "anomaly_type",
    "reason",
    "kwh",
    "p",
    "q",
    "s",
    "pf",
    "ia",
    "ib",
    "ic",
    "iavg",
    "vavg",
    "voltage_imbalance_pct",
    "current_imbalance_pct",
    "thd_current",
    "thd_voltage",
    "temperature_c",
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
        result = _apply_operational_classification(_normal_result(data), data)
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
        result["model_anomaly"] = result["is_anomaly"]
        result_frames.append(result)

    result = pd.concat(result_frames, ignore_index=True)
    if "model_anomaly" not in result:
        result["model_anomaly"] = result["is_anomaly"]
    result = _apply_operational_classification(result, data)
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
    result.loc[
        result["rule_severity"].eq("Medium") & result["severity"].eq("Normal"),
        "severity",
    ] = "Medium"
    result.loc[result["rule_severity"].eq("High"), "severity"] = "High"
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
        data["kwh_detection"] = data["kwh"] if "kwh" in data else pd.NA
    if (
        request.source_policy in {"data_2026", "telemetry_only"}
        and data["kwh_telemetry"].notna().any()
    ):
        data["kwh_detection"] = data["kwh_telemetry"]
        data["kwh_source"] = "data_2026"
    elif "kwh_source" not in data:
        data["kwh_source"] = "data_2026"

    if "kwh_telemetry_raw_delta" not in data:
        data["kwh_telemetry_raw_delta"] = pd.NA
    issue_mask = data["kwh_telemetry_issue"].fillna("").ne("")
    raw_delta = pd.to_numeric(data["kwh_telemetry_raw_delta"], errors="coerce")
    missing_issue_kwh = issue_mask & data["kwh_detection"].isna()
    data.loc[missing_issue_kwh & raw_delta.notna(), "kwh_detection"] = raw_delta
    data.loc[missing_issue_kwh & data["kwh_detection"].isna(), "kwh_detection"] = 0.0

    for column in ANOMALY_FEATURE_COLUMNS:
        if column not in data:
            data[column] = pd.NA
    for column in ANOMALY_FEATURE_COLUMNS:
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
    result["model_anomaly"] = False
    result["is_anomaly"] = False
    result["severity"] = "Normal"
    result["kwh"] = result["kwh_detection"]
    return result


def _apply_operational_classification(result, threshold_data):
    thresholds = {
        meter: _meter_thresholds(group)
        for meter, group in threshold_data.groupby("meter", sort=False)
    }
    analyses = [
        _classify_row(row, thresholds.get(row["meter"], {}))
        for _, row in result.iterrows()
    ]
    result = result.copy()
    result["anomaly_type"] = [analysis[0] for analysis in analyses]
    result["reason"] = [analysis[1] for analysis in analyses]
    result["rule_severity"] = [analysis[2] for analysis in analyses]
    result["is_anomaly"] = result["model_anomaly"] | result["rule_severity"].isin(
        {"Medium", "High"}
    )
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
        "p_median": _median(group["p"]),
        "p_high": _upper_limit(group["p"]),
        "s_high": _upper_limit(group["s"]),
        "iavg_median": _median(group["iavg"]),
        "iavg_high": _upper_limit(group["iavg"]),
        "current_imbalance_high": _upper_limit(group["current_imbalance_pct"]),
        "voltage_imbalance_high": _upper_limit(group["voltage_imbalance_pct"]),
        "vavg_median": _median(group["vavg"]),
    }


def _classify_row(row, thresholds: dict[str, float]) -> tuple[str, str, str]:
    findings = _operational_findings(row, thresholds)
    if findings:
        categories = "; ".join(dict.fromkeys(finding[0] for finding in findings))
        reasons = "; ".join(dict.fromkeys(finding[1] for finding in findings))
        severity = (
            "High" if any(finding[2] == "High" for finding in findings) else "Medium"
        )
        return categories, reasons, severity
    if bool(row.get("model_anomaly")):
        return "Model anomaly", "Isolation Forest high anomaly score", "Normal"
    return "Normal", "Normal", "Normal"


def _operational_findings(
    row, thresholds: dict[str, float]
) -> list[tuple[str, str, str]]:
    findings = []
    issue = str(row.get("kwh_telemetry_issue") or "")
    if issue == "kwh_reset_or_negative_delta":
        findings.append(
            ("Telemetry issue", "KWH telemetry reset/negative delta", "High")
        )
    elif issue == "kwh_delta_outlier":
        findings.append(("Telemetry issue", "KWH telemetry delta outlier", "High"))

    kwh = _float_or_none(row.get("kwh_detection"))
    p = _float_or_none(row.get("p"))
    s = _float_or_none(row.get("s"))
    pf = _float_or_none(row.get("pf"))
    ia = _float_or_none(row.get("ia"))
    ib = _float_or_none(row.get("ib"))
    ic = _float_or_none(row.get("ic"))
    iavg = _float_or_none(row.get("iavg"))
    vavg = _float_or_none(row.get("vavg"))
    voltage_imbalance = _float_or_none(row.get("voltage_imbalance_pct"))
    current_imbalance = _float_or_none(row.get("current_imbalance_pct"))
    thd_current = _float_or_none(row.get("thd_current"))
    thd_voltage = _float_or_none(row.get("thd_voltage"))
    hour = _float_or_none(row.get("hour"))
    kwh_median = thresholds.get("kwh_median", 0.0)
    if kwh is not None:
        if kwh > thresholds.get("kwh_high", float("inf")):
            findings.append(("Consumption spike", "kWh consumption spike", "Medium"))
        if kwh_median > 0 and kwh < max(
            thresholds.get("kwh_low", 0.0), kwh_median * 0.25
        ):
            findings.append(("Consumption drop", "kWh consumption drop", "Medium"))
    if pf is not None and pf < 0.85:
        severity = "High" if pf < 0.75 else "Medium"
        findings.append(("Low power factor", "Low power factor", severity))
    if _above_high_limit(iavg, thresholds.get("iavg_high", float("inf"))):
        findings.append(("System overload", "High average current", "Medium"))
    if _above_high_limit(p, thresholds.get("p_high", float("inf"))):
        findings.append(("System overload", "High active power", "Medium"))
    if _above_high_limit(s, thresholds.get("s_high", float("inf"))):
        findings.append(("System overload", "High apparent power", "Medium"))
    if current_imbalance is not None:
        current_limit = thresholds.get("current_imbalance_high", float("inf"))
        if current_imbalance >= 15.0 or _above_high_limit(
            current_imbalance, current_limit
        ):
            severity = "High" if current_imbalance >= 20.0 else "Medium"
            findings.append(
                ("Phase imbalance", "High current phase imbalance", severity)
            )
    if voltage_imbalance is not None:
        voltage_limit = thresholds.get("voltage_imbalance_high", float("inf"))
        if voltage_imbalance >= 2.0 or _above_high_limit(
            voltage_imbalance, voltage_limit
        ):
            severity = "High" if voltage_imbalance >= 3.0 else "Medium"
            findings.append(("Voltage abnormal", "High voltage imbalance", severity))
    if vavg is not None:
        vavg_median = thresholds.get("vavg_median", 0.0)
        nominal_voltage = vavg_median if vavg_median > 0 else 400.0
        if vavg < nominal_voltage * 0.90 or vavg > nominal_voltage * 1.10:
            findings.append(
                ("Voltage abnormal", "Average voltage outside range", "High")
            )
    if thd_current is not None and thd_current >= 10.0:
        severity = "High" if thd_current >= 15.0 else "Medium"
        findings.append(("Harmonic distortion", "High current THD", severity))
    if thd_voltage is not None and thd_voltage >= 3.0:
        severity = "High" if thd_voltage >= 5.0 else "Medium"
        findings.append(("Harmonic distortion", "High voltage THD", severity))
    if (
        iavg is not None
        and kwh is not None
        and kwh_median > 0
        and iavg <= max(0.1, thresholds.get("iavg_median", 0.0) * 0.05)
        and kwh > kwh_median * 0.5
    ):
        findings.append(
            ("Equipment abnormal", "kWh while average current is near zero", "High")
        )
    if _phase_currents_abnormal(ia, ib, ic, iavg):
        findings.append(
            ("Equipment abnormal", "Current phase reading is abnormal", "Medium")
        )
    if _off_hours(hour) and _off_hour_consumption(kwh, p, thresholds):
        findings.append(
            (
                "Off-hours consumption",
                "High consumption outside operating hours",
                "Medium",
            )
        )
    return findings


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


def _above_high_limit(value: float | None, limit: float) -> bool:
    return value is not None and limit != float("inf") and value > limit


def _phase_currents_abnormal(
    ia: float | None,
    ib: float | None,
    ic: float | None,
    iavg: float | None,
) -> bool:
    phases = [value for value in [ia, ib, ic] if value is not None]
    if len(phases) < 3:
        return False
    phase_avg = sum(phases) / len(phases)
    reference = iavg if iavg is not None and iavg > 0 else phase_avg
    if reference <= 0:
        return False
    return max(abs(value - reference) for value in phases) / reference > 0.30


def _off_hours(hour: float | None) -> bool:
    return hour is not None and (hour <= 5 or hour >= 23)


def _off_hour_consumption(
    kwh: float | None,
    p: float | None,
    thresholds: dict[str, float],
) -> bool:
    kwh_median = thresholds.get("kwh_median", 0.0)
    p_median = thresholds.get("p_median", 0.0)
    return (kwh is not None and kwh_median > 0 and kwh > kwh_median * 1.25) or (
        p is not None and p_median > 0 and p > p_median * 1.25
    )


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
