from __future__ import annotations

from electricity_forecast.data import (
    parse_meter_name,
    read_cumulative_kwh_hourly,
    read_guest_counts,
    read_kwh_target,
)


def test_parse_meter_name_extracts_meter_area_metric():
    parsed = parse_meter_name("System1:PMS_PARCEL1N_MSB02_KWH.value.PVLAST")
    assert parsed.meter == "PARCEL1N_MSB02"
    assert parsed.area == "PARCEL1N"
    assert parsed.metric == "KWH"


def test_parse_meter_name_handles_hyphen_metrics():
    parsed = parse_meter_name("System1:PMS_SHOW_MSB01_THD-R-I3.value.PVLAST")
    assert parsed.meter == "SHOW_MSB01"
    assert parsed.area == "SHOW"
    assert parsed.metric == "THD-R-I3"


def test_read_kwh_target_treats_time_hour_as_local(tmp_path):
    csv_path = tmp_path / "data_kwh.csv"
    csv_path.write_text(
        "name,time,hour,value\n"
        "System1:PMS_FB2_MSB01_KWH.value.PVLAST,2026-01-01,7,125.5\n",
        encoding="utf-8",
    )
    df = read_kwh_target(csv_path)
    row = df.iloc[0]
    assert row["meter"] == "FB2_MSB01"
    assert row["timestamp_local"].hour == 7
    assert str(row["timestamp_local"].tz) == "Asia/Ho_Chi_Minh"
    assert row["kwh"] == 125.5


def test_read_guest_counts_treats_datetime_as_local(tmp_path):
    csv_path = tmp_path / "guests.csv"
    csv_path.write_text(
        "datetime,day,hour,visitors\n2026-01-01 07:00,Thu,7,331\n",
        encoding="utf-8",
    )
    df = read_guest_counts(csv_path)
    row = df.iloc[0]
    assert row["timestamp_local"].hour == 7
    assert str(row["timestamp_local"].tz) == "Asia/Ho_Chi_Minh"
    assert row["guest_count"] == 331


def test_read_cumulative_kwh_hourly_converts_utc_and_diffs(tmp_path):
    csv_path = tmp_path / "data_2026.csv"
    csv_path.write_text(
        "time,name,original_value_float\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,100\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,112\n",
        encoding="utf-8",
    )
    df = read_cumulative_kwh_hourly(csv_path)
    second = df.sort_values("timestamp_local").iloc[1]
    assert second["timestamp_local"].hour == 8
    assert str(second["timestamp_local"].tz) == "Asia/Ho_Chi_Minh"
    assert second["kwh_telemetry"] == 12


def test_read_cumulative_kwh_hourly_flags_negative_delta(tmp_path):
    csv_path = tmp_path / "data_2026.csv"
    csv_path.write_text(
        "time,name,original_value_float\n"
        "2026-01-01 00:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,100\n"
        "2026-01-01 01:00:00+00,System1:PMS_FB2_MSB01_KWH.value.PVLAST,90\n",
        encoding="utf-8",
    )
    df = read_cumulative_kwh_hourly(csv_path)
    second = df.sort_values("timestamp_local").iloc[1]
    assert second["kwh_telemetry_issue"] == "kwh_reset_or_negative_delta"
    assert second["kwh_telemetry"] != second["kwh_telemetry"]
