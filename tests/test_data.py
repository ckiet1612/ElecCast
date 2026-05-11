from __future__ import annotations

from electricity_forecast.data import (
    parse_meter_name,
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
