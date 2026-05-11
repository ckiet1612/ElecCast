from __future__ import annotations

from datetime import date

import electricity_forecast.weather as weather


def test_month_bounds_accepts_year_month():
    start, end = weather.month_bounds("2026-02")
    assert start == date(2026, 2, 1)
    assert end == date(2026, 2, 28)


def test_resolve_location_uses_predefined_hon_thom_without_api(monkeypatch):
    def fail_get_json(url, params):
        raise AssertionError("predefined locations should not call API")

    monkeypatch.setattr(weather, "_get_json", fail_get_json)
    location = weather.resolve_location("Hon Thom, Phu Quoc")
    assert location.label == "Hòn Thơm, Phú Quốc"
    assert round(location.latitude, 3) == 9.957


def test_resolve_location_prefers_exact_predefined_location(monkeypatch):
    def fail_get_json(url, params):
        raise AssertionError("predefined locations should not call API")

    monkeypatch.setattr(weather, "_get_json", fail_get_json)
    location = weather.resolve_location("Phu Quoc")
    assert location.label == "Phú Quốc"


def test_monthly_average_temperature_averages_daily_values(monkeypatch):
    weather.monthly_average_temperature.cache_clear()

    def fake_get_json(url, params):
        if url == weather.GEOCODING_URL:
            return {
                "results": [
                    {
                        "name": "Test City",
                        "latitude": 10.0,
                        "longitude": 104.0,
                        "timezone": "Asia/Ho_Chi_Minh",
                        "country_code": "VN",
                        "country": "Vietnam",
                    }
                ]
            }
        assert params["daily"] == "temperature_2m_mean"
        assert params["start_date"] == "2026-01-01"
        assert params["end_date"] == "2026-01-31"
        return {"daily": {"temperature_2m_mean": [20.0, None, 22.0, 24.0]}}

    monkeypatch.setattr(weather, "_get_json", fake_get_json)
    result = weather.monthly_average_temperature("Test City", "2026-01")
    assert result.average_c == 22.0
    assert result.location_label == "Test City, Vietnam"
