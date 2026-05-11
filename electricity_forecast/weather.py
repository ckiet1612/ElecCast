from __future__ import annotations

import calendar
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
CLIMATE_MODEL = "CMCC_CM2_VHR4"
HTTP_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class WeatherLocation:
    label: str
    latitude: float
    longitude: float
    timezone: str = "Asia/Ho_Chi_Minh"


@dataclass(frozen=True)
class MonthlyTemperature:
    location_label: str
    month: str
    average_c: float
    latitude: float
    longitude: float
    source: str


class WeatherApiError(RuntimeError):
    pass


DEFAULT_WEATHER_LOCATIONS = [
    WeatherLocation("Hòn Thơm, Phú Quốc", 9.9566, 104.0165),
    WeatherLocation("An Thới, Phú Quốc", 10.01914, 104.01499),
    WeatherLocation("Phú Quốc", 10.28715, 104.01047),
    WeatherLocation("Rạch Giá", 10.01245, 105.08091),
    WeatherLocation("Cần Thơ", 10.04516, 105.74685),
    WeatherLocation("TP. Hồ Chí Minh", 10.82302, 106.62965),
    WeatherLocation("Hà Nội", 21.02776, 105.83416),
]


def default_weather_location_label() -> str:
    return DEFAULT_WEATHER_LOCATIONS[0].label


def weather_location_labels() -> list[str]:
    return [location.label for location in DEFAULT_WEATHER_LOCATIONS]


def default_weather_month(reference: date | datetime | None = None) -> str:
    if reference is None:
        reference = datetime.now()
    current = date(reference.year, reference.month, 1)
    return current.strftime("%Y-%m")


def month_options(
    reference: date | datetime | None = None,
    months_back: int = 6,
    months_ahead: int = 24,
) -> list[str]:
    if reference is None:
        reference = datetime.now()
    current = date(reference.year, reference.month, 1)
    start = _add_months(current, -months_back)
    return [
        _add_months(start, offset).strftime("%Y-%m")
        for offset in range(months_back + months_ahead + 1)
    ]


@lru_cache(maxsize=256)
def monthly_average_temperature(location_query: str, month: str) -> MonthlyTemperature:
    location = resolve_location(location_query)
    start_date, end_date = month_bounds(month)
    data = _get_json(
        CLIMATE_URL,
        {
            "latitude": f"{location.latitude:.5f}",
            "longitude": f"{location.longitude:.5f}",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "models": CLIMATE_MODEL,
            "daily": "temperature_2m_mean",
            "temperature_unit": "celsius",
            "timezone": location.timezone or "auto",
        },
    )
    values = data.get("daily", {}).get("temperature_2m_mean", [])
    temperatures = [float(value) for value in values if _is_number(value)]
    if not temperatures:
        raise WeatherApiError("Weather API did not return daily temperature values.")
    return MonthlyTemperature(
        location_label=location.label,
        month=start_date.strftime("%Y-%m"),
        average_c=round(sum(temperatures) / len(temperatures), 1),
        latitude=location.latitude,
        longitude=location.longitude,
        source=f"Open-Meteo Climate API / {CLIMATE_MODEL}",
    )


def resolve_location(location_query: str) -> WeatherLocation:
    query = (
        location_query.strip() if location_query else default_weather_location_label()
    )
    predefined = _predefined_location(query)
    if predefined is not None:
        return predefined
    coordinates = _parse_coordinates(query)
    if coordinates is not None:
        latitude, longitude = coordinates
        return WeatherLocation(query, latitude, longitude, "auto")

    results = []
    for candidate in _unique([query, _ascii_text(query)]):
        if len(candidate) < 2:
            continue
        data = _get_json(
            GEOCODING_URL,
            {
                "name": candidate,
                "count": "10",
                "language": "en",
                "format": "json",
            },
        )
        results.extend(data.get("results", []) or [])
        if results:
            break
    if not results:
        raise WeatherApiError(f"Could not find location: {query}")

    result = max(results, key=lambda item: _geocode_score(item, query))
    latitude = result.get("latitude")
    longitude = result.get("longitude")
    if latitude is None or longitude is None:
        raise WeatherApiError(f"Location has no coordinates: {query}")
    return WeatherLocation(
        label=_location_label(result),
        latitude=float(latitude),
        longitude=float(longitude),
        timezone=result.get("timezone", "auto"),
    )


def month_bounds(month: str | date | datetime) -> tuple[date, date]:
    if isinstance(month, datetime):
        year = month.year
        month_number = month.month
    elif isinstance(month, date):
        year = month.year
        month_number = month.month
    else:
        text = month.strip()
        if "/" in text:
            first, second = text.split("/", 1)
            if len(first) == 4:
                year, month_number = int(first), int(second)
            else:
                month_number, year = int(first), int(second)
        else:
            year_text, month_text = text.split("-", 1)
            year, month_number = int(year_text), int(month_text)
    if month_number < 1 or month_number > 12:
        raise WeatherApiError("Month must be between 1 and 12.")
    last_day = calendar.monthrange(year, month_number)[1]
    return date(year, month_number, 1), date(year, month_number, last_day)


def _get_json(url: str, params: dict[str, str]) -> dict:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={"User-Agent": "ElectricityForecast/0.1 Open-Meteo Client"},
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("error"):
                raise WeatherApiError(str(data.get("reason", "Weather API error.")))
            return data
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise WeatherApiError(f"Weather API error {exc.code}: {body}") from exc
    except URLError as exc:
        raise WeatherApiError(f"Weather API connection failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise WeatherApiError("Weather API request timed out.") from exc
    except json.JSONDecodeError as exc:
        raise WeatherApiError("Weather API returned invalid JSON.") from exc


def _predefined_location(query: str) -> WeatherLocation | None:
    folded_query = _fold_text(query)
    for location in DEFAULT_WEATHER_LOCATIONS:
        folded_label = _fold_text(location.label)
        if folded_query in {folded_label, _fold_text(_ascii_text(location.label))}:
            return location

    query_tokens = [token.strip(",") for token in folded_query.split() if token]
    for location in DEFAULT_WEATHER_LOCATIONS:
        folded_label = _fold_text(location.label)
        label_without_punctuation = folded_label.replace(",", "")
        if query_tokens and all(
            token in label_without_punctuation for token in query_tokens
        ):
            return location
    return None


def _parse_coordinates(query: str) -> tuple[float, float] | None:
    cleaned = query.replace(";", ",")
    parts = [part.strip() for part in cleaned.split(",", 1)]
    if len(parts) != 2:
        return None
    try:
        latitude = float(parts[0])
        longitude = float(parts[1])
    except ValueError:
        return None
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None


def _geocode_score(result: dict, query: str) -> float:
    folded_query = _fold_text(query)
    name = _fold_text(str(result.get("name", "")))
    combined = _fold_text(
        " ".join(
            str(result.get(key, ""))
            for key in ["name", "admin1", "admin2", "country"]
            if result.get(key)
        )
    )
    score = 0.0
    if result.get("country_code") == "VN":
        score += 2.0
    if folded_query == name:
        score += 10.0
    elif folded_query in name or name in folded_query:
        score += 6.0
    query_tokens = [token for token in folded_query.split() if token]
    if query_tokens and all(token in combined for token in query_tokens):
        score += 3.0
    population = result.get("population") or 0
    score += min(float(population) / 10_000_000, 1.0)
    return score


def _location_label(result: dict) -> str:
    parts = [
        str(result.get(key, "")).strip()
        for key in ["name", "admin1", "country"]
        if result.get(key)
    ]
    return ", ".join(_unique(parts))


def _add_months(value: date, offset: int) -> date:
    total = value.year * 12 + value.month - 1 + offset
    return date(total // 12, total % 12 + 1, 1)


def _ascii_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _fold_text(value: str) -> str:
    return " ".join(_ascii_text(value).casefold().split())


def _unique(values):
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _is_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)
