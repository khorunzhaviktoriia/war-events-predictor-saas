import datetime as dt
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

KYIV_TZ = ZoneInfo("Europe/Kyiv")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

REGION_META = {
    2:  {"city_name": "Vinnytsia",       "location": "Vinnytsia, Ukraine",       "region_key": "Вінницька",        "lat": 49.2331, "lon": 28.4682},
    3:  {"city_name": "Lutsk",           "location": "Lutsk, Ukraine",           "region_key": "Волинська",         "lat": 50.7472, "lon": 25.3254},
    4:  {"city_name": "Dnipro",          "location": "Dnipro, Ukraine",          "region_key": "Дніпропетровська",  "lat": 48.4647, "lon": 35.0462},
    5:  {"city_name": "Donetsk",         "location": "Kramatorsk, Ukraine",      "region_key": "Донецька",          "lat": 48.7021, "lon": 37.5022},
    6:  {"city_name": "Zhytomyr",        "location": "Zhytomyr, Ukraine",        "region_key": "Житомирська",       "lat": 50.2547, "lon": 28.6587},
    7:  {"city_name": "Uzhgorod",        "location": "Uzhhorod, Ukraine",        "region_key": "Закарпатська",      "lat": 48.6208, "lon": 22.2879},
    8:  {"city_name": "Zaporozhye",      "location": "Zaporizhzhia, Ukraine",    "region_key": "Запорізька",        "lat": 47.8388, "lon": 35.1396},
    9:  {"city_name": "Ivano-Frankivsk", "location": "Ivano-Frankivsk, Ukraine", "region_key": "Івано-Франківська", "lat": 48.9226, "lon": 24.7111},
    10: {"city_name": "Kyiv",            "location": "Kyiv, Ukraine",            "region_key": "Київська",          "lat": 50.4501, "lon": 30.5234},
    11: {"city_name": "Kropyvnytskyi",   "location": "Kropyvnytskyi, Ukraine",   "region_key": "Кіровоградська",    "lat": 48.5079, "lon": 32.2623},
    13: {"city_name": "Lviv",            "location": "Lviv, Ukraine",            "region_key": "Львівська",         "lat": 49.8397, "lon": 24.0297},
    14: {"city_name": "Mykolaiv",        "location": "Mykolaiv, Ukraine",        "region_key": "Миколаївська",      "lat": 46.9750, "lon": 31.9946},
    15: {"city_name": "Odesa",           "location": "Odesa, Ukraine",           "region_key": "Одеська",           "lat": 46.4825, "lon": 30.7233},
    16: {"city_name": "Poltava",         "location": "Poltava, Ukraine",         "region_key": "Полтавська",        "lat": 49.5883, "lon": 34.5514},
    17: {"city_name": "Rivne",           "location": "Rivne, Ukraine",           "region_key": "Рівненська",        "lat": 50.6199, "lon": 26.2516},
    18: {"city_name": "Sumy",            "location": "Sumy, Ukraine",            "region_key": "Сумська",           "lat": 50.9077, "lon": 34.7981},
    19: {"city_name": "Ternopil",        "location": "Ternopil, Ukraine",        "region_key": "Тернопільська",     "lat": 49.5535, "lon": 25.5948},
    20: {"city_name": "Kharkiv",         "location": "Kharkiv, Ukraine",         "region_key": "Харківська",        "lat": 49.9935, "lon": 36.2304},
    21: {"city_name": "Kherson",         "location": "Kherson, Ukraine",         "region_key": "Херсонська",        "lat": 46.6354, "lon": 32.6169},
    22: {"city_name": "Khmelnytskyi",    "location": "Khmelnytskyi, Ukraine",    "region_key": "Хмельницька",       "lat": 49.4229, "lon": 26.9870},
    23: {"city_name": "Cherkasy",        "location": "Cherkasy, Ukraine",        "region_key": "Черкаська",         "lat": 49.4444, "lon": 32.0598},
    24: {"city_name": "Chernivtsi",      "location": "Chernivtsi, Ukraine",      "region_key": "Чернівецька",       "lat": 48.2921, "lon": 25.9358},
    25: {"city_name": "Chernihiv",       "location": "Chernihiv, Ukraine",       "region_key": "Чернігівська",      "lat": 51.4982, "lon": 31.2893},
    26: {"city_name": "Kyiv",            "location": "Kyiv, Ukraine",            "region_key": "Київ",              "lat": 50.4501, "lon": 30.5234},
}

REGIONS_COORDS = {
    region_id: (meta["city_name"], meta["lat"], meta["lon"])
    for region_id, meta in REGION_META.items()
}

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "wind_gusts_10m",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
    "shortwave_radiation",
    "uv_index",
    "weather_code",
]

DAILY_VARS = [
    "sunrise",
    "sunset",
]

def _get_with_retry(url, params=None, timeout=60, max_retries=3):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r

            print(f"status {r.status_code}, attempt {attempt}/{max_retries}")
            last_error = Exception(f"HTTP {r.status_code}")

        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_error = e
            print(f"{type(e).__name__}, attempt {attempt}/{max_retries}")

        time.sleep(5 * attempt)  # 5s, 10s, 15s

    raise Exception(f"Failed after {max_retries} retries: {url}. Last error: {last_error}")


def _round_or_none(value, ndigits=2):
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _parse_hms(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value).strftime("%H:%M:%S")
    except Exception:
        return value[-8:] if len(value) >= 8 else value


def _moon_phase_fraction(day: dt.date) -> float:
    known_new_moon = dt.datetime(2000, 1, 6, 18, 14, tzinfo=dt.timezone.utc)
    current = dt.datetime.combine(day, dt.time(12, 0), tzinfo=dt.timezone.utc)
    synodic_month = 29.53058867
    days_since = (current - known_new_moon).total_seconds() / 86400
    return round((days_since % synodic_month) / synodic_month, 2)


def _simple_condition_flags(weather_code):
    clear_codes = {0, 1}
    cloudy_codes = {2, 3, 45, 48}
    snow_codes = {71, 73, 75, 77, 85, 86}
    rain_codes = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}

    return {
        "hour_conditions_simple_Clear": weather_code in clear_codes,
        "hour_conditions_simple_Cloudy": weather_code in cloudy_codes,
        "hour_conditions_simple_Rain": weather_code in rain_codes,
        "hour_conditions_simple_Snow": weather_code in snow_codes,
    }


def _get_val(hourly: dict, key: str, i: int):
    values = hourly.get(key, [])
    return values[i] if i < len(values) else None


def _safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe_max(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _safe_min(values):
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _build_daily_lookup(daily: dict) -> dict:
    result = {}
    dates = daily.get("time", []) or []

    for i, day_str in enumerate(dates):
        sunrise_values = daily.get("sunrise", []) or []
        sunset_values = daily.get("sunset", []) or []

        result[day_str] = {
            "day_sunrise": _parse_hms(sunrise_values[i] if i < len(sunrise_values) else None),
            "day_sunset": _parse_hms(sunset_values[i] if i < len(sunset_values) else None),
        }

    return result


def _build_full_hour_rows(hourly: dict):
    # збираємо всі години, а не тільки наступні 24, щоб денні фічі рахувались по повному дню
    rows = []
    times = hourly.get("time", []) or []

    for i, time_str in enumerate(times):
        hour_dt = dt.datetime.strptime(time_str, "%Y-%m-%dT%H:%M").replace(tzinfo=KYIV_TZ)

        rows.append({
            "datetime_hour_dt": hour_dt,
            "day_datetime": hour_dt.strftime("%Y-%m-%d"),
            "hour_temp": _get_val(hourly, "temperature_2m", i),
            "hour_feelslike": _get_val(hourly, "apparent_temperature", i),
            "hour_humidity": _get_val(hourly, "relative_humidity_2m", i),
            "hour_dew": _get_val(hourly, "dew_point_2m", i),
            "hour_precip": _get_val(hourly, "precipitation", i),
            "hour_snow": _get_val(hourly, "snowfall", i),
            "hour_snowdepth": _get_val(hourly, "snow_depth", i),
            "hour_windgust": _get_val(hourly, "wind_gusts_10m", i),
            "hour_windspeed": _get_val(hourly, "wind_speed_10m", i),
            "hour_winddir": _get_val(hourly, "wind_direction_10m", i),
            "hour_pressure": _get_val(hourly, "pressure_msl", i),
            "hour_cloudcover": _get_val(hourly, "cloud_cover", i),
            "hour_solarradiation": _get_val(hourly, "shortwave_radiation", i),
            "hour_uvindex": _get_val(hourly, "uv_index", i),
            "weather_code": _get_val(hourly, "weather_code", i),
        })

    return rows


def _build_daily_aggregates(full_rows: list[dict]) -> dict:
    by_day = {}
    for row in full_rows:
        by_day.setdefault(row["day_datetime"], []).append(row)

    daily_agg = {}
    for day, day_rows in by_day.items():
        precip_hours = sum(
            1
            for r in day_rows
            if (r["hour_precip"] or 0) > 0 or (r["hour_snow"] or 0) > 0
        )

        daily_agg[day] = {
            "day_tempmax": _safe_max([r["hour_temp"] for r in day_rows]),
            "day_tempmin": _safe_min([r["hour_temp"] for r in day_rows]),
            "day_temp": _safe_mean([r["hour_temp"] for r in day_rows]),
            "day_dew": _safe_mean([r["hour_dew"] for r in day_rows]),
            "day_humidity": _safe_mean([r["hour_humidity"] for r in day_rows]),
            "day_precip": sum((r["hour_precip"] or 0) for r in day_rows),
            "day_precipcover": (precip_hours / len(day_rows) * 100) if day_rows else None,
            "day_snow": sum((r["hour_snow"] or 0) for r in day_rows),
            "day_windgust": _safe_max([r["hour_windgust"] for r in day_rows]),
            "day_cloudcover": _safe_mean([r["hour_cloudcover"] for r in day_rows]),
            "day_moonphase": _moon_phase_fraction(dt.datetime.strptime(day, "%Y-%m-%d").date()),
        }

    return daily_agg


def build_flat_weather_row(region_id: int,row: dict,daily_lookup: dict,daily_agg: dict,) -> dict:
    meta = REGION_META[region_id]
    hour_dt = row["datetime_hour_dt"]
    day = row["day_datetime"]

    return {
        "city_name": meta["city_name"],
        "datetime_hour": hour_dt.strftime("%Y-%m-%d %H:%M:%S"),

        "day_tempmax": _round_or_none(daily_agg[day]["day_tempmax"], 1),
        "day_tempmin": _round_or_none(daily_agg[day]["day_tempmin"], 1),
        "day_temp": _round_or_none(daily_agg[day]["day_temp"], 1),
        "day_dew": _round_or_none(daily_agg[day]["day_dew"], 1),
        "day_humidity": _round_or_none(daily_agg[day]["day_humidity"], 1),
        "day_precip": _round_or_none(daily_agg[day]["day_precip"], 3),
        "day_precipcover": _round_or_none(daily_agg[day]["day_precipcover"], 2),
        "day_snow": _round_or_none(daily_agg[day]["day_snow"], 1),
        "day_windgust": _round_or_none(daily_agg[day]["day_windgust"], 1),
        "day_cloudcover": _round_or_none(daily_agg[day]["day_cloudcover"], 1),
        "day_moonphase": _round_or_none(daily_agg[day]["day_moonphase"], 2),

        "hour_temp": _round_or_none(row["hour_temp"], 1),
        "hour_feelslike": _round_or_none(row["hour_feelslike"], 1),
        "hour_humidity": _round_or_none(row["hour_humidity"], 2),
        "hour_dew": _round_or_none(row["hour_dew"], 1),
        "hour_precip": _round_or_none(row["hour_precip"], 3),
        "hour_snow": _round_or_none(row["hour_snow"], 1),
        "hour_snowdepth": _round_or_none(row["hour_snowdepth"], 2),
        "hour_windgust": _round_or_none(row["hour_windgust"], 1),
        "hour_windspeed": _round_or_none(row["hour_windspeed"], 1),
        "hour_winddir": _round_or_none(row["hour_winddir"], 1),
        "hour_pressure": _round_or_none(row["hour_pressure"], 1),
        "hour_cloudcover": _round_or_none(row["hour_cloudcover"], 1),
        "hour_solarradiation": _round_or_none(row["hour_solarradiation"], 1),
        "hour_solarenergy": _round_or_none((row["hour_solarradiation"] or 0) * 0.0036, 4),
        "hour_uvindex": _round_or_none(row["hour_uvindex"], 1),

        "day_of_week": hour_dt.weekday(),
        "hour": hour_dt.hour,
        "day_datetime": day,
        "day_sunrise": daily_lookup.get(day, {}).get("day_sunrise"),
        "day_sunset": daily_lookup.get(day, {}).get("day_sunset"),

        **_simple_condition_flags(row["weather_code"]),

        "region_key": meta["region_key"],
        "region_id": region_id,
    }


def get_weather(lat: float, lon: float, forecast_start: dt.datetime):
    end_date = forecast_start.date() + dt.timedelta(days=2)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "daily": ",".join(DAILY_VARS),
        "start_date": forecast_start.date().strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Europe/Kyiv",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }

    response = _get_with_retry(FORECAST_URL, params=params, timeout=60, max_retries=5)
    payload = response.json()
    return payload.get("hourly", {}), payload.get("daily", {})


def build_rows_for_region(region_id: int, hourly: dict, daily: dict, forecast_start: dt.datetime):
    full_rows = _build_full_hour_rows(hourly)
    daily_lookup = _build_daily_lookup(daily)
    daily_agg = _build_daily_aggregates(full_rows)

    future_rows = [row for row in full_rows if row["datetime_hour_dt"] >= forecast_start][:24]

    result = []
    for row in future_rows:
        result.append(
            build_flat_weather_row(
                region_id=region_id,
                row=row,
                daily_lookup=daily_lookup,
                daily_agg=daily_agg,
            )
        )
    return result


def get_weather_for_all_regions(forecast_start: dt.datetime) -> dict:
    result = {}

    for region_id, meta in REGION_META.items():
        try:
            hourly, daily = get_weather(meta["lat"], meta["lon"], forecast_start)
            result[region_id] = build_rows_for_region(region_id, hourly, daily, forecast_start)
        except Exception as exc:
            print(f"error for [{region_id:02d}] {meta['location']}: {exc}")
            result[region_id] = []

        time.sleep(0.4)

    return result


def save_weather_data(data: dict, forecast_start: dt.datetime):
    base_dir = Path(__file__).resolve().parent.parent

    date_str = forecast_start.strftime("%Y-%m-%d")
    hour_str = forecast_start.strftime("%H-%M")

    dir_path = base_dir / "data" / "raw_snapshots" / "weather_forecast_24h" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"weather_forecast_{date_str}_{hour_str}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"saved forecast {file_path.name}")


if __name__ == "__main__":
    now_kyiv = dt.datetime.now(KYIV_TZ).replace(minute=0, second=0, microsecond=0)
    result = get_weather_for_all_regions(now_kyiv)
    save_weather_data(result, now_kyiv)