import datetime as dt
import json
import requests
from pathlib import Path
from zoneinfo import ZoneInfo
import time

KYIV_TZ = ZoneInfo("Europe/Kyiv")

REGIONS_COORDS = {
    2:  ("Vinnytsia, Ukraine",       49.2331, 28.4682),
    3:  ("Lutsk, Ukraine",           50.7472, 25.3254),
    4:  ("Dnipro, Ukraine",          48.4647, 35.0462),
    5:  ("Kramatorsk, Ukraine",      48.7021, 37.5022),
    6:  ("Zhytomyr, Ukraine",        50.2547, 28.6587),
    7:  ("Uzhhorod, Ukraine",        48.6208, 22.2879),
    8:  ("Zaporizhzhia, Ukraine",    47.8388, 35.1396),
    9:  ("Ivano-Frankivsk, Ukraine", 48.9226, 24.7111),
    10: ("Kyiv, Ukraine",            50.4501, 30.5234),
    11: ("Kropyvnytskyi, Ukraine",   48.5079, 32.2623),
    13: ("Lviv, Ukraine",            49.8397, 24.0297),
    14: ("Mykolaiv, Ukraine",        46.9750, 31.9946),
    15: ("Odesa, Ukraine",           46.4825, 30.7233),
    16: ("Poltava, Ukraine",         49.5883, 34.5514),
    17: ("Rivne, Ukraine",           50.6199, 26.2516),
    18: ("Sumy, Ukraine",            50.9077, 34.7981),
    19: ("Ternopil, Ukraine",        49.5535, 25.5948),
    20: ("Kharkiv, Ukraine",         49.9935, 36.2304),
    21: ("Kherson, Ukraine",         46.6354, 32.6169),
    22: ("Khmelnytskyi, Ukraine",    49.4229, 26.9870),
    23: ("Cherkasy, Ukraine",        49.4444, 32.0598),
    24: ("Chernivtsi, Ukraine",      48.2921, 25.9358),
    25: ("Chernihiv, Ukraine",       51.4982, 31.2893),
    26: ("Kyiv, Ukraine",            50.4501, 30.5234),
}

WMO_CODE_MAP = {
    0:  ("Clear",                   "clear-day"),
    1:  ("Mostly Clear",            "clear-day"),
    2:  ("Partially cloudy",        "partly-cloudy-day"),
    3:  ("Overcast",                "cloudy"),
    45: ("Fog",                     "fog"),
    48: ("Fog",                     "fog"),
    51: ("Drizzle",                 "rain"),
    53: ("Drizzle",                 "rain"),
    55: ("Drizzle",                 "rain"),
    56: ("Freezing Drizzle",        "rain"),
    57: ("Freezing Drizzle",        "rain"),
    61: ("Rain",                    "rain"),
    63: ("Rain",                    "rain"),
    65: ("Rain",                    "rain"),
    66: ("Freezing Rain",           "rain"),
    67: ("Freezing Rain",           "rain"),
    71: ("Snow",                    "snow"),
    73: ("Snow",                    "snow"),
    75: ("Snow",                    "snow"),
    77: ("Snow",                    "snow"),
    80: ("Rain, Partially cloudy",  "rain"),
    81: ("Rain, Overcast",          "rain"),
    82: ("Rain, Overcast",          "rain"),
    85: ("Snow, Partially cloudy",  "snow"),
    86: ("Snow, Overcast",          "snow"),
    95: ("Thunderstorm",            "thunder-rain"),
    96: ("Thunderstorm",            "thunder-rain"),
    99: ("Thunderstorm",            "thunder-rain"),
}

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

def wmo_to_conditions(code, hour: int):
    conditions, icon = WMO_CODE_MAP.get(code, ("Unknown", "cloudy"))
    is_night = hour < 6 or hour >= 21
    if is_night:
        icon = icon.replace("clear-day", "clear-night") \
                   .replace("partly-cloudy-day", "partly-cloudy-night")
    return conditions, icon

def get_weather(lat: float, lon: float, forecast_start: dt.datetime) -> list[dict]:
    url = "https://api.open-meteo.com/v1/forecast"

    today_kyiv = forecast_start.date()
    end_date = today_kyiv + dt.timedelta(days=2)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relativehumidity_2m",
            "dewpoint_2m",
            "precipitation",
            "precipitation_probability",
            "snowfall",
            "snow_depth",
            "windgusts_10m",
            "windspeed_10m",
            "winddirection_10m",
            "surface_pressure",
            "cloudcover",
            "shortwave_radiation",
            "uv_index",
            "weathercode",
        ]),
        "start_date": today_kyiv.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Europe/Kiev",   # Open-Meteo returns hours in Kyiv time
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = _get_with_retry(url, params=params, timeout=60, max_retries=3)
    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])

    next_24 = []
    for i, time_str in enumerate(times):
        hour_dt_kyiv = dt.datetime.strptime(time_str, "%Y-%m-%dT%H:%M").replace(tzinfo=KYIV_TZ)
        if hour_dt_kyiv >= forecast_start:
            next_24.append(_build_vc_hour(hourly, i, hour_dt_kyiv))
        if len(next_24) == 24:
            break

    return next_24

def _build_vc_hour(hourly: dict, i: int, hour_dt: dt.datetime) -> dict:
    def v(key):
        vals = hourly.get(key, [])
        return vals[i] if vals and i < len(vals) else None

    hour = hour_dt.hour
    snow_depth_m = v("snow_depth")
    solar_rad = v("shortwave_radiation") or 0.0
    solar_energy = round(solar_rad * 3600 / 1_000_000, 4)
    wmo_code = v("weathercode") or 0
    conditions, icon = wmo_to_conditions(wmo_code, hour)

    precip_type = None
    if wmo_code in (71, 73, 75, 77, 85, 86):
        precip_type = ["snow"]
    elif wmo_code in (56, 57, 66, 67):
        precip_type = ["freezingrain"]
    elif wmo_code in list(range(51, 68)) + [80, 81, 82, 95, 96, 99]:
        precip_type = ["rain"]

    epoch = int(hour_dt.timestamp())

    return {
        "date":           hour_dt.strftime("%Y-%m-%d"),
        "datetime":       hour_dt.strftime("%H:%M:%S"),
        "datetimeEpoch":  epoch,
        "temp":           v("temperature_2m"),
        "feelslike":      v("apparent_temperature"),
        "humidity":       v("relativehumidity_2m"),
        "dew":            v("dewpoint_2m"),
        "precip":         v("precipitation"),
        "precipprob":     v("precipitation_probability"),
        "snow":           v("snowfall"),
        "snowdepth":      round(snow_depth_m * 100, 2) if snow_depth_m is not None else 0.0,
        "preciptype":     precip_type,
        "windgust":       v("windgusts_10m"),
        "windspeed":      v("windspeed_10m"),
        "winddir":        v("winddirection_10m"),
        "pressure":       v("surface_pressure"),
        "visibility":     None,
        "cloudcover":     v("cloudcover"),
        "solarradiation": solar_rad,
        "solarenergy":    solar_energy,
        "uvindex":        v("uv_index") or 0.0,
        "conditions":     conditions,
        "icon":           icon,
        "stations":       ["open-meteo"],
        "source":         "open-meteo",
    }

def get_weather_for_all_regions(forecast_start: dt.datetime) -> dict:
    result = {}
    for region_id, (location, lat, lon) in REGIONS_COORDS.items():
        try:
            result[region_id] = get_weather(lat, lon, forecast_start)
        except Exception as e:
            print(f"error for [{region_id:02d}] {location}: {e}")
            result[region_id] = []
    return result

def save_weather_data(data: dict, forecast_start: dt.datetime):
    base_dir = Path(__file__).resolve().parent.parent

    date_str = forecast_start.strftime("%Y-%m-%d")
    hour_str = forecast_start.replace(minute=0, second=0, microsecond=0).strftime("%H-%M")

    dir_path = base_dir / "data" / "raw_snapshots" / "weather_forecast_24h" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"weather_forecast_{date_str}_{hour_str}.json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"saved forecast {file_path.name}")

if __name__ == "__main__":
    now_kyiv = dt.datetime.now(KYIV_TZ)
    forecast_start = now_kyiv.replace(minute=0, second=0, microsecond=0)
    result = get_weather_for_all_regions(forecast_start)
    save_weather_data(result, forecast_start)