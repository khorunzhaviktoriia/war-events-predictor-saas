import requests
import pandas as pd
from pathlib import Path

LAT = 48.002
LON = 37.8145
START = "2025-03-02"
END = "2026-03-16"
TIMEZONE = "Europe/Kyiv"

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

hourly_vars = [
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

daily_vars = [
    "sunrise",
    "sunset",
]

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "timezone": TIMEZONE,
    "hourly": ",".join(hourly_vars),
    "daily": ",".join(daily_vars),
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    "cell_selection": "land",
}

response = requests.get(BASE_URL, params=params, timeout=60)
response.raise_for_status()
data = response.json()

hourly_df = pd.DataFrame(data["hourly"])
daily_df = pd.DataFrame(data["daily"])

output_dir = Path("data/openmeteo_donetsk_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

hourly_path = output_dir / "donetsk_hourly_raw.csv"
daily_path = output_dir / "donetsk_daily_raw.csv"

hourly_df.to_csv(hourly_path, index=False, encoding="utf-8-sig")
daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")

print("Saved hourly:", hourly_path.resolve())
print("Saved daily:", daily_path.resolve())
print("Hourly shape:", hourly_df.shape)
print("Daily shape:", daily_df.shape)
print(hourly_df.head())
print(daily_df.head())