from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import json
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from collect_alarms_hourly import save_result, DISTRICT_TO_OBLAST, date_alarm, API_KEY
from get_weather_24h_OpenMeteo import REGIONS_COORDS, WMO_CODE_MAP, wmo_to_conditions

base_isw_url = "https://understandingwar.org/research/russia-ukraine"
base_alarms_url = API_KEY

REGIONS = {
    2:  ("Vinnytsia",       "Vinnytsia, Ukraine"),
    3:  ("Volyn",           "Lutsk, Ukraine"),
    4:  ("Dnipropetrovsk",  "Dnipro, Ukraine"),
    5:  ("Donetsk",         "Kramatorsk, Ukraine"),
    6:  ("Zhytomyr",        "Zhytomyr, Ukraine"),
    7:  ("Zakarpattia",     "Uzhhorod, Ukraine"),
    8:  ("Zaporizhzhia",    "Zaporizhzhia, Ukraine"),
    9:  ("Ivano-Frankivsk", "Ivano-Frankivsk, Ukraine"),
    10: ("Kyiv Oblast",     "Brovary, Ukraine"),
    11: ("Kirovohrad",      "Kropyvnytskyi, Ukraine"),
    13: ("Lviv",            "Lviv, Ukraine"),
    14: ("Mykolaiv",        "Mykolaiv, Ukraine"),
    15: ("Odesa",           "Odesa, Ukraine"),
    16: ("Poltava",         "Poltava, Ukraine"),
    17: ("Rivne",           "Rivne, Ukraine"),
    18: ("Sumy",            "Sumy, Ukraine"),
    19: ("Ternopil",        "Ternopil, Ukraine"),
    20: ("Kharkiv",         "Kharkiv, Ukraine"),
    21: ("Kherson",         "Kherson, Ukraine"),
    22: ("Khmelnytskyi",    "Khmelnytskyi, Ukraine"),
    23: ("Cherkasy",        "Cherkasy, Ukraine"),
    24: ("Chernivtsi",      "Chernivtsi, Ukraine"),
    25: ("Chernihiv",       "Chernihiv, Ukraine"),
    26: ("Kyiv",            "Kyiv, Ukraine"),
}

def collect_isw(date: datetime) -> dict:
    date_str = f"{date.strftime('%B').lower()}-{date.day}-{date.year}"
    url = f"{base_isw_url}/russian-offensive-campaign-assessment-{date_str}/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        return {
            "date": date.strftime("%Y-%m-%d"),
            "title": None,
            "url": url,
            "text": None
        }

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    article = soup.find("article")

    if article is None:
        article = soup.find("main")

    if article is None:
        article = soup.body

    if article is None:
        full_text = None
    else:
        full_text = article.get_text("\n", strip=True)

    return {
        "date": date.strftime("%Y-%m-%d"),
        "title": title,
        "url": url,
        "text": full_text
    }

def save_isw(data: dict, date: datetime) -> None:
    base_dir = Path(__file__).resolve().parent.parent

    date_str = date.strftime("%Y-%m-%d")
    hour_str = date.strftime("%H-%M")

    dir_path = base_dir / "data" / "raw_snapshots" / "isw_reports" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"isw_report_{date_str}_{hour_str}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_isw_in_period(date_start: datetime, date_end: datetime) -> None:
    current_day = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = date_end.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_day <= last_day:
        article = collect_isw(current_day)

        if current_day.date() == date_start.date():
            start_hour = date_start.hour
        else:
            start_hour = 0

        if current_day.date() == date_end.date():
            end_hour = date_end.hour
        else:
            end_hour = 24

        for hour in range(start_hour, end_hour):
            current_hour = current_day + timedelta(hours=hour)
            save_isw(article, current_hour)

        print(f"saved isw for {current_day.date()} hours {start_hour}..{end_hour - 1}")
        current_day += timedelta(days=1)



def alarms_in_hour(alarms: dict, hour_start:datetime) -> dict:
    hour_end = hour_start + timedelta(hours=1)
    result = {region_id: False for region_id in REGIONS.keys()}

    for alarm in alarms:
        start = datetime.strptime(alarm["startDate"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        end_str = alarm.get("endDate")

        if alarm.get("isContinue") or not end_str or end_str.startswith("0001"):
            end = datetime.now(timezone.utc)
        else:
            end = datetime.strptime(end_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

        region_id = DISTRICT_TO_OBLAST.get(int(alarm.get("regionId")))

        if region_id:
            if start < hour_end and end > hour_start:
                result[region_id] = True
            elif region_id not in result:
                result[region_id] = False

    return result

def save_alarms_in_period(date_start:datetime, date_end:datetime) -> None:
    date_start = date_start.replace(tzinfo=ZoneInfo("Europe/Kyiv"))
    current = date_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = date_end.replace(tzinfo=ZoneInfo("Europe/Kyiv")).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

    current_date = current.date()
    alarms_current = date_alarm(current)
    print(f"success! saving {current.date()}")

    while current <  end:
        alarms = alarms_in_hour(alarms_current, current)
        save_result(alarms, current.astimezone(ZoneInfo("Europe/Kyiv")))

        current += timedelta(hours=1)

        if current.date() != current_date:
            print(f"saved {current_date}!!!!!!")
            current_date = current.date()
            time.sleep(15)
            alarms_current = date_alarm(current)
            print(f"saving {current.date()}")



def is_nighttime(hour: int) -> bool:
    return hour < 6 or hour >= 21

def calc_dew_point(temp_c: float, rh: float) -> float:
    if temp_c is None or rh is None:
        return None
    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    return round((b * alpha) / (a - alpha), 2)

def calc_feels_like(temp_c: float, rh: float, windspeed_kmh: float) -> float:
    if temp_c is None:
        return None
    if temp_c <= 10 and windspeed_kmh is not None and windspeed_kmh > 4.8:
        wc = (13.12 + 0.6215 * temp_c
              - 11.37 * (windspeed_kmh ** 0.16)
              + 0.3965 * temp_c * (windspeed_kmh ** 0.16))
        return round(wc, 1)
    elif temp_c >= 27 and rh is not None:
        T, R = temp_c, rh
        hi = (-8.78469475556
              + 1.61139411 * T + 2.33854883889 * R
              - 0.14611605 * T * R - 0.012308094 * T**2
              - 0.0164248277778 * R**2 + 0.002211732 * T**2 * R
              + 0.00072546 * T * R**2 - 0.000003582 * T**2 * R**2)
        return round(hi, 1)
    return round(temp_c, 1)

def get_day_weather(lat: float, lon: float, date: datetime.date) -> dict:

    date_str = date.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude":          lat,
        "longitude":         lon,
        "start_date":        date_str,
        "end_date":          date_str,
        "hourly": ",".join([
            "temperature_2m",         # -> temp
            "apparent_temperature",   # -> feelslike
            "relativehumidity_2m",    # -> humidity
            "dewpoint_2m",            # -> dew
            "precipitation",          # -> precip
            "precipitation_probability",  # -> precipprob
            "snowfall",               # -> snow (mm water eq.)
            "snow_depth",             # -> snowdepth (m -> cm)
            "windgusts_10m",          # -> windgust
            "windspeed_10m",          # -> windspeed
            "winddirection_10m",      # -> winddir
            "surface_pressure",       # -> pressure
            "cloudcover",             # -> cloudcover
            "shortwave_radiation",    # -> solarradiation (W/m²)
            "uv_index",               # -> uvindex
            "weathercode",            # -> conditions + icon
        ]),
        "timezone":           "UTC",
        "wind_speed_unit":    "kmh",   # VC uses km/h
        "precipitation_unit": "mm",
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        return response.json().get("hourly", {})
    else:
        raise Exception(
            f"Open-Meteo {response.status_code} for ({lat},{lon}) on {date_str}: {response.text[:300]}"
        )

def build_vc_weather(hourly: dict, hour: int) -> dict:

    def v(key):
        vals = hourly.get(key, [])
        return vals[hour] if vals and hour < len(vals) else None

    temp        = v("temperature_2m")
    rh          = v("relativehumidity_2m")
    windspeed   = v("windspeed_10m")
    windgust    = v("windgusts_10m")
    snowfall_mm = v("snowfall")
    snow_depth  = v("snow_depth")
    solar_rad   = v("shortwave_radiation")
    wmo_code    = v("weathercode")
    precip_prob = v("precipitation_probability")

    solar_energy = round(solar_rad * 3600 / 1_000_000, 4) if solar_rad is not None else 0.0

    uvindex = v("uv_index") or 0.0

    precip_type = None
    if wmo_code is not None:
        if wmo_code in (71, 73, 75, 77, 85, 86):
            precip_type = ["snow"]
        elif wmo_code in (56, 57, 66, 67):
            precip_type = ["freezingrain"]
        elif wmo_code in range(51, 68) or wmo_code in (80, 81, 82):
            precip_type = ["rain"]
        elif wmo_code in (95, 96, 99):
            precip_type = ["rain"]

    conditions, icon = wmo_to_conditions(wmo_code or 0, is_nighttime(hour))

    time_str = (hourly.get("time") or [])[hour] if hourly.get("time") else None
    epoch = None
    if time_str:
        try:
            epoch = int(datetime.strptime(time_str, "%Y-%m-%dT%H:%M").replace(
                tzinfo=timezone.utc).timestamp())
        except Exception:
            epoch = None

    return {
        "datetime":       f"{hour:02d}:00:00",
        "datetimeEpoch":  epoch,
        "temp":           temp,
        "feelslike":      v("apparent_temperature"),
        "humidity":       rh,
        "dew":            v("dewpoint_2m"),
        "precip":         v("precipitation"),
        "precipprob":     precip_prob,
        "snow":           snowfall_mm,
        "snowdepth":      round(snow_depth * 100, 2) if snow_depth is not None else 0.0,  # m -> cm
        "preciptype":     precip_type,
        "windgust":       windgust,
        "windspeed":      windspeed,
        "winddir":        v("winddirection_10m"),
        "pressure":       v("surface_pressure"),
        "visibility":     None,
        "cloudcover":     v("cloudcover"),
        "solarradiation": solar_rad or 0.0,
        "solarenergy":    solar_energy,
        "uvindex":        uvindex,
        "conditions":     conditions,
        "icon":           icon,
        "stations":       ["open-meteo"],
        "source":         "open-meteo",
    }

def build_hourly_snapshot(date: datetime.date, hour: int, all_regions_data: dict) -> dict:
    regions_list = []
    for region_id, (region_name, _, _) in REGIONS_COORDS.items():
        hourly = all_regions_data.get(region_id, {})
        weather = build_vc_weather(hourly, hour) if hourly else {}
        regions_list.append({
            "region_id": region_id,
            "region":    region_name,
            "weather":   weather,
        })
    return {
        "date":    date.strftime("%Y-%m-%d"),
        "hour":    f"{hour:02d}:00",
        "regions": regions_list,
    }

def save_hourly_snapshot(snapshot: dict, date: datetime.date, hour: int, base_dir: Path) -> None:
    date_str = date.strftime("%Y-%m-%d")
    dir_path = base_dir / "data" / "raw_snapshots" / "weather_forecast" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"weather_{date_str}_{hour:02d}-00.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

def save_weather_in_period(date_start: datetime, date_end: datetime) -> None:
    base_dir = Path(__file__).resolve().parent.parent

    current_day = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = date_end.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_day <= last_day:
        all_regions_data: dict = {}

        for region_id, (region_name, lat, lon) in REGIONS_COORDS.items():
            try:
                hourly = get_day_weather(lat, lon, current_day.date())
                all_regions_data[region_id] = hourly
            except Exception as e:
                all_regions_data[region_id] = {}
                print(f"error for {region_name} on {current_day.date()}: {e}")

            time.sleep(0.1)

        if current_day.date() == date_start.date():
            start_hour = date_start.hour
        else:
            start_hour = 0

        if current_day.date() == date_end.date():
            end_hour = date_end.hour
        else:
            end_hour = 24

        for hour in range(start_hour, end_hour):
            snapshot = build_hourly_snapshot(current_day.date(), hour, all_regions_data)
            save_hourly_snapshot(snapshot, current_day.date(), hour, base_dir)

        print(f"saved weather for {current_day.date()} hours {start_hour}..{end_hour - 1}")
        current_day += timedelta(days=1)

def save_everything(date_start:datetime):
    date_end = datetime.now().replace(minute=0, second=0)

    save_isw_in_period(date_start, date_end)
    save_alarms_in_period(date_start, date_end)
    save_weather_in_period(date_start, date_end)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "final_merged_dataset.parquet"
    df = pd.read_parquet(data_path)

    first_date = df["datetime_hour"].max() + timedelta(days=1)

    save_everything(first_date)