from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import json
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from get_weather_24h_OpenMeteo import (
    KYIV_TZ,
    REGION_META,
    REGIONS_COORDS,
    HOURLY_VARS,
    DAILY_VARS,
    _get_with_retry,
    _build_daily_lookup,
    _build_full_hour_rows,
    _build_daily_aggregates,
    build_flat_weather_row,
)

base_isw_url = "https://understandingwar.org/research/russia-ukraine"
API_KEY = ''
BASE_URL = "https://api.ukrainealarm.com/api/v3"
HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

REGIONS = {
    2:  ("Vinnytsia",       "Vinnytsia, Ukraine"),
    3:  ("Volyn",           "Lutsk, Ukraine"),
    4:  ("Dnipropetrovsk",  "Dnipro, Ukraine"),
    5:  ("Donetsk",         "Kramatorsk, Ukraine"),
    6:  ("Zhytomyr",        "Zhytomyr, Ukraine"),
    7:  ("Zakarpattia",     "Uzhhorod, Ukraine"),
    8:  ("Zaporizhzhia",    "Zaporizhzhia, Ukraine"),
    9:  ("Ivano-Frankivsk", "Ivano-Frankivsk, Ukraine"),
    10: ("Kyiv Oblast",     "Kyiv, Ukraine"),
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

DISTRICT_TO_OBLAST = {
    36: 2, 37: 2, 33: 2, 32: 2, 35: 2, 34: 2,        # Vinnytsia
    39: 3, 38: 3, 40: 3, 41: 3,                        # Volyn
    43: 4, 48: 4, 44: 4, 45: 4, 47: 4, 46: 4, 42: 4,  # Dnipropetrovsk
    49: 5, 55: 5, 53: 5, 52: 5, 50: 5, 54: 5, 51: 5, 56: 5,  # Donetsk
    58: 6, 59: 6, 60: 6, 57: 6,                        # Zhytomyr
    63: 7, 66: 7, 61: 7, 62: 7, 64: 7, 65: 7,          # Zakarpattia
    147: 8, 146: 8, 145: 8, 148: 8, 149: 8,            # Zaporizhzhia
    68: 9, 72: 9, 69: 9, 70: 9, 67: 9, 71: 9,          # Ivano-Frankivsk
    77: 10, 78: 10, 73: 10, 74: 10, 75: 10, 76: 10, 79: 10,  # Kyiv Oblast
    81: 11, 80: 11, 82: 11, 83: 11,                    # Kirovohrad
    92: 13, 90: 13, 93: 13, 89: 13, 94: 13, 88: 13, 91: 13,  # Lviv
    96: 14, 97: 14, 98: 14, 95: 14,                    # Mykolaiv
    103: 15, 100: 15, 105: 15, 104: 15, 101: 15, 99: 15, 102: 15,  # Odesa
    107: 16, 106: 16, 109: 16, 108: 16,                # Poltava
    110: 17, 113: 17, 111: 17, 112: 17,                # Rivne
    115: 18, 118: 18, 116: 18, 117: 18, 114: 18,       # Sumy
    121: 19, 120: 19, 119: 19,                          # Ternopil
    127: 20, 124: 20, 126: 20, 122: 20, 128: 20, 125: 20, 123: 20,  # Kharkiv
    130: 21, 133: 21, 131: 21, 132: 21, 129: 21,       # Kherson
    135: 22, 136: 22, 134: 22,                          # Khmelnytskyi
    150: 23, 151: 23, 152: 23, 153: 23,                # Cherkasy
    139: 24, 137: 24, 138: 24,                          # Chernivtsi
    143: 25, 140: 25, 142: 25, 141: 25, 144: 25,       # Chernihiv
    31: 26,                                             # Kyiv city
}

KYIV_TZ = ZoneInfo("Europe/Kyiv")


def collect_isw(date: datetime) -> dict | None:
    date_slug = f"{date.strftime('%B').lower()}-{date.day}-{date.year}"
    url = f"{base_isw_url}/russian-offensive-campaign-assessment-{date_slug}/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        print(f"RequestError: {e}")
        return None

    if response.status_code != 200:
        print(f"ISW report not found for {date.strftime('%Y-%m-%d')} (status {response.status_code})")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    article = soup.find("article")
    if article is None:
        article = soup.find("main")
    if article is None:
        article = soup.body

    full_text = article.get_text("\n", strip=True) if article is not None else None

    if not title or not full_text:
        print(f"Failed to parse ISW content for {date.strftime('%Y-%m-%d')}")
        return None

    return {
        "date": date.strftime("%Y-%m-%d"),
        "title": title,
        "url": url,
        "text": full_text
    }

def save_isw(data: dict, report_date: datetime) -> None:
    base_dir = Path(__file__).resolve().parent.parent

    date_str = report_date.strftime("%Y-%m-%d")
    dir_path = base_dir / "data" / "raw_snapshots" / "isw_reports" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"isw_report_{date_str}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_isw_in_period(date_start: datetime, date_end: datetime) -> None:
    current_day = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = date_end.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_day <= last_day:
        article = collect_isw(current_day)

        if article is not None:
            save_isw(article, current_day)
            print(f"saved isw for {current_day.date()}")
        else:
            print(f"no isw report yet for {current_day.date()}")

        current_day += timedelta(days=1)



def date_alarm(date: datetime) -> json:
    date_str = date.strftime("%Y%m%d")

    while True:

        response = requests.get(
            f"{BASE_URL}/alerts/dateHistory",
            headers={"Authorization": API_KEY},
            params={"date": date_str}
        )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error {response.status_code}. Trying again")
            time.sleep(5)
            continue

def save_result(result: dict, hour: datetime) -> None:
    base_dir = Path(__file__).resolve().parent.parent

    date_str = hour.strftime("%Y-%m-%d")
    time_str = hour.replace(minute=0, second=0).strftime("%H-%M")

    dir_path = base_dir / "data" / "raw_snapshots" / "alarms_hourly" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"alarms_hour_{date_str}_{time_str}.json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def alarms_in_hour(alarms: list, hour_start: datetime) -> list:
    hour_end = hour_start + timedelta(hours=1)
    dt_str = hour_start.astimezone(KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        {
            "datetime_hour": dt_str,
            "region_id": region_id,
            "region_key": REGION_META[region_id]["region_key"],
            "alarm_active": 0,
        }
        for region_id in REGIONS
    ]
    row_index = {row["region_id"]: row for row in rows}

    for alarm in alarms:
        start = datetime.strptime(alarm["startDate"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        end_str = alarm.get("endDate")
        if alarm.get("isContinue") or not end_str or end_str.startswith("0001"):
            end = datetime.now(timezone.utc)
        else:
            end = datetime.strptime(end_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

        region_id = DISTRICT_TO_OBLAST.get(int(alarm.get("regionId")))
        if not region_id:
            continue

        if start < hour_end and end > hour_start:
            row_index[region_id]["alarm_active"] = 1

    return rows

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


def get_day_weather(lat: float, lon: float, date_value: datetime.date) -> tuple[dict, dict]:
    date_str = date_value.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_VARS),
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Kyiv",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }

    response = _get_with_retry(HISTORY_URL, params=params, timeout=60, max_retries=5)
    payload = response.json()
    return payload.get("hourly", {}), payload.get("daily", {})

def build_hourly_snapshot(date_value: datetime.date, hour: int, all_regions_data: dict) -> dict:
    regions_list = []

    for region_id, meta in REGION_META.items():
        region_payload = all_regions_data.get(region_id, {})
        full_rows = region_payload.get("full_rows", [])
        daily_lookup = region_payload.get("daily_lookup", {})
        daily_agg = region_payload.get("daily_agg", {})

        weather = {}
        target_dt = datetime.combine(date_value, datetime.min.time(), tzinfo=KYIV_TZ).replace(hour=hour)

        for row in full_rows:
            if row["datetime_hour_dt"] == target_dt:
                weather = build_flat_weather_row(
                    region_id=region_id,
                    row=row,
                    daily_lookup=daily_lookup,
                    daily_agg=daily_agg,
                )
                break

        regions_list.append({
            "region_id": region_id,
            "region": meta["region_key"],
            "weather": weather,
        })

    return {
        "date": date_value.strftime("%Y-%m-%d"),
        "hour": f"{hour:02d}:00",
        "regions": regions_list,
    }

def save_hourly_snapshot(snapshot: dict, date_value: datetime.date, hour: int, base_dir: Path) -> None:
    date_str = date_value.strftime("%Y-%m-%d")
    dir_path = base_dir / "data" / "raw_snapshots" / "weather_historical" / date_str
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
        failed_regions = []

        for region_id, (_, lat, lon) in REGIONS_COORDS.items():
            try:
                hourly, daily = get_day_weather(lat, lon, current_day.date())
                full_rows = _build_full_hour_rows(hourly)
                daily_lookup = _build_daily_lookup(daily)
                daily_agg = _build_daily_aggregates(full_rows)

                all_regions_data[region_id] = {
                    "full_rows": full_rows,
                    "daily_lookup": daily_lookup,
                    "daily_agg": daily_agg,
                }

            except Exception as exc:
                region_name = REGION_META[region_id]["city_name"]
                print(f"error for {region_name} on {current_day.date()}: {exc}")
                failed_regions.append(region_name)

            time.sleep(0.4)

        if failed_regions:
            print(f"skip save for {current_day.date()} because failed regions: {failed_regions}")
            current_day += timedelta(days=1)
            continue

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
    date_end = datetime.now(KYIV_TZ).replace(minute=0, second=0, microsecond=0)

    save_isw_in_period(date_start, date_end)
    save_alarms_in_period(date_start, date_end)
    save_weather_in_period(date_start, date_end)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "final_merged_dataset.parquet"
    df = pd.read_parquet(data_path)

    first_date = df["datetime_hour"].max()
    if first_date.tzinfo is None:
        first_date = first_date.replace(tzinfo=KYIV_TZ)
    first_date = first_date + timedelta(hours=1)

    save_everything(first_date)