import os
import json
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

API_KEY = ""
BASE_URL = "https://api.ukrainealarm.com/api/v3"

REGIONS = {
    2: "Vinnytsia", 3: "Volyn", 4: "Dnipropetrovsk", 5: "Donetsk", 6: "Zhytomyr", 7: "Zakarpattia",
    8: "Zaporizhzhia", 9: "Ivano-Frankivsk", 10: "Kyiv Oblast", 11: "Kirovohrad", 13: "Lviv",
    14: "Mykolaiv", 15: "Odesa", 16: "Poltava", 17: "Rivne", 18: "Sumy", 19: "Ternopil",
    20: "Kharkiv", 21: "Kherson", 22: "Khmelnytskyi", 23: "Cherkasy", 24: "Chernivtsi", 25: "Chernihiv", 26: "Kyiv",
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

API_TO_OURS = {
    4: 2, 8: 3, 9: 4, 28: 5, 10: 6, 11: 7, 12: 8, 13: 9, 14: 10, 15: 11, 16: 12, 27: 13, 17: 14,
    18: 15, 19: 16, 5: 17, 20: 18, 21: 19, 22: 20, 23: 21, 3: 22, 24: 23, 26: 24, 25: 25, 31: 26,
}

def date_alarm(date: datetime) -> json:
    date_str = date.strftime("%Y%m%d")

    response = requests.get(
        f"{BASE_URL}/alerts/dateHistory",
        headers={"Authorization": API_KEY},
        params={"date": date_str}
    )

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"{response.status_code}: {response.text}")

def alarms_in_hour(date_hour: datetime) -> dict:
    hour_start = date_hour.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    hour_end = (date_hour + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    result = {region_id: False for region_id in REGIONS.keys()}

    alarms = date_alarm(date_hour)
    if not alarms:
        return result

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

def save_result(result: dict, hour: datetime) -> None:
    folder = "alarms"
    os.makedirs(folder, exist_ok=True)
    filename = f"{hour.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def get_all_regions():
    respose = requests.get(
        f"{BASE_URL}/regions",
        headers={"Authorization": API_KEY},
    )

    print(respose.status_code)
    return respose.json()

if __name__ == "__main__":
    target = datetime.now(timezone.utc) - timedelta(hours=1)
    result = alarms_in_hour(target)
    print(f"alarms between {target.astimezone(ZoneInfo("Europe/Kyiv")).replace(minute=0, second=0, microsecond=0)} and {target.astimezone(ZoneInfo("Europe/Kyiv")).replace(minute=0, second=0, microsecond=0)+timedelta(hours=1)}")
    print(result)
    save_result(result, target.astimezone(ZoneInfo("Europe/Kyiv")))