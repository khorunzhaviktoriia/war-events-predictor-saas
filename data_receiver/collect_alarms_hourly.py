import datetime
from datetime import datetime, timedelta
import json
import requests
import time
import os

API_KEY = ""
BASE_URL = "https://api.ukrainealarm.com/api/v3"

REGIONS = {
    2: "Vinnytsia", 3: "Volyn", 4: "Dnipropetrovsk", 5: "Donetsk", 6: "Zhytomyr", 7: "Zakarpattia",
    8: "Zaporizhzhia", 9: "Ivano-Frankivsk", 10: "Kyiv Oblast", 11: "Kirovohrad", 12: "Luhansk", 13: "Lviv",
    14: "Mykolaiv", 15: "Odesa", 16: "Poltava", 17: "Rivne", 18: "Sumy", 19: "Ternopil",
    20: "Kharkiv", 21: "Kherson", 22: "Khmelnytskyi", 23: "Cherkasy", 24: "Chernivtsi", 25: "Chernihiv", 26: "Kyiv",
}

API_OBLAST_TO_OURS = {
    4: 2, 8: 3, 9: 4, 28: 5, 10: 6, 11: 7, 12: 8, 13: 9, 14: 10, 15: 11, 16: 12, 27: 13, 17: 14,
    18: 15, 19: 16, 5: 17, 20: 18, 21: 19, 22: 20, 23: 21, 3: 22, 24: 23, 26: 24, 25: 25, 31: 26,
}

def build_oblast_to_districts_map(regions_data: dict) -> dict:
    result = {}
    for oblast in regions_data.get("states", []):
        api_oblast_id = int(oblast["regionId"])
        our_id = API_OBLAST_TO_OURS.get(api_oblast_id)
        if not our_id:
            continue
        districts = oblast.get("regionChildIds", [])
        if districts:
            result[our_id] = [int(d["regionId"]) for d in districts]
        else:
            result[our_id] = [api_oblast_id]
    return result


def had_district_alarm(district_id: int, hour: datetime) -> bool:
    time.sleep(5.5)

    response = requests.get(
        f"{BASE_URL}/alerts/regionHistory",
        headers={"Authorization": API_KEY},
        params={"regionId": district_id})

    result = False
    print(response.status_code)

    if response.status_code == 200:
        data = response.json()
        alarms = data[0].get("alarms", [])

        hour_start = hour - timedelta(hours=1)
        hour_end = hour

        print(alarms)

        for alarm in alarms[:2]:
            start = datetime.strptime(alarm["startDate"][:19], "%Y-%m-%dT%H:%M:%S")
            end_str = alarm.get("endDate")
            end = datetime.strptime(end_str[:19], "%Y-%m-%dT%H:%M:%S") if end_str else datetime.now()

            if start < hour_end and end > hour_start:
                result = True
                break
    return result


def had_region_alarm(our_region_id: int, hour: datetime, oblast_districts_map: dict) -> bool:
    districts = oblast_districts_map.get(our_region_id, [])
    print(f"region id: {our_region_id}")
    print(f"region name: {REGIONS.get(our_region_id)}")
    print(districts)

    result = False

    for district in districts:
        if had_district_alarm(district, hour):
            result = True
            break
    return result


def had_regions_alarms(hour: datetime) -> dict:
    regions_response = requests.get(
        f"{BASE_URL}/regions",
        headers={"Authorization": API_KEY}
    )
    oblast_districts_map = build_oblast_to_districts_map(regions_response.json())

    result = {}
    for our_id in REGIONS.keys():
        result[our_id] = had_region_alarm(our_id, hour, oblast_districts_map)
    return result


def save_result(result:dict, hour: datetime) -> None:
    folder = "alarms"
    filename = f"{hour.strftime("%Y-%m-%d_%H-%M-%S")}.json"

    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    now_time = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours = 3)
    result = had_regions_alarms(now_time)
    print(now_time)
    print(result)