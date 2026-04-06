import datetime as dt
import json
import requests
import os

VCRO_KEY = ''

REGIONS = {
    2:  "Vinnytsia, Ukraine",
    3:  "Lutsk, Ukraine",
    4:  "Dnipro, Ukraine",
    5:  "Kramatorsk, Ukraine",
    6:  "Zhytomyr, Ukraine",
    7:  "Uzhhorod, Ukraine",
    8:  "Zaporizhzhia, Ukraine",
    9:  "Ivano-Frankivsk, Ukraine",
    10: "Kyiv, Ukraine",
    11: "Kropyvnytskyi, Ukraine",
    13: "Lviv, Ukraine",
    14: "Mykolaiv, Ukraine",
    15: "Odesa, Ukraine",
    16: "Poltava, Ukraine",
    17: "Rivne, Ukraine",
    18: "Sumy, Ukraine",
    19: "Ternopil, Ukraine",
    20: "Kharkiv, Ukraine",
    21: "Kherson, Ukraine",
    22: "Khmelnytskyi, Ukraine",
    23: "Cherkasy, Ukraine",
    24: "Chernivtsi, Ukraine",
    25: "Chernihiv, Ukraine",
    26: "Kyiv, Ukraine",
}

def save_weather_data(data: json):
    os.makedirs("data/weather_forecast", exist_ok=True)
    now_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"data/weather_forecast/forecast_{now_str}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

def get_weather(location: str):
    url_base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    today = dt.datetime.now().date()
    tomorrow = today + dt.timedelta(days=1)

    url = f"{url_base_url}/{location}/{today}/{tomorrow}?unitGroup=metric&include=hours&key={VCRO_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        now = dt.datetime.now()
        next_24_hours = []

        for day in data.get("days", []):
            for hour in day.get("hours", []):
                hour_time = dt.datetime.strptime(f"{day['datetime']} {hour['datetime']}","%Y-%m-%d %H:%M:%S")
                if hour_time >= now:
                    next_24_hours.append(hour)

        next_24_hours = next_24_hours[:24]
        return next_24_hours

    else:
        raise Exception(f"{response.status_code}: {response.text}")

def get_weather_for_all_regions():
    result = {}

    for id, location in REGIONS.items():
        print(f"Fetching {location}...")
        try:
            result[id] = get_weather(location)
        except Exception as e:
            print(f"Error for {location}: {e}")
            result[id] = []

    return result

if __name__ == "__main__":
    result = get_weather_for_all_regions()
    save_weather_data(result)