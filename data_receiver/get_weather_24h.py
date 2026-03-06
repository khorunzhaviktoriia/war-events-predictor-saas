import datetime as dt
import json
import requests
import os
import argparse

VCRO_KEY = ""

def save_weather_data(data: json, location: str):
    os.makedirs("data/weather_forecast", exist_ok=True)
    now_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"data/weather_forecast/forecast_{location}_{now_str}.json"
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

        save_weather_data(next_24_hours, location)
        return next_24_hours

    else:
        raise Exception(f"{response.status_code}: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get 24h weather forecast")
    parser.add_argument("location", type=str, help="Location, e.g. 'Kyiv, Ukraine'")

    args = parser.parse_args()
    location = args.location

    result = get_weather(location)
    print(json.dumps(result, indent=4))