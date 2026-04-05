import json

FILE_PATH = "data/forecast_latest.json"


def load_forecast():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_forecast(data):
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
