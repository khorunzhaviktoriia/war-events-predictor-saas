from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta
import random
from app.services.storage import save_forecast

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "final_merged_dataset.parquet"


def load_regions():
    df = pd.read_parquet(DATASET_PATH)

    region_map = (
        df[["region_id", "city_name"]]
        .drop_duplicates()
        .sort_values("region_id")
    )

    return region_map.to_dict(orient="records")


def generate_mock_forecast():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    regions = load_regions()

    regions_forecast = {}

    for item in regions:
        region_id = str(item["region_id"])
        hourly_forecast = {}

        for i in range(24):
            forecast_time = now + timedelta(hours=i)
            hour_str = forecast_time.strftime("%H:%M")
            hourly_forecast[hour_str] = random.choice([True, False])

        regions_forecast[region_id] = hourly_forecast

    forecast = {
        "last_model_train_time": "2026-03-25T10:15:30Z",
        "last_prediction_time": now.isoformat().replace("+00:00", "Z"),
        "model_name": "mock_model_v1",
        "forecast_horizon_hours": 24,
        "regions_forecast": regions_forecast
    }

    return forecast


def update_forecast_file():
    forecast = generate_mock_forecast()
    save_forecast(forecast)
    return forecast
