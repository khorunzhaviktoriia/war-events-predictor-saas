from pathlib import Path
import pandas as pd
from app.services.storage import save_forecast, load_forecast

BASE_DIR = Path(__file__).resolve().parents[4]
DATASET_PATH = BASE_DIR / "data" / "final_merged_dataset.parquet"
REGIONS_PATH = BASE_DIR / "data" / "regions.csv"


def load_regions():
    df = pd.read_csv(REGIONS_PATH)

    region_map = (
        df[["region_id", "city_name"]]
        .drop_duplicates()
        .sort_values("region_id")
    )

    return region_map.to_dict(orient="records")


def normalize_forecast(raw_data):
    result = {}

    for region_name, region_data in raw_data["regions_forecast"].items():
        region_id = str(region_data["region_id"])
        forecast = {}

        for dt, proba in region_data["forecast_proba"].items():
            time = dt[11:16]
            forecast[time] = round(proba, 3)

        result[region_id] = forecast

    return {
        "last_model_train_time": raw_data["last_model_train_time"],
        "last_prediction_time": raw_data["last_prediction_time"],
        "model_name": "real_model_v2",
        "forecast_horizon_hours": raw_data.get("hours", 24),
        "regions_forecast": result
    }


def update_forecast_file():
    forecast = load_forecast()
    save_forecast(forecast)
    return forecast