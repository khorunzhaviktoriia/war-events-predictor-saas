# SETUP:
#   1. git clone <repo_url>
#   2. cd <repo_folder>
#   3. pip install -r requirements_for_inference.txt
#   4. python inference.py --model "path/to/1__hist_gradient_boosting__v1.pkl"

import pickle
import argparse
import numpy as np
import pandas as pd

TEST_DATA = {
    "region_id": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "day_tempmax": [12.0, 12.0, 8.5, 15.0, 15.0, 5.0, 5.0, 20.0, 20.0, 10.0],
    "day_tempmin": [2.0, 2.0, 1.0, 6.0, 6.0, -1.0, -1.0, 11.0, 11.0, 3.0],
    "day_temp": [7.0, 7.0, 4.5, 10.0, 10.0, 2.0, 2.0, 15.0, 15.0, 6.0],
    "day_dew": [3.0, 3.0, 2.0, 5.0, 5.0, -2.0, -2.0, 8.0, 8.0, 2.0],
    "day_humidity": [70.0, 70.0, 75.0, 65.0, 65.0, 80.0, 80.0, 60.0, 60.0, 72.0],
    "day_precip": [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5],
    "day_precipprob": [10.0, 10.0, 60.0, 5.0, 5.0, 40.0, 40.0, 0.0, 0.0, 30.0],
    "day_precipcover": [0.0, 0.0, 20.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 5.0],
    "day_snow": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    "day_windgust": [20.0, 20.0, 25.0, 15.0, 15.0, 30.0, 30.0, 10.0, 10.0, 22.0],
    "day_cloudcover": [50.0, 50.0, 80.0, 30.0, 30.0, 90.0, 90.0, 10.0, 10.0, 60.0],
    "day_moonphase": [0.5, 0.5, 0.6, 0.3, 0.3, 0.7, 0.7, 0.1, 0.1, 0.4],
    "hour_temp": [5.0, 6.0, 3.0, 9.0, 10.0, 1.0, 0.0, 14.0, 15.0, 5.0],
    "hour_feelslike": [2.0, 3.0, 0.0, 6.0, 7.0, -3.0, -4.0, 12.0, 13.0, 2.0],
    "hour_humidity": [72.0, 70.0, 78.0, 66.0, 64.0, 82.0, 84.0, 58.0, 56.0, 74.0],
    "hour_dew": [1.0, 1.5, 0.5, 4.0, 4.5, -3.0, -3.5, 6.0, 6.5, 1.0],
    "hour_precip": [0.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1],
    "hour_precipprob": [5.0, 5.0, 50.0, 3.0, 3.0, 35.0, 20.0, 0.0, 0.0, 25.0],
    "hour_snow": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    "hour_snowdepth": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    "hour_windgust": [15.0, 16.0, 20.0, 10.0, 11.0, 25.0, 26.0, 8.0, 8.0, 18.0],
    "hour_windspeed": [10.0, 11.0, 14.0, 7.0, 8.0, 18.0, 19.0, 5.0, 5.0, 12.0],
    "hour_winddir": [180.0, 185.0, 200.0, 270.0, 275.0, 90.0, 95.0, 45.0, 50.0, 160.0],
    "hour_pressure": [1015.0, 1015.0, 1010.0, 1018.0, 1018.0, 1008.0, 1008.0, 1020.0, 1020.0, 1013.0],
    "hour_cloudcover": [55.0, 50.0, 85.0, 25.0, 20.0, 95.0, 95.0, 5.0, 5.0, 65.0],
    "hour_solarradiation": [100.0, 120.0, 50.0, 200.0, 210.0, 30.0, 20.0, 350.0, 360.0, 80.0],
    "hour_solarenergy": [0.36, 0.43, 0.18, 0.72, 0.76, 0.11, 0.07, 1.26, 1.30, 0.29],
    "hour_uvindex": [1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 1.0],
    "hour_conditions_simple_Clear": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    "hour_conditions_simple_Cloudy": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    "hour_conditions_simple_Rain": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "hour_conditions_simple_Snow": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "day_of_week": [0, 0, 1, 2, 2, 3, 3, 4, 4, 5],
    "hour": [8, 9, 14, 10, 11, 3, 4, 15, 16, 7],
    "alarm_lag_1": [0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    "alarm_lag_3": [1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    "alarm_lag_6": [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    "alarm_lag_12": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    "alarms_in_last_24h": [3, 4, 2, 1, 2, 5, 5, 0, 1, 2],
    "is_weekend": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    "is_night": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "total_active_alarms_lag1": [5, 6, 3, 2, 3, 8, 8, 1, 2, 4],
    "neighbour_alarms": [2, 2, 1, 1, 2, 3, 3, 0, 0, 1],
    "hours_since_last_alarm": [1, 0, 3, 5, 0, 0, 0, 10, 8, 2],
    "isw_total_intensity": [0.5, 0.5, 0.3, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6, 0.3],
    "isw_topic_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "isw_topic_max": [0.3, 0.3, 0.2, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4, 0.2],
    "isw_topic_mean": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "isw_topic_entropy": [3.5, 3.5, 3.4, 3.5, 3.5, 3.4, 3.4, 3.6, 3.6, 3.4],
    "isw_velocity_24h": [0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.2, 0.2, 0.0],
    "isw_intensity_ema": [0.4, 0.4, 0.3, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5, 0.3],
    "tg_total_intensity": [0.6, 0.7, 0.3, 0.5, 0.6, 0.8, 0.8, 0.2, 0.3, 0.4],
    "tg_topic_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "tg_topic_max": [0.4, 0.4, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 0.3],
    "tg_topic_entropy": [3.2, 3.2, 3.1, 3.2, 3.2, 3.3, 3.3, 3.0, 3.0, 3.1],
    "tg_velocity_3h": [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1, 0.1],
    "tg_intensity_ema_6h": [0.5, 0.5, 0.3, 0.4, 0.5, 0.7, 0.7, 0.2, 0.2, 0.4],
    "tg_intensity_zscore": [0.5, 0.6, 0.0, 0.3, 0.5, 1.2, 1.2, -0.5, -0.3, 0.2],
}

# isw_topic_0..149 and tg_topic_0..249 — set to 0.0 (neutral baseline)
for i in range(150):
    TEST_DATA[f"isw_topic_{i}"] = [0.0] * 10
for i in range(250):
    TEST_DATA[f"tg_topic_{i}"] = [0.0] * 10


def load_model(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_names = bundle["feature_names"]

    # reorder columns to match training order
    X = df[feature_names].copy()

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df[["region_id"]].copy() if "region_id" in df.columns else pd.DataFrame()
    result["alarm_probability"] = proba.round(4)
    result["alarm_predicted"] = pred
    result["threshold_used"] = threshold
    return result


def main():
    parser = argparse.ArgumentParser(description="Air raid alarm inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="../data/1__hist_gradient_boosting__v1.pkl",
        help="Path to the pickle model file",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    bundle = load_model(args.model)
    print(f"Model loaded. Threshold: {bundle['threshold']}, Features: {len(bundle['feature_names'])}")

    df = pd.DataFrame(TEST_DATA)
    print(f"\nTest dataframe shape: {df.shape}")

    results = predict(bundle, df)
    print("\n=== Predictions ===")
    print(results.to_string(index=True))


if __name__ == "__main__":
    main()
