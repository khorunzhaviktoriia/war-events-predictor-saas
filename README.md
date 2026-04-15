# Air Raid Alarm Forecasting in Ukraine

This repository contains a practical pipeline for collecting data, updating a historical dataset, generating **next-24-hour air raid alarm forecasts**, and periodically retraining the production model.

## Project Overview

The goal of this project is to predict whether an air raid alarm will be active for each region at hourly granularity.

The repository includes:
- training and evaluation of 6 models
- scripts to collect raw snapshots from multiple sources
- preprocessing and merge logic
- a recursive forecasting script for the next 24 hours
- a retraining script that compares a candidate model with the current production model


## Data Sources

The project combines several data sources:

- **weather**
  - historical weather data
  - 24-hour weather forecast snapshots
- **air alarms**
  - hourly alarm snapshots/processed alarm history
- **Telegram**
  - scraped messages 
- **ISW reports**
  - daily text reports

## Local Data Note

The `data/` directory should be treated as **local project data** and is **not intended to be stored in GitHub**.

It contains files such as:

- `data/raw_snapshots/`
- `data/alarms-merged.csv`
- `data/all_weather_by_hour_2023-2026_v1.csv`
- `data/regions.csv`

If you clone the repository, you must prepare the local `data/` directory yourself.


## 2. Notebook workflow/research reproduction

These notebooks were used for research, exploratory analysis, feature preparation, and model experiments.

Suggested order:

1. `forecasting/eda_nlp_preparation.ipynb`  
   EDA, text preprocessing, and source-level preparation.

2. `forecasting/data_merge_feature_engineering.ipynb`  
   Dataset merge and feature engineering.

3. `data_receiver/fetch_donetsk_weather.py`
    The provided data `data/all_weather_by_hour_2023-2026_v1.csv` did not include Donetsk for the last year, so we collected this data ourselves

4. `forecasting/donetsk_weather_patch.ipynb`

5. Model notebooks

## How to Run

Run commands from the repository root.

### 1. Run collectors

```bash
python runners/run_collectors.py
```

This runner calls:

- `data_receiver/collect_absent_data.py`
- `data_receiver/telegram_scraper_cron.py`
- `data_receiver/get_weather_24h_OpenMeteo.py`

These scripts collect or refresh raw snapshots under `data/raw_snapshots/`.

### 2. Update the historical dataset

After collecting snapshots, run:

```bash
python runners/update_final_merged_dataset.py
```

This script:

- reads raw snapshots
- preprocesses new rows
- appends them to processed source table
- rebuilds `data/final_merged_dataset.parquet`
- prepares runtime weather forecast input for inference

### 3. Run hourly forecast

```bash
python runners/predict_next_24h.py
```

This script:

- loads the production model artifact
- loads the historical merged dataset
- loads prepared weather forecast input
- generates recursive predictions for the next 24 hours
- saves the result to:

```text
data/predictions/next_24h_predictions.json
```

### 4. Run retraining

```bash
python runners/retrain_top_model.py
```

This script:

- reads the latest `data/final_merged_dataset.parquet`
- retrains the top model
- compares the candidate model with the current production model
- replaces the active model only if the new one is not worse
- saves backup artifacts and retraining logs in `models/backups/` and `models/retrain_logs/`

## Prediction Output Format

The forecasting script saves predictions as a JSON file. The output contains metadata and per-region forecasts.

Example structure:

```json
{
  "last_model_train_time": "2026-04-15T03:03:03+03:00",
  "last_prediction_time": "2026-04-15T06:00:17+03:00",
  "threshold": 0.6,
  "hours": 24,
  "regions": 24,
  "rows": 576,
  "regions_forecast": {
    "Вінницька": {
      "region_id": 1,
      "city_name": "Vinnytsia",
      "forecast": {
        "2026-04-14 20:00": true,
        "2026-04-14 21:00": true
      },
      "forecast_proba": {
        "2026-04-14 20:00": 0.9314,
        "2026-04-14 21:00": 0.9339
      }
    }
  }
}
```

## Backend Overview

The project also includes a simple serving idea where forecast data is read from a saved prediction file instead of recomputing model inference for every API request.
