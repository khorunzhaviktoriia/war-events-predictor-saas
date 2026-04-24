# Air Alarm Forecasting in Ukraine

This repository contains a machine learning pipeline for predicting whether an air alarm will be active in Ukrainian regions at hourly granularity. The project includes historical data preparation, model experiments, automated data collection, hourly dataset updates, next-24-hour prediction, model retraining, and a simple web interface.

The project was developed as a university project. It is not an official warning system.

## Table of Contents

- [Project Overview and Data Flow](#project-overview-and-data-flow)
- [Local Data and Artifacts](#local-data-and-artifacts)
- [Installation](#installation)
- [How to Start](#how-to-start)
- [Runtime Pipeline](#runtime-pipeline)
- [Prediction Output](#prediction-output)
- [Backend and Frontend](#backend-and-frontend)
- [System Diagram](#system-diagram)
- [User Interface](#user-interface)

## Project Overview and Data Flow

The project predicts the binary target `alarm_active` for each region and each hour:

- `1` - an air alarm is active during the hour;
- `0` - no air alarm is active during the hour.

The project has two main parts:

### 1. Historical preparation

The initial historical datasets are prepared in notebooks. This stage is used to explore, clean, transform, and merge the historical data.

Main steps:

1. Prepare weather, alarm, ISW, and Telegram datasets.
2. Apply NLP preprocessing to ISW reports and Telegram messages.
3. Save vectorizers and SVD models into `data/nlp_artifacts/`.
4. Merge all processed sources into `data/final_merged_dataset.parquet`.
5. Train and compare 6 models.
6. Save the selected production model into `models/`.

### 2. Runtime automation

After the historical dataset and artifacts exist locally, the runtime pipeline can update the project automatically.

The runtime flow is:

```text
run_collectors.py
        ↓
update_final_merged_dataset.py
        ↓
predict_next_24h.py
        ↓
retrain_top_model.py   (periodically)
```

In simple terms:

1. `run_collectors.py` collects new raw snapshots.
2. `update_final_merged_dataset.py` preprocesses the snapshots and rebuilds the final merged dataset.
3. `predict_next_24h.py` generates a 24-hour forecast.
4. `retrain_top_model.py` trains a new model and replaces the production model only if the new one is not worse.

---

## Local Data and Artifacts

The `data/` directory should be treated as **local project data** and is **not intended to be stored in GitHub**.

It contains files such as:

- `data/alarms-merged.csv`
- `data/all_weather_by_hour_2023-2026_v1.csv`
- `data/regions.csv`
- `data/telegram_data_v2`
- `data/isw_reports_v3`

If you clone the repository, you must prepare the local `data/` directory yourself, besides data from ISW and Telegram, you can collect it using scripts `data_receiver/collect_historical_isw_data_v2.py` and `data_receiver/telegram_scraper.py`

## Installation
Clone the repository:

```bash
git clone https://github.com/khorunzhaviktoriia/war-events-predictor-saas.git
cd war-events-predictor-saas
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## How to Start

### 1. Prepare historical data

Suggested order:

1. `forecasting/eda_nlp_preparation.ipynb`  
   EDA, text preprocessing, and source-level preparation.

2. `forecasting/data_merge_feature_engineering.ipynb`  
   Datasets merge and feature engineering. After this step, the main local dataset should be available as:
   ```text
   data/final_merged_dataset.parquet
   ```
   
3. `data_receiver/fetch_donetsk_weather.py`
    The provided data `data/all_weather_by_hour_2023-2026_v1.csv` did not include Donetsk for the last year, so we collected this data from another source(Open-Meteo Historical Weather API)

4. `forecasting/donetsk_weather_patch.ipynb`

5. Model notebooks
   You can only run the production model `forecasting/HistGradientBoostingClassifier.ipynb`.

### 2. Collect new data

```bash
python runners/run_collectors.py
```

This runner calls:

- `data_receiver/collect_absent_data.py`
- `data_receiver/telegram_scraper_cron.py`
- `data_receiver/get_weather_24h_OpenMeteo.py`

These scripts collect raw snapshots under `data/raw_snapshots/`. Also it collect 24-hour weather forecast.

### 3. Update the historical dataset(`data/final_merged_dataset.parquet`)

```bash
python runners/update_final_merged_dataset.py
```

This script:

- reads raw snapshots
- preprocesses new rows
- appends them to processed source table
- rebuilds `data/final_merged_dataset.parquet`
- prepares weather forecast input for inference

### 4. Run hourly forecast

```bash
python runners/predict_next_24h.py
```

Predictions are saved to:

```text
data/predictions/
```

### 5. Retrain the model

```bash
python runners/retrain_top_model.py
```

The retraining script trains a new model, compares it with the current production model, and replaces the production model only if the new one is not worse according to the selected validation logic.

## Runtime Pipeline

The expected runtime order is:

```text
run_collectors.py
        ↓
update_final_merged_dataset.py
        ↓
predict_next_24h.py
        ↓
retrain_top_model.py  # optional, not necessarily hourly
```

For hourly automation, the first three scripts can be scheduled with cron. Retraining can be scheduled less frequently because it is heavier than inference.

Example cron idea:

```cron
0 * * * * cd /path/to/war-events-predictor-saas && .venv/bin/python runners/run_collectors.py
10 * * * * cd /path/to/war-events-predictor-saas && .venv/bin/python runners/update_final_merged_dataset.py
20 * * * * cd /path/to/war-events-predictor-saas && .venv/bin/python runners/predict_next_24h.py
```
Retraining can be scheduled less often, for example once per day or once per week:

```cron
30 3 * * * cd /path/to/war-events-predictor-saas && /path/to/.venv/bin/python runners/retrain_top_model.py >> logs/retrain.log 2>&1
```

## Prediction Output

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

## Backend and Frontend

### Backend

```bash
cd app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Instead of **<0.0.0.0>**, write the address of the device on which you are running the commands.

## System Diagram

![System Diagram](images/system_diagram.png)

## User Interface

![User Interface](images/user_interface.png)
