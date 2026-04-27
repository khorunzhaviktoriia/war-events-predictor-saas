import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]
FILE_PATH = BASE_DIR / "data" / "predictions" / "next_24h_predictions_latest.json"


def _normalize_hour_key(raw_key: str) -> str:
    raw_key = str(raw_key)
    if len(raw_key) >= 5:
        return raw_key[-5:]
    return raw_key


def _to_legacy_forecast_payload(data: dict) -> dict:
    regions_forecast = {}

    for raw_region_key, region_block in data.get("regions_forecast", {}).items():
        if isinstance(region_block, dict) and all(
            isinstance(v, (int, float, bool)) for v in region_block.values()
        ):
            region_id = str(raw_region_key)
            regions_forecast[region_id] = {
                _normalize_hour_key(k): float(v)
                for k, v in region_block.items()
            }
            continue
        
        region_id = str(region_block.get("region_id", raw_region_key))
        proba_map = region_block.get("forecast_proba", {})

        regions_forecast[region_id] = {
            _normalize_hour_key(k): float(v)
            for k, v in proba_map.items()
        }

    return {
        "last_model_train_time": data.get("last_model_train_time", "—"),
        "last_prediction_time": data.get("last_prediction_time", "—"),
        "model_name": data.get("model_name", "real_model_v2"),
        "forecast_horizon_hours": data.get(
            "forecast_horizon_hours",
            data.get("hours", 24),
        ),
        "regions_forecast": regions_forecast,
    }


def load_forecast():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _to_legacy_forecast_payload(raw)


def save_forecast(data):
    FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
