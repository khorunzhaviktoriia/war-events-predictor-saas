import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

KYIV_TZ = "Europe/Kyiv"
EXPECTED_REGION_COUNT = 24
TG_DECAY_HALFLIFE_HOURS = 6.0
OUTPUT_FILENAME = "next_24h_predictions.json"

NEIGHBOURING_REGIONS = {
    1: [21],
    2: [6, 10, 11, 15, 22, 23, 24],
    3: [13, 17],
    4: [5, 8, 11, 14, 16, 20, 21],
    5: [4, 8, 12, 20],
    6: [2, 10, 17, 22],
    7: [9, 13],
    8: [4, 5, 21],
    9: [7, 13, 19, 24],
    10: [2, 6, 16, 23, 25],
    11: [2, 4, 14, 15, 16, 23],
    12: [5, 20],
    13: [3, 7, 9, 17, 19],
    14: [4, 11, 15, 21],
    15: [2, 11, 14],
    16: [4, 10, 11, 18, 20, 23, 25],
    17: [3, 6, 13, 19, 22],
    18: [16, 20, 25],
    19: [9, 13, 17, 22, 24],
    20: [4, 5, 12, 16, 18],
    21: [1, 4, 8, 14],
    22: [2, 6, 17, 19, 24],
    23: [2, 10, 11, 16],
    24: [2, 9, 19, 22],
    25: [10, 16, 18],
    26: [10],
}


@dataclass
class ProjectPaths:
    project_root: Path
    data_dir: Path
    runtime_dir: Path
    model_path: Path
    final_parquet: Path
    final_weather_csv: Path
    isw_csv: Path
    telegram_csv: Path
    forecast_weather_parquet: Path
    output_dir: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        runtime_dir = data_dir / "runtime"
        output_dir = data_dir / "predictions"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            runtime_dir=runtime_dir,
            model_path=project_root / "models" / "2__hist_gradient_boosting__v1.pkl",
            final_parquet=data_dir / "final_merged_dataset.parquet",
            final_weather_csv=data_dir / "final_weather.csv",
            isw_csv=data_dir / "isw_processed_svd.csv",
            telegram_csv=data_dir / "telegram_processed_svd.csv",
            forecast_weather_parquet=runtime_dir / "weather_forecast_processed.parquet",
            output_dir=output_dir,
        )


def log(msg: str) -> None:
    ts = pd.Timestamp.now(tz=KYIV_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {msg}")


def load_model_artifact(model_path: Path) -> dict[str, Any]:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    required = {"model", "threshold", "feature_names"}
    missing = required - set(model.keys())
    if missing:
        raise ValueError(f"Model missing keys: {sorted(missing)}")
    return model


def ensure_numeric_bool_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=feature_names, fill_value=0).copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.int8)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X


def safe_entropy_from_abs(values: pd.Series) -> float:
    arr = np.abs(values.to_numpy(dtype=float))
    total = arr.sum()
    if total <= 0:
        return 0.0
    probs = arr / total
    return float(-(probs * np.log(probs + 1e-9)).sum())


def ewm_last(series: pd.Series, span: int) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(float).ewm(span=span).mean().iloc[-1])


def zscore_last(series: pd.Series, window: int) -> float:
    if series.empty:
        return 0.0

    s = series.astype(float)
    mean = s.rolling(window, min_periods=1).mean().iloc[-1]
    std = s.rolling(window, min_periods=1).std().iloc[-1]

    if pd.isna(std) or float(std) == 0.0:
        return 0.0

    return float((s.iloc[-1] - mean) / (std + 1e-9))


def load_history(paths: ProjectPaths) -> pd.DataFrame:
    hist = pd.read_parquet(paths.final_parquet)
    hist["datetime_hour"] = pd.to_datetime(hist["datetime_hour"], errors="coerce")
    hist["region_id"] = pd.to_numeric(hist["region_id"], errors="coerce").astype("Int64")
    hist = hist[hist["datetime_hour"].notna() & hist["region_id"].notna()].copy()
    hist["region_id"] = hist["region_id"].astype(int)
    hist = hist.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last")
    return hist.reset_index(drop=True)


def load_forecast_weather(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_parquet(paths.forecast_weather_parquet)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["datetime_hour"].notna() & df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last")
    return df.reset_index(drop=True)


def load_processed_weather(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.final_weather_csv, low_memory=False)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["datetime_hour"].notna() & df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last")
    return df.reset_index(drop=True)


def load_isw_daily(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.isw_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()].copy()
    return df.sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)


def load_tg_hourly(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.telegram_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    tg_cols = [c for c in df.columns if c.startswith("tg_topic_")]
    if not tg_cols:
        raise ValueError("telegram_processed_svd.csv has no tg_topic_* columns")
    df["datetime_hour"] = df["date"].dt.floor("h")
    hourly = df.groupby("datetime_hour", as_index=False)[tg_cols].mean()
    return hourly.sort_values("datetime_hour").reset_index(drop=True)


def build_weather_lookup(actual_weather: pd.DataFrame, forecast_weather: pd.DataFrame) -> tuple[dict[tuple[pd.Timestamp, int], dict[str, Any]], list[str], list[str]]:
    combined = pd.concat([actual_weather, forecast_weather], ignore_index=True)
    combined = combined.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last")
    hour_weather_cols = [c for c in combined.columns if c.startswith("hour_")]
    day_weather_cols = [
        c for c in combined.columns
        if c.startswith("day_") and c not in {"day_datetime", "day_sunrise", "day_sunset", "day_moonphase", "day_of_week"}
    ]
    lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]] = {}
    for row in combined.to_dict("records"):
        lookup[(pd.Timestamp(row["datetime_hour"]), int(row["region_id"]))] = row
    return lookup, hour_weather_cols, day_weather_cols


def build_isw_lookup(isw_daily: pd.DataFrame) -> tuple[dict[pd.Timestamp, dict[str, float]], list[str], pd.Timestamp | None]:
    isw_cols = [c for c in isw_daily.columns if c.startswith("isw_topic_")]
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for row in isw_daily.to_dict("records"):
        lookup[pd.Timestamp(row["date"]).normalize()] = {col: float(row[col]) for col in isw_cols}
    last_date = max(lookup.keys()) if lookup else None
    return lookup, isw_cols, last_date


def build_tg_lookup(tg_hourly: pd.DataFrame) -> tuple[dict[pd.Timestamp, dict[str, float]], list[str], pd.Timestamp | None]:
    tg_cols = [c for c in tg_hourly.columns if c.startswith("tg_topic_")]
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for row in tg_hourly.to_dict("records"):
        lookup[pd.Timestamp(row["datetime_hour"])] = {col: float(row[col]) for col in tg_cols}
    last_dt = max(lookup.keys()) if lookup else None
    return lookup, tg_cols, last_dt


def validate_inputs(history: pd.DataFrame, forecast_weather: pd.DataFrame, model_features: list[str]) -> None:
    if history.empty:
        raise ValueError("Historical dataset is empty")
    if forecast_weather.empty:
        raise ValueError("Forecast weather dataset is empty")
    if not model_features:
        raise ValueError("Model feature_names is empty")

    hist_regions = sorted(history["region_id"].dropna().unique().tolist())
    forecast_regions = sorted(forecast_weather["region_id"].dropna().unique().tolist())
    if hist_regions != forecast_regions:
        raise ValueError("History and forecast weather have different region_id sets")
    if len(hist_regions) != EXPECTED_REGION_COUNT:
        raise ValueError(f"Expected {EXPECTED_REGION_COUNT} regions, got {len(hist_regions)}")

    future_hours = sorted(forecast_weather["datetime_hour"].drop_duplicates())
    if len(future_hours) != 24:
        raise ValueError(f"Expected exactly 24 forecast hours, got {len(future_hours)}")

    expected_rows = EXPECTED_REGION_COUNT * 24
    if len(forecast_weather) != expected_rows:
        raise ValueError(f"Expected {expected_rows} forecast rows, got {len(forecast_weather)}")


def get_region_meta(history: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["region_id", "region_key", "city_name"] if c in history.columns]
    meta = history.sort_values("datetime_hour").groupby("region_id", as_index=False)[cols[1:]].last()
    return meta.sort_values("region_id").reset_index(drop=True)


def split_history_by_region(history: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for region_id, part in history.groupby("region_id", sort=False):
        out[int(region_id)] = part.sort_values("datetime_hour").reset_index(drop=True)
    return out


def build_hourly_totals(history: pd.DataFrame) -> dict[pd.Timestamp, float]:
    totals = history.groupby("datetime_hour")["alarm_active"].sum()
    return {pd.Timestamp(idx): float(val) for idx, val in totals.items()}


def latest_row_before(region_hist: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    region_hist = region_hist[region_hist["datetime_hour"] < dt]
    if region_hist.empty:
        return None
    return region_hist.iloc[-1]


def get_alarm_value(region_hist: pd.DataFrame, dt: pd.Timestamp) -> int:
    row = region_hist.loc[region_hist["datetime_hour"] == dt, "alarm_active"]
    return int(row.iloc[-1]) if not row.empty else 0


def get_hours_since_last_alarm(region_hist: pd.DataFrame, dt: pd.Timestamp) -> float:
    hist = region_hist[region_hist["datetime_hour"] < dt]
    if hist.empty:
        return 0.0
    positives = hist[hist["alarm_active"] == 1]
    if positives.empty:
        return float(len(hist))
    last_alarm_dt = positives["datetime_hour"].iloc[-1]
    diff_hours = int((dt - last_alarm_dt) / pd.Timedelta(hours=1)) - 1
    return float(max(diff_hours, 0))


def get_alarms_last_24h(region_hist: pd.DataFrame, dt: pd.Timestamp) -> float:
    start = dt - pd.Timedelta(hours=24)
    hist = region_hist[(region_hist["datetime_hour"] >= start) & (region_hist["datetime_hour"] < dt)]
    return float(hist["alarm_active"].sum())


def get_total_active_alarms_lag1(hourly_totals: dict[pd.Timestamp, float], dt: pd.Timestamp) -> float:
    return float(hourly_totals.get(dt - pd.Timedelta(hours=1), 0.0))


def get_neighbour_alarms(history_by_region: dict[int, pd.DataFrame], region_id: int, dt: pd.Timestamp) -> float:
    prev_dt = dt - pd.Timedelta(hours=1)
    neighbours = NEIGHBOURING_REGIONS.get(region_id, [])
    total = 0.0
    for nb in neighbours:
        region_hist = history_by_region.get(nb)
        if region_hist is None:
            continue
        total += get_alarm_value(region_hist, prev_dt)
    return float(total)


def get_weather_values(
    weather_lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]],
    region_id: int,
    target_dt: pd.Timestamp,
    hour_weather_cols: list[str],
    day_weather_cols: list[str],
    fallback_row: pd.Series | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    prev_hour_row = weather_lookup.get((target_dt - pd.Timedelta(hours=1), region_id))
    prev_day_row = weather_lookup.get((target_dt - pd.Timedelta(hours=24), region_id))

    for col in hour_weather_cols:
        if prev_hour_row is not None and col in prev_hour_row:
            values[col] = prev_hour_row[col]
        elif fallback_row is not None and col in fallback_row.index:
            values[col] = fallback_row[col]
        else:
            values[col] = 0

    for col in day_weather_cols:
        if prev_day_row is not None and col in prev_day_row:
            values[col] = prev_day_row[col]
        elif fallback_row is not None and col in fallback_row.index:
            values[col] = fallback_row[col]
        else:
            values[col] = 0

    return values


def get_shifted_isw_values(
    isw_lookup: dict[pd.Timestamp, dict[str, float]],
    isw_cols: list[str],
    last_isw_date: pd.Timestamp | None,
    target_dt: pd.Timestamp,
    fallback_row: pd.Series | None,
) -> dict[str, float]:
    src_date = (target_dt - pd.Timedelta(hours=24)).normalize()
    raw = isw_lookup.get(src_date)
    if raw is not None:
        return {col: float(raw.get(col, 0.0)) for col in isw_cols}
    if last_isw_date is not None and last_isw_date in isw_lookup:
        raw = isw_lookup[last_isw_date]
        return {col: float(raw.get(col, 0.0)) for col in isw_cols}
    if fallback_row is not None:
        return {col: float(fallback_row.get(col, 0.0)) for col in isw_cols}
    return {col: 0.0 for col in isw_cols}


def tg_decay_factor(missing_hours: int) -> float:
    if missing_hours <= 0:
        return 1.0
    return float(0.5 ** (missing_hours / TG_DECAY_HALFLIFE_HOURS))


def get_shifted_tg_values(
    tg_lookup: dict[pd.Timestamp, dict[str, float]],
    tg_cols: list[str],
    last_real_tg_dt: pd.Timestamp | None,
    target_dt: pd.Timestamp,
    fallback_row: pd.Series | None,
) -> dict[str, float]:
    src_dt = target_dt - pd.Timedelta(hours=1)
    raw = tg_lookup.get(src_dt)
    if raw is not None:
        return {col: float(raw.get(col, 0.0)) for col in tg_cols}

    if last_real_tg_dt is not None and last_real_tg_dt in tg_lookup:
        base = tg_lookup[last_real_tg_dt]
        missing_hours = int((src_dt - last_real_tg_dt) / pd.Timedelta(hours=1))
        decay = tg_decay_factor(missing_hours)
        return {col: float(base.get(col, 0.0)) * decay for col in tg_cols}

    if fallback_row is not None:
        return {col: float(fallback_row.get(col, 0.0)) for col in tg_cols}
    return {col: 0.0 for col in tg_cols}


def add_topic_derived_features(
    row: dict[str, Any],
    region_hist: pd.DataFrame,
    isw_cols: list[str],
    tg_cols: list[str],
) -> None:
    if isw_cols:
        isw_vals = pd.Series({c: row.get(c, 0.0) for c in isw_cols}, dtype=float)
        row["isw_total_intensity"] = float(np.abs(isw_vals).sum())
        row["isw_topic_std"] = float(np.abs(isw_vals).std()) if len(isw_vals) else 0.0
        row["isw_topic_max"] = float(np.abs(isw_vals).max()) if len(isw_vals) else 0.0
        row["isw_topic_mean"] = float(np.abs(isw_vals).mean()) if len(isw_vals) else 0.0
        row["isw_topic_entropy"] = safe_entropy_from_abs(isw_vals)

        if "isw_total_intensity" in region_hist.columns:
            intensity_series = pd.concat([
                region_hist["isw_total_intensity"].astype(float),
                pd.Series([row["isw_total_intensity"]]),
            ], ignore_index=True)
            row["isw_velocity_24h"] = float(intensity_series.diff(24).iloc[-1]) if len(intensity_series) > 24 else 0.0
            row["isw_intensity_ema"] = ewm_last(intensity_series, span=24)

    if tg_cols:
        tg_vals = pd.Series({c: row.get(c, 0.0) for c in tg_cols}, dtype=float)
        row["tg_total_intensity"] = float(np.abs(tg_vals).sum())
        row["tg_topic_std"] = float(np.abs(tg_vals).std()) if len(tg_vals) else 0.0
        row["tg_topic_max"] = float(np.abs(tg_vals).max()) if len(tg_vals) else 0.0
        row["tg_topic_entropy"] = safe_entropy_from_abs(tg_vals)

        if "tg_total_intensity" in region_hist.columns:
            intensity_series = pd.concat([
                region_hist["tg_total_intensity"].astype(float),
                pd.Series([row["tg_total_intensity"]]),
            ], ignore_index=True)
            row["tg_velocity_3h"] = float(intensity_series.diff(3).iloc[-1]) if len(intensity_series) > 3 else 0.0
            row["tg_intensity_ema_6h"] = ewm_last(intensity_series, span=6)
            row["tg_intensity_zscore"] = zscore_last(intensity_series, window=24)


def build_future_hour_rows(
    target_dt: pd.Timestamp,
    history_by_region: dict[int, pd.DataFrame],
    hourly_totals: dict[pd.Timestamp, float],
    region_meta: pd.DataFrame,
    weather_lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]],
    hour_weather_cols: list[str],
    day_weather_cols: list[str],
    isw_lookup: dict[pd.Timestamp, dict[str, float]],
    isw_cols: list[str],
    last_isw_date: pd.Timestamp | None,
    tg_lookup: dict[pd.Timestamp, dict[str, float]],
    tg_cols: list[str],
    last_real_tg_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for meta in region_meta.to_dict("records"):
        region_id = int(meta["region_id"])
        region_hist = history_by_region[region_id]
        prev_row = latest_row_before(region_hist, target_dt)

        row: dict[str, Any] = {
            "datetime_hour": target_dt,
            "region_id": region_id,
            "region_key": meta.get("region_key"),
            "city_name": meta.get("city_name"),
            "day_datetime": target_dt.strftime("%Y-%m-%d"),
            "day_of_week": int(target_dt.dayofweek),
            "hour": int(target_dt.hour),
            "is_weekend": int(target_dt.dayofweek in [5, 6]),
            "is_night": int((target_dt.hour >= 23) or (target_dt.hour <= 6)),
            "alarm_minutes_in_hour": 0.0,
            "alarm_active": 0,
            "alarm_lag_1": float(get_alarm_value(region_hist, target_dt - pd.Timedelta(hours=1))),
            "alarm_lag_3": float(get_alarm_value(region_hist, target_dt - pd.Timedelta(hours=3))),
            "alarm_lag_6": float(get_alarm_value(region_hist, target_dt - pd.Timedelta(hours=6))),
            "alarm_lag_12": float(get_alarm_value(region_hist, target_dt - pd.Timedelta(hours=12))),
            "alarms_in_last_24h": float(get_alarms_last_24h(region_hist, target_dt)),
            "total_active_alarms_lag1": float(get_total_active_alarms_lag1(hourly_totals, target_dt)),
            "neighbour_alarms": float(get_neighbour_alarms(history_by_region, region_id, target_dt)),
            "hours_since_last_alarm": float(get_hours_since_last_alarm(region_hist, target_dt)),
        }

        row.update(get_weather_values(weather_lookup, region_id, target_dt, hour_weather_cols, day_weather_cols, prev_row))
        row.update(get_shifted_isw_values(isw_lookup, isw_cols, last_isw_date, target_dt, prev_row))
        row.update(get_shifted_tg_values(tg_lookup, tg_cols, last_real_tg_dt, target_dt, prev_row))
        add_topic_derived_features(row, region_hist, isw_cols, tg_cols)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)


def append_predictions_to_state(
    hour_rows: pd.DataFrame,
    pred: np.ndarray,
    history_by_region: dict[int, pd.DataFrame],
    hourly_totals: dict[pd.Timestamp, float],
) -> None:
    hour_rows = hour_rows.copy()
    hour_rows["alarm_active"] = pred.astype(int)
    target_dt = pd.Timestamp(hour_rows["datetime_hour"].iloc[0])
    hourly_totals[target_dt] = float(pred.sum())

    for region_id, part in hour_rows.groupby("region_id", sort=False):
        rid = int(region_id)
        history_by_region[rid] = pd.concat([history_by_region[rid], part], ignore_index=True)
        history_by_region[rid] = history_by_region[rid].sort_values("datetime_hour").reset_index(drop=True)


def model_train_time_from_file(model_path: Path) -> str:
    if not model_path.exists():
        return ""
    return pd.Timestamp(model_path.stat().st_mtime, unit="s", tz=KYIV_TZ).isoformat()


def save_predictions(predictions: pd.DataFrame, paths: ProjectPaths, threshold: float) -> Path:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.output_dir / OUTPUT_FILENAME

    compact = predictions[[
        "datetime_hour",
        "region_id",
        "region_key",
        "city_name",
        "prediction_proba",
        "prediction",
    ]].copy()
    compact["datetime_hour"] = pd.to_datetime(compact["datetime_hour"], errors="coerce")

    regions_forecast: dict[str, dict[str, Any]] = {}
    for region_key, part in compact.groupby("region_key", sort=False):
        part = part.sort_values("datetime_hour").reset_index(drop=True)
        first = part.iloc[0]
        forecast_map = {
            dt.strftime("%Y-%m-%d %H:%M"): bool(pred)
            for dt, pred in zip(part["datetime_hour"], part["prediction"])
        }
        forecast_proba_map = {
            dt.strftime("%Y-%m-%d %H:%M"): float(proba)
            for dt, proba in zip(part["datetime_hour"], part["prediction_proba"])
        }

        regions_forecast[str(region_key)] = {
            "region_id": int(first["region_id"]),
            "city_name": str(first["city_name"]),
            "forecast": forecast_map,
            "forecast_proba": forecast_proba_map,
        }

    payload = {
        "last_model_train_time": model_train_time_from_file(paths.model_path),
        "last_prediction_time": pd.Timestamp.now(tz=KYIV_TZ).isoformat(),
        "threshold": threshold,
        "hours": int(compact["datetime_hour"].nunique()),
        "regions": int(compact["region_id"].nunique()),
        "rows": int(len(compact)),
        "regions_forecast": regions_forecast,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    paths = ProjectPaths.default()

    log("Loading model artifact")
    model_artifact  = load_model_artifact(paths.model_path)
    model = model_artifact["model"]
    threshold = float(model_artifact["threshold"])
    feature_names = list(model_artifact["feature_names"])

    log("Loading historical context and exogenous sources")
    history = load_history(paths)
    forecast_weather = load_forecast_weather(paths)
    actual_weather = load_processed_weather(paths)
    isw_daily = load_isw_daily(paths)
    tg_hourly = load_tg_hourly(paths)

    validate_inputs(history, forecast_weather, feature_names)

    region_meta = get_region_meta(history)
    history_by_region = split_history_by_region(history)
    hourly_totals = build_hourly_totals(history)

    weather_lookup, hour_weather_cols, day_weather_cols = build_weather_lookup(actual_weather, forecast_weather)
    isw_lookup, isw_cols, last_isw_date = build_isw_lookup(isw_daily)
    tg_lookup, tg_cols, last_real_tg_dt = build_tg_lookup(tg_hourly)

    future_hours = sorted(forecast_weather["datetime_hour"].drop_duplicates())
    all_preds: list[pd.DataFrame] = []

    log(f"Starting recursive prediction for {len(future_hours)} forecast hours")
    for target_dt in future_hours:
        hour_rows = build_future_hour_rows(
            target_dt=pd.Timestamp(target_dt),
            history_by_region=history_by_region,
            hourly_totals=hourly_totals,
            region_meta=region_meta,
            weather_lookup=weather_lookup,
            hour_weather_cols=hour_weather_cols,
            day_weather_cols=day_weather_cols,
            isw_lookup=isw_lookup,
            isw_cols=isw_cols,
            last_isw_date=last_isw_date,
            tg_lookup=tg_lookup,
            tg_cols=tg_cols,
            last_real_tg_dt=last_real_tg_dt,
        )

        X_hour = ensure_numeric_bool_matrix(hour_rows, feature_names)
        proba = model.predict_proba(X_hour)[:, 1]
        pred = (proba >= threshold).astype(int)

        hour_rows["prediction_proba"] = proba
        hour_rows["prediction"] = pred
        all_preds.append(hour_rows.copy())

        append_predictions_to_state(hour_rows, pred, history_by_region, hourly_totals)
        log(f"Predicted {pd.Timestamp(target_dt)}")

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions = predictions.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    out_path = save_predictions(predictions, paths, threshold)

    log(f"Saved predictions to: {out_path}")
    log(f"Rows={len(predictions)} | hours={predictions['datetime_hour'].nunique()} | regions={predictions['region_id'].nunique()}")


if __name__ == "__main__":
    main()
