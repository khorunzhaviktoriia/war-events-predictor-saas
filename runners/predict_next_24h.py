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


MC_N_SAMPLES = 10
MC_RANDOM_SEED = 42

HISTORY_TAIL_HOURS = 168


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
    tg_region_csv: Path
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
            tg_region_csv=data_dir / "tg_region_features.csv",
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
    hist = hist.sort_values(["datetime_hour", "region_id"]).drop_duplicates(
        ["datetime_hour", "region_id"], keep="last"
    )
    return hist.reset_index(drop=True)


def load_forecast_weather(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_parquet(paths.forecast_weather_parquet)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["datetime_hour"].notna() & df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(
        ["datetime_hour", "region_id"], keep="last"
    )
    return df.reset_index(drop=True)


def load_processed_weather(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.final_weather_csv, low_memory=False)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["datetime_hour"].notna() & df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(
        ["datetime_hour", "region_id"], keep="last"
    )
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


def load_tg_region(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.tg_region_csv, low_memory=False)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["datetime_hour"].notna() & df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    for col in ["tg_region_threat_count", "tg_region_allclear_count", "tg_region_mention_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(
        ["datetime_hour", "region_id"], keep="last"
    ).reset_index(drop=True)



def build_weather_lookup(
    actual_weather: pd.DataFrame, forecast_weather: pd.DataFrame
) -> tuple[dict[tuple[pd.Timestamp, int], dict[str, Any]], list[str], list[str]]:
    combined = pd.concat([actual_weather, forecast_weather], ignore_index=True)
    combined = combined.sort_values(["datetime_hour", "region_id"]).drop_duplicates(
        ["datetime_hour", "region_id"], keep="last"
    )
    hour_weather_cols = [c for c in combined.columns if c.startswith("hour_")]
    day_weather_cols = [
        c for c in combined.columns
        if c.startswith("day_") and c not in {"day_datetime", "day_sunrise", "day_sunset", "day_moonphase", "day_of_week"}
    ]
    lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]] = {}
    for row in combined.to_dict("records"):
        lookup[(pd.Timestamp(row["datetime_hour"]), int(row["region_id"]))] = row
    return lookup, hour_weather_cols, day_weather_cols


def build_isw_lookup(
    isw_daily: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, dict[str, float]], list[str], pd.Timestamp | None]:
    isw_cols = [c for c in isw_daily.columns if c.startswith("isw_topic_")]
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for row in isw_daily.to_dict("records"):
        lookup[pd.Timestamp(row["date"]).normalize()] = {col: float(row[col]) for col in isw_cols}
    last_date = max(lookup.keys()) if lookup else None
    return lookup, isw_cols, last_date


def build_tg_lookup(
    tg_hourly: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, dict[str, float]], list[str], pd.Timestamp | None]:
    tg_cols = [c for c in tg_hourly.columns if c.startswith("tg_topic_")]
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for row in tg_hourly.to_dict("records"):
        lookup[pd.Timestamp(row["datetime_hour"])] = {col: float(row[col]) for col in tg_cols}
    last_dt = max(lookup.keys()) if lookup else None
    return lookup, tg_cols, last_dt


TG_REGION_COLS = ["tg_region_threat_count", "tg_region_allclear_count", "tg_region_mention_count"]


def build_tg_region_lookup(
    tg_region: pd.DataFrame,
) -> tuple[dict[tuple[pd.Timestamp, int], dict[str, float]], pd.Timestamp | None]:
    lookup: dict[tuple[pd.Timestamp, int], dict[str, float]] = {}
    available_cols = [c for c in TG_REGION_COLS if c in tg_region.columns]
    last_dt: pd.Timestamp | None = None
    for row in tg_region.to_dict("records"):
        key = (pd.Timestamp(row["datetime_hour"]), int(row["region_id"]))
        lookup[key] = {col: float(row.get(col, 0.0)) for col in available_cols}
        dt = pd.Timestamp(row["datetime_hour"])
        if last_dt is None or dt > last_dt:
            last_dt = dt
    return lookup, last_dt



def validate_inputs(
    history: pd.DataFrame, forecast_weather: pd.DataFrame, model_features: list[str]
) -> None:
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


def build_hourly_totals(history: pd.DataFrame) -> dict[pd.Timestamp, float]:
    totals = history.groupby("datetime_hour")["alarm_active"].sum()
    return {pd.Timestamp(idx): float(val) for idx, val in totals.items()}



def trim_history_tail(history: pd.DataFrame, tail_hours: int) -> pd.DataFrame:

    max_dt = history["datetime_hour"].max()
    cutoff = max_dt - pd.Timedelta(hours=tail_hours)
    return history[history["datetime_hour"] > cutoff].copy()


def split_history_by_region_indexed(history: pd.DataFrame,) -> dict[int, dict[pd.Timestamp, int]]:
    out: dict[int, dict[pd.Timestamp, int]] = {}
    for region_id, part in history.groupby("region_id", sort=False):
        rid = int(region_id)
        out[rid] = {
            pd.Timestamp(row["datetime_hour"]): int(row["alarm_active"])
            for row in part[["datetime_hour", "alarm_active"]].to_dict("records")
        }
    return out


def split_history_for_derived(history: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for region_id, part in history.groupby("region_id", sort=False):
        out[int(region_id)] = part.sort_values("datetime_hour").reset_index(drop=True)
    return out


def get_alarm_value_fast(
    alarm_index: dict[pd.Timestamp, int], dt: pd.Timestamp
) -> int:
    return alarm_index.get(dt, 0)


def get_hours_since_last_alarm_fast(
    alarm_index: dict[pd.Timestamp, int], dt: pd.Timestamp
) -> float:
    prev_alarms = [ts for ts, val in alarm_index.items() if ts < dt and val == 1]
    if not prev_alarms:
        total = sum(1 for ts in alarm_index if ts < dt)
        return float(total)
    last_alarm_dt = max(prev_alarms)
    diff_hours = int((dt - last_alarm_dt) / pd.Timedelta(hours=1)) - 1
    return float(max(diff_hours, 0))


def get_alarms_last_24h_fast(
    alarm_index: dict[pd.Timestamp, int], dt: pd.Timestamp
) -> float:
    start = dt - pd.Timedelta(hours=24)
    return float(sum(val for ts, val in alarm_index.items() if start <= ts < dt))


def get_total_active_alarms_lag1(
    hourly_totals: dict[pd.Timestamp, float], dt: pd.Timestamp
) -> float:
    return float(hourly_totals.get(dt - pd.Timedelta(hours=1), 0.0))


def get_neighbour_alarms_fast(
    alarm_indexes: dict[int, dict[pd.Timestamp, int]],
    region_id: int,
    dt: pd.Timestamp,
) -> float:
    prev_dt = dt - pd.Timedelta(hours=1)
    neighbours = NEIGHBOURING_REGIONS.get(region_id, [])
    return float(sum(alarm_indexes.get(nb, {}).get(prev_dt, 0) for nb in neighbours))


def get_weather_values(
    weather_lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]],
    region_id: int,
    target_dt: pd.Timestamp,
    hour_weather_cols: list[str],
    day_weather_cols: list[str],
    fallback_vals: dict[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    prev_hour_row = weather_lookup.get((target_dt - pd.Timedelta(hours=1), region_id))
    prev_day_row = weather_lookup.get((target_dt - pd.Timedelta(hours=24), region_id))

    for col in hour_weather_cols:
        if prev_hour_row is not None and col in prev_hour_row:
            values[col] = prev_hour_row[col]
        elif fallback_vals is not None and col in fallback_vals:
            values[col] = fallback_vals[col]
        else:
            values[col] = 0

    for col in day_weather_cols:
        if prev_day_row is not None and col in prev_day_row:
            values[col] = prev_day_row[col]
        elif fallback_vals is not None and col in fallback_vals:
            values[col] = fallback_vals[col]
        else:
            values[col] = 0

    return values


def tg_decay_factor(missing_hours: int) -> float:
    if missing_hours <= 0:
        return 1.0
    return float(0.5 ** (missing_hours / TG_DECAY_HALFLIFE_HOURS))


def get_shifted_isw_values(
    isw_lookup: dict[pd.Timestamp, dict[str, float]],
    isw_cols: list[str],
    last_isw_date: pd.Timestamp | None,
    target_dt: pd.Timestamp,
    fallback_vals: dict[str, Any] | None,
) -> dict[str, float]:
    src_date = (target_dt - pd.Timedelta(hours=24)).normalize()
    raw = isw_lookup.get(src_date)
    if raw is not None:
        return {col: float(raw.get(col, 0.0)) for col in isw_cols}
    if last_isw_date is not None and last_isw_date in isw_lookup:
        raw = isw_lookup[last_isw_date]
        return {col: float(raw.get(col, 0.0)) for col in isw_cols}
    if fallback_vals is not None:
        return {col: float(fallback_vals.get(col, 0.0)) for col in isw_cols}
    return {col: 0.0 for col in isw_cols}


def get_shifted_tg_values(
    tg_lookup: dict[pd.Timestamp, dict[str, float]],
    tg_cols: list[str],
    last_real_tg_dt: pd.Timestamp | None,
    target_dt: pd.Timestamp,
    fallback_vals: dict[str, Any] | None,
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
    if fallback_vals is not None:
        return {col: float(fallback_vals.get(col, 0.0)) for col in tg_cols}
    return {col: 0.0 for col in tg_cols}


def get_shifted_tg_region_values(
    tg_region_lookup: dict[tuple[pd.Timestamp, int], dict[str, float]],
    last_tg_region_dt: pd.Timestamp | None,
    target_dt: pd.Timestamp,
    region_id: int,
) -> dict[str, float]:
    src_dt = target_dt - pd.Timedelta(hours=1)
    key = (src_dt, region_id)
    if key in tg_region_lookup:
        return dict(tg_region_lookup[key])
    if last_tg_region_dt is not None:
        fallback_key = (last_tg_region_dt, region_id)
        if fallback_key in tg_region_lookup:
            missing_hours = int((src_dt - last_tg_region_dt) / pd.Timedelta(hours=1))
            decay = tg_decay_factor(missing_hours)
            base = tg_region_lookup[fallback_key]
            return {col: float(base.get(col, 0.0)) * decay for col in TG_REGION_COLS}
    return {col: 0.0 for col in TG_REGION_COLS}


def add_topic_derived_features(
    row: dict[str, Any],
    region_hist_df: pd.DataFrame,
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

        if "isw_total_intensity" in region_hist_df.columns:
            intensity_series = pd.concat([
                region_hist_df["isw_total_intensity"].astype(float),
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

        if "tg_total_intensity" in region_hist_df.columns:
            intensity_series = pd.concat([
                region_hist_df["tg_total_intensity"].astype(float),
                pd.Series([row["tg_total_intensity"]]),
            ], ignore_index=True)
            row["tg_velocity_3h"] = float(intensity_series.diff(3).iloc[-1]) if len(intensity_series) > 3 else 0.0
            row["tg_intensity_ema_6h"] = ewm_last(intensity_series, span=6)
            row["tg_intensity_zscore"] = zscore_last(intensity_series, window=24)



def build_future_hour_rows(
    target_dt: pd.Timestamp,
    alarm_indexes: dict[int, dict[pd.Timestamp, int]],
    hourly_totals: dict[pd.Timestamp, float],
    region_meta_records: list[dict[str, Any]],
    history_for_derived: dict[int, pd.DataFrame],
    fallback_vals_by_region: dict[int, dict[str, Any]],
    weather_lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]],
    hour_weather_cols: list[str],
    day_weather_cols: list[str],
    isw_lookup: dict[pd.Timestamp, dict[str, float]],
    isw_cols: list[str],
    last_isw_date: pd.Timestamp | None,
    tg_lookup: dict[pd.Timestamp, dict[str, float]],
    tg_cols: list[str],
    last_real_tg_dt: pd.Timestamp | None,
    tg_region_lookup: dict[tuple[pd.Timestamp, int], dict[str, float]],
    last_tg_region_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for meta in region_meta_records:
        region_id = int(meta["region_id"])
        alarm_idx = alarm_indexes[region_id]
        fallback = fallback_vals_by_region.get(region_id)
        region_hist_df = history_for_derived[region_id]

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
            "alarm_lag_1": float(get_alarm_value_fast(alarm_idx, target_dt - pd.Timedelta(hours=1))),
            "alarm_lag_3": float(get_alarm_value_fast(alarm_idx, target_dt - pd.Timedelta(hours=3))),
            "alarm_lag_6": float(get_alarm_value_fast(alarm_idx, target_dt - pd.Timedelta(hours=6))),
            "alarm_lag_12": float(get_alarm_value_fast(alarm_idx, target_dt - pd.Timedelta(hours=12))),
            "alarms_in_last_24h": float(get_alarms_last_24h_fast(alarm_idx, target_dt)),
            "total_active_alarms_lag1": float(get_total_active_alarms_lag1(hourly_totals, target_dt)),
            "neighbour_alarms": float(get_neighbour_alarms_fast(alarm_indexes, region_id, target_dt)),
            "hours_since_last_alarm": float(get_hours_since_last_alarm_fast(alarm_idx, target_dt)),
        }

        row.update(get_weather_values(weather_lookup, region_id, target_dt, hour_weather_cols, day_weather_cols, fallback))
        row.update(get_shifted_isw_values(isw_lookup, isw_cols, last_isw_date, target_dt, fallback))
        row.update(get_shifted_tg_values(tg_lookup, tg_cols, last_real_tg_dt, target_dt, fallback))
        row.update(get_shifted_tg_region_values(tg_region_lookup, last_tg_region_dt, target_dt, region_id))
        add_topic_derived_features(row, region_hist_df, isw_cols, tg_cols)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)



def copy_alarm_indexes(
    alarm_indexes: dict[int, dict[pd.Timestamp, int]]
) -> dict[int, dict[pd.Timestamp, int]]:

    return {rid: dict(idx) for rid, idx in alarm_indexes.items()}


def append_stochastic_state(
    target_dt: pd.Timestamp,
    region_ids: list[int],
    proba: np.ndarray,
    alarm_indexes: dict[int, dict[pd.Timestamp, int]],
    hourly_totals: dict[pd.Timestamp, float],
    rng: np.random.Generator,
) -> None:

    sampled = (rng.random(len(proba)) < proba).astype(int)
    for i, region_id in enumerate(region_ids):
        alarm_indexes[region_id][target_dt] = int(sampled[i])
    hourly_totals[target_dt] = float(sampled.sum())


def run_single_mc_sample(
    future_hours: list[pd.Timestamp],
    alarm_indexes_init: dict[int, dict[pd.Timestamp, int]],
    hourly_totals_init: dict[pd.Timestamp, float],
    region_meta_records: list[dict[str, Any]],
    history_for_derived: dict[int, pd.DataFrame],
    fallback_vals_by_region: dict[int, dict[str, Any]],
    model: Any,
    feature_names: list[str],
    weather_lookup: dict,
    hour_weather_cols: list[str],
    day_weather_cols: list[str],
    isw_lookup: dict,
    isw_cols: list[str],
    last_isw_date: pd.Timestamp | None,
    tg_lookup: dict,
    tg_cols: list[str],
    last_real_tg_dt: pd.Timestamp | None,
    tg_region_lookup: dict,
    last_tg_region_dt: pd.Timestamp | None,
    rng: np.random.Generator,
) -> np.ndarray:

    alarm_idx = copy_alarm_indexes(alarm_indexes_init)
    ht = dict(hourly_totals_init)
    region_ids = [int(m["region_id"]) for m in region_meta_records]

    run_probas: list[np.ndarray] = []

    for target_dt in future_hours:
        hour_rows = build_future_hour_rows(
            target_dt=target_dt,
            alarm_indexes=alarm_idx,
            hourly_totals=ht,
            region_meta_records=region_meta_records,
            history_for_derived=history_for_derived,
            fallback_vals_by_region=fallback_vals_by_region,
            weather_lookup=weather_lookup,
            hour_weather_cols=hour_weather_cols,
            day_weather_cols=day_weather_cols,
            isw_lookup=isw_lookup,
            isw_cols=isw_cols,
            last_isw_date=last_isw_date,
            tg_lookup=tg_lookup,
            tg_cols=tg_cols,
            last_real_tg_dt=last_real_tg_dt,
            tg_region_lookup=tg_region_lookup,
            last_tg_region_dt=last_tg_region_dt,
        )

        X_hour = ensure_numeric_bool_matrix(hour_rows, feature_names)
        proba = model.predict_proba(X_hour)[:, 1]
        run_probas.append(proba.copy())

        append_stochastic_state(target_dt, region_ids, proba, alarm_idx, ht, rng)

    return np.concatenate(run_probas)



def model_train_time_from_file(model_path: Path) -> str:
    if not model_path.exists():
        return ""
    return pd.Timestamp(model_path.stat().st_mtime, unit="s", tz=KYIV_TZ).isoformat()


def save_predictions(
    predictions: pd.DataFrame, paths: ProjectPaths, threshold: float, n_samples: int
) -> Path:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now(tz=KYIV_TZ).strftime("%d_%m_%Y_%H-%M")
    out_path = paths.output_dir / f"next_24h_predictions_{ts}.json"
    latest_path = paths.output_dir / "next_24h_predictions_latest.json"

    compact = predictions[[
        "datetime_hour", "region_id", "region_key", "city_name",
        "prediction_proba", "prediction_proba_std", "prediction",
    ]].copy()
    compact["datetime_hour"] = pd.to_datetime(compact["datetime_hour"], errors="coerce")

    regions_forecast: dict[str, dict[str, Any]] = {}
    for region_key, part in compact.groupby("region_key", sort=False):
        part = part.sort_values("datetime_hour").reset_index(drop=True)
        first = part.iloc[0]
        regions_forecast[str(first["region_id"])] = {
            "region_id": int(first["region_id"]),
            "city_name": str(first["city_name"]),
            "forecast": {
                dt.strftime("%Y-%m-%d %H:%M"): bool(pred)
                for dt, pred in zip(part["datetime_hour"], part["prediction"])
            },
            "forecast_proba": {
                dt.strftime("%Y-%m-%d %H:%M"): round(float(proba), 4)
                for dt, proba in zip(part["datetime_hour"], part["prediction_proba"])
            },
            "forecast_uncertainty": {
                dt.strftime("%Y-%m-%d %H:%M"): round(float(std), 4)
                for dt, std in zip(part["datetime_hour"], part["prediction_proba_std"])
            },
        }

    payload = {
        "last_model_train_time": model_train_time_from_file(paths.model_path),
        "last_prediction_time": pd.Timestamp.now(tz=KYIV_TZ).isoformat(),
        "threshold": threshold,
        "mc_samples": n_samples,
        "hours": int(compact["datetime_hour"].nunique()),
        "regions": int(compact["region_id"].nunique()),
        "rows": int(len(compact)),
        "regions_forecast": regions_forecast,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path



def main() -> None:
    paths = ProjectPaths.default()

    log("Loading model artifact")
    model_artifact = load_model_artifact(paths.model_path)
    model = model_artifact["model"]
    threshold = float(model_artifact["threshold"])
    feature_names = list(model_artifact["feature_names"])

    log("Loading historical context")
    history = load_history(paths)
    forecast_weather = load_forecast_weather(paths)
    actual_weather = load_processed_weather(paths)
    isw_daily = load_isw_daily(paths)
    tg_hourly = load_tg_hourly(paths)
    tg_region = load_tg_region(paths)

    validate_inputs(history, forecast_weather, feature_names)

    region_meta = get_region_meta(history)
    region_meta_records = region_meta.sort_values("region_id").to_dict("records")

    history_tail = trim_history_tail(history, HISTORY_TAIL_HOURS)
    log(f"History tail: {len(history_tail)} rows (last {HISTORY_TAIL_HOURS}h per all regions)")

    alarm_indexes_init = split_history_by_region_indexed(history_tail)
    history_for_derived = split_history_for_derived(history_tail)
    hourly_totals_init = build_hourly_totals(history_tail)

    fallback_vals_by_region: dict[int, dict[str, Any]] = {
        rid: df.iloc[-1].to_dict()
        for rid, df in history_for_derived.items()
        if not df.empty
    }

    weather_lookup, hour_weather_cols, day_weather_cols = build_weather_lookup(actual_weather, forecast_weather)
    isw_lookup, isw_cols, last_isw_date = build_isw_lookup(isw_daily)
    tg_lookup, tg_cols, last_real_tg_dt = build_tg_lookup(tg_hourly)
    tg_region_lookup, last_tg_region_dt = build_tg_region_lookup(tg_region)

    future_hours = sorted(forecast_weather["datetime_hour"].drop_duplicates())
    future_hours = [pd.Timestamp(dt) for dt in future_hours]

    n_regions = len(region_meta_records)
    n_hours = len(future_hours)
    total_rows = n_hours * n_regions

    log(f"Starting Monte Carlo forecast: {MC_N_SAMPLES} samples × {n_hours}h × {n_regions} regions")

    master_rng = np.random.default_rng(MC_RANDOM_SEED)
    all_run_probas = np.zeros((MC_N_SAMPLES, total_rows), dtype=np.float64)

    shared_kwargs: dict[str, Any] = dict(
        future_hours=future_hours,
        alarm_indexes_init=alarm_indexes_init,
        hourly_totals_init=hourly_totals_init,
        region_meta_records=region_meta_records,
        history_for_derived=history_for_derived,
        fallback_vals_by_region=fallback_vals_by_region,
        model=model,
        feature_names=feature_names,
        weather_lookup=weather_lookup,
        hour_weather_cols=hour_weather_cols,
        day_weather_cols=day_weather_cols,
        isw_lookup=isw_lookup,
        isw_cols=isw_cols,
        last_isw_date=last_isw_date,
        tg_lookup=tg_lookup,
        tg_cols=tg_cols,
        last_real_tg_dt=last_real_tg_dt,
        tg_region_lookup=tg_region_lookup,
        last_tg_region_dt=last_tg_region_dt,
    )

    for run_idx in range(MC_N_SAMPLES):
        run_rng = np.random.default_rng(master_rng.integers(0, 2**32))
        all_run_probas[run_idx] = run_single_mc_sample(**shared_kwargs, rng=run_rng)

        if (run_idx + 1) % 5 == 0 or run_idx == MC_N_SAMPLES - 1:
            log(f"  Completed sample {run_idx + 1}/{MC_N_SAMPLES}")

    mean_proba = all_run_probas.mean(axis=0)
    std_proba = all_run_probas.std(axis=0)
    final_pred = (mean_proba >= threshold).astype(int)

    index_rows: list[dict[str, Any]] = []
    for target_dt in future_hours:
        for meta in region_meta_records:
            index_rows.append({
                "datetime_hour": target_dt,
                "region_id": int(meta["region_id"]),
                "region_key": meta.get("region_key"),
                "city_name": meta.get("city_name"),
            })

    predictions = pd.DataFrame(index_rows)
    predictions["prediction_proba"] = mean_proba
    predictions["prediction_proba_std"] = std_proba
    predictions["prediction"] = final_pred
    predictions = predictions.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)

    out_path = save_predictions(predictions, paths, threshold, n_samples=MC_N_SAMPLES)

    log(f"Monte Carlo forecast complete ({MC_N_SAMPLES} samples)")
    log(f"Saved: {out_path}")
    log(f"Rows={len(predictions)} | hours={predictions['datetime_hour'].nunique()} | regions={predictions['region_id'].nunique()}")

    alarm_regions = predictions[predictions["prediction"] == 1]["region_key"].nunique()
    mean_std = std_proba.mean()
    log(f"Alarm regions: {alarm_regions}/{n_regions} | Mean uncertainty (std): {mean_std:.4f}")


if __name__ == "__main__":
    main()
