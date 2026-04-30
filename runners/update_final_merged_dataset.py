import json
import os
import pickle
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

KYIV_TZ = "Europe/Kyiv"
EXPECTED_REGION_COUNT = 24
HISTORY_TAIL_HOURS_FOR_UNSHIFTED = 24 * 30
HISTORY_TAIL_HOURS_FOR_FINAL = 72

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

WEATHER_DROP_COLUMNS = [
    "city_latitude",
    "city_longitude",
    "day_snowdepth",
    "day_windspeed",
    "day_winddir",
    "day_pressure",
    "day_solarradiation",
    "day_solarenergy",
    "day_uvindex",
    "day_conditions_simple_Clear",
    "day_conditions_simple_Cloudy",
    "day_conditions_simple_Rain",
    "day_conditions_simple_Snow",
    "year",
    "month",
    "city_timezone",
    "hour_datetime",
    "hour_precipprob",
    "day_precipprob",
]

ISW_TEXT_TOPLEVEL_COLUMNS = ["date", "title", "url", "text"]
TELEGRAM_TEXT_TOPLEVEL_COLUMNS = ["date", "channel", "message"]

TG_REGION_COLS = [
    "tg_region_threat_count",
    "tg_region_allclear_count",
    "tg_region_mention_count",
]


TG_REGION_PATTERNS: dict[int, str] = {
    2: r"вінниц",
    3: r"волин|луцьк",
    4: r"дніпр|дніпропетровськ",
    5: r"донецьк|донеччин",
    6: r"житомир",
    7: r"закарпат|ужгород",
    8: r"запоріж",
    9: r"івано.франків",
    10: r"київськ|київщин",
    11: r"кіровоград|кропивницьк",
    13: r"львів",
    14: r"миколаїв",
    15: r"одес",
    16: r"полтав",
    17: r"рівн",
    18: r"сум",
    19: r"тернопіл",
    20: r"харків|харківщин",
    21: r"херсон",
    22: r"хмельниц",
    23: r"черкас",
    24: r"чернівц",
    25: r"чернігів",
    26: r"київ(?!ськ)",
}

TG_THREAT_PATTERN = r"загроза|виліт|пуск|ракет|шахед|герань|бпла|дрон|балістич|крилат"
TG_ALLCLEAR_PATTERN = r"відбій"

UNSHIFTED_DERIVED_COLS = {
    "alarm_lag_1",
    "alarm_lag_3",
    "alarm_lag_6",
    "alarm_lag_12",
    "alarms_in_last_24h",
    "is_weekend",
    "is_night",
    "total_active_alarms_lag1",
    "neighbour_alarms",
    "hours_since_last_alarm",
    "isw_total_intensity",
    "isw_topic_std",
    "isw_topic_max",
    "isw_topic_mean",
    "isw_topic_entropy",
    "isw_velocity_24h",
    "isw_intensity_ema",
    "tg_total_intensity",
    "tg_topic_std",
    "tg_topic_max",
    "tg_topic_entropy",
    "tg_velocity_3h",
    "tg_intensity_ema_6h",
    "tg_intensity_zscore",
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


def warn(msg: str) -> None:
    print(f"[{now_str()}] WARNING: {msg}")


@dataclass
class ProjectPaths:
    project_root: Path
    data_dir: Path
    snapshots: Path
    artifacts_dir: Path
    runtime_dir: Path
    unshifted_parquet: Path
    final_parquet: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            snapshots=data_dir / "raw_snapshots",
            artifacts_dir=data_dir / "nlp_artifacts",
            runtime_dir=data_dir / "runtime",
            unshifted_parquet=data_dir / "merged_sources_unshifted.parquet",
            final_parquet=data_dir / "final_merged_dataset.parquet",
        )

    def validate(self) -> None:
        if not self.snapshots.exists() or not self.snapshots.is_dir():
            raise FileNotFoundError(
                f"Missing snapshots folder: {self.snapshots}. Expected: data/raw_snapshots/"
            )
        if not self.artifacts_dir.exists() or not self.artifacts_dir.is_dir():
            raise FileNotFoundError(
                f"Missing NLP artifacts folder: {self.artifacts_dir}. Expected: data/nlp_artifacts/"
            )
        if not self.unshifted_parquet.exists():
            raise FileNotFoundError(
                f"Missing {self.unshifted_parquet}. Run the historical merge notebook first so it saves "
                "data/merged_sources_unshifted.parquet before the final shifts."
            )
        if not self.final_parquet.exists():
            raise FileNotFoundError(
                f"Missing {self.final_parquet}. Run the historical merge notebook first."
            )


@dataclass
class SnapshotStore:
    source: Path

    def __post_init__(self) -> None:
        self.source = Path(self.source)
        if not self.source.exists() or not self.source.is_dir():
            raise FileNotFoundError(
                f"Snapshot folder not found: {self.source}. Expected: data/raw_snapshots/"
            )

    def list_json_files(self, relative_source_dir: str) -> list[Path]:
        base = self.source / relative_source_dir
        if not base.exists():
            return []
        return sorted(base.rglob("*.json"))

    @staticmethod
    def read_json(path_ref: str | Path) -> Any:
        with open(path_ref, "r", encoding="utf-8") as f:
            return json.load(f)


def infer_completed_hour(tz: str = KYIV_TZ) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz).tz_localize(None).floor("h") - pd.Timedelta(hours=1)


def remove_spring_dst_hour(df: pd.DataFrame, dt_col: str = "datetime_hour") -> pd.DataFrame:
    if df.empty or dt_col not in df.columns:
        return df
    years = sorted(pd.to_datetime(df[dt_col], errors="coerce").dt.year.dropna().unique())
    rows_to_remove: list[pd.Timestamp] = []
    for year in years:
        last_day = pd.Timestamp(year=int(year), month=3, day=31)
        while last_day.weekday() != 6:
            last_day -= pd.Timedelta(days=1)
        rows_to_remove.append(last_day.replace(hour=3, minute=0, second=0))
    mask = pd.to_datetime(df[dt_col], errors="coerce").isin(rows_to_remove)
    removed = int(mask.sum())
    if removed:
        warn(f"Removing {removed} rows matching spring-DST cleanup rule.")
    return df.loc[~mask].copy()


def ensure_date_string_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def coerce_bool_columns(df: pd.DataFrame, bool_cols: Iterable[str]) -> pd.DataFrame:
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def safe_entropy_from_abs(df_values: pd.DataFrame) -> pd.Series:
    row_sums = df_values.sum(axis=1)
    probs = df_values.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=1)
    return entropy.fillna(0)


def hours_since_last_alarm_vectorized(series: pd.Series) -> pd.Series:
    shifted = series.shift(1).fillna(0)
    alarm_cumsum = shifted.cumsum()
    result = shifted.groupby(alarm_cumsum).cumcount()
    return result.astype(float)


def maybe_file_relevant_by_name(path_ref: Path, start_hour: pd.Timestamp, end_hour: pd.Timestamp) -> bool:
    text = str(path_ref)
    matches = re.findall(r"(20\d{2})[-_](\d{2})[-_](\d{2})", text)
    if not matches:
        return True
    dates = [pd.Timestamp(year=int(y), month=int(m), day=int(d)) for y, m, d in matches]
    start_day = start_hour.floor("D") - pd.Timedelta(days=1)
    end_day = end_hour.floor("D") + pd.Timedelta(days=1)
    return any(start_day <= d <= end_day for d in dates)




def _require_pyarrow() -> None:
    try:
        import pyarrow
    except ImportError as exc:
        raise RuntimeError("This runner needs pyarrow to read/write parquet files.") from exc


def parquet_schema(path: Path) -> dict[str, Any]:
    _require_pyarrow()
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    path = Path(path)
    if path.is_dir():
        schema = ds.dataset(str(path), format="parquet").schema
    else:
        schema = pq.read_schema(path)
    return {
        "columns": list(schema.names),
        "dtypes": {name: str(schema.field(name).type) for name in schema.names},
    }


def read_parquet_any(
    path: str | os.PathLike[str],
    columns: list[str] | None = None,
    filters: Any | None = None,
) -> pd.DataFrame:
    try:
        kwargs: dict[str, Any] = {}
        if columns is not None:
            kwargs["columns"] = columns
        if filters is not None:
            kwargs["filters"] = filters
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:
        raise RuntimeError("Could not read parquet. Install pyarrow in the project environment.") from exc


def write_single_parquet_file_atomic(df: pd.DataFrame, path: Path) -> None:
    _require_pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".__tmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.parquet")
    df.to_parquet(tmp_path, index=False, engine="pyarrow")

    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    tmp_path.replace(path)


def ensure_single_parquet_file(path: Path, label: str) -> None:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return
    log(f"Converting existing {label} directory back to a single parquet file")
    df = read_parquet_any(path)
    write_single_parquet_file_atomic(df, path)
    log(f"{label} is now a single parquet file again: {path}")


def parquet_max_timestamp(path: Path, column: str) -> pd.Timestamp | None:
    if not Path(path).exists():
        return None
    df = read_parquet_any(path, columns=[column])
    if df.empty:
        return None
    dt = pd.to_datetime(df[column], errors="coerce")
    mx = dt.max()
    return None if pd.isna(mx) else pd.Timestamp(mx).tz_localize(None) if getattr(mx, "tzinfo", None) else pd.Timestamp(mx)


def read_parquet_tail_by_hours(path: Path, max_hour: pd.Timestamp, tail_hours: int) -> pd.DataFrame:
    start = max_hour - pd.Timedelta(hours=tail_hours)
    try:
        df = read_parquet_any(path, filters=[("datetime_hour", ">=", start)])
    except Exception:
        df = read_parquet_any(path)
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
        df = df[df["datetime_hour"] >= start].copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    return df.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)


def read_parquet_range_by_hours(
    path: Path,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    start_hour = pd.Timestamp(start_hour)
    end_hour = pd.Timestamp(end_hour)
    try:
        df = read_parquet_any(
            path,
            columns=columns,
            filters=[
                ("datetime_hour", ">=", start_hour),
                ("datetime_hour", "<=", end_hour),
            ],
        )
    except Exception:
        df = read_parquet_any(path, columns=columns)
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
        df = df[(df["datetime_hour"] >= start_hour) & (df["datetime_hour"] <= end_hour)].copy()

    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    if "region_id" in df.columns:
        df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
        df = df[df["region_id"].notna()].copy()
        df["region_id"] = df["region_id"].astype(int)
    sort_cols = [c for c in ["datetime_hour", "region_id"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)


def find_earliest_missing_final_hour(
    unshifted_path: Path,
    final_path: Path,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
) -> pd.Timestamp | None:

    key_cols = ["datetime_hour", "region_id"]
    expected = read_parquet_range_by_hours(unshifted_path, start_hour, end_hour, columns=key_cols)
    if expected.empty:
        return None
    expected = expected.drop_duplicates(key_cols)

    if not Path(final_path).exists():
        return pd.Timestamp(expected["datetime_hour"].min())

    existing = read_parquet_range_by_hours(final_path, start_hour, end_hour, columns=key_cols)
    if existing.empty:
        return pd.Timestamp(expected["datetime_hour"].min())
    existing = existing.drop_duplicates(key_cols)

    merged = expected.merge(existing, on=key_cols, how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    if missing.empty:
        return None
    return pd.Timestamp(missing["datetime_hour"].min())


def log_recent_dataset_health(path: Path, label: str, lookback_days: int = 60) -> None:
    max_hour = parquet_max_timestamp(path, "datetime_hour")
    if max_hour is None:
        warn(f"{label}: could not read max datetime_hour")
        return
    start = max_hour - pd.Timedelta(days=lookback_days)
    keys = read_parquet_range_by_hours(path, start, max_hour, columns=["datetime_hour", "region_id"])
    if keys.empty:
        warn(f"{label}: no rows in the last {lookback_days} days")
        return
    dups = int(keys.duplicated(["datetime_hour", "region_id"]).sum())
    counts = keys.groupby("datetime_hour")["region_id"].nunique()
    bad_hours = int((counts != 24).sum())
    log(
        f"{label} health: max={max_hour}, recent_rows={len(keys):,}, "
        f"recent_hours={counts.size:,}, duplicate_keys={dups}, hours_with_not_24_regions={bad_hours}"
    )


def rewrite_parquet_file_with_new_rows(
    path: Path,
    new_rows: pd.DataFrame,
    label: str,
    key_cols: list[str] | None = None,
) -> None:
    if new_rows.empty:
        log(f"No new rows for {label}; file was not rewritten")
        return

    path = Path(path)
    if path.exists():
        existing = read_parquet_any(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()

    if "datetime_hour" in combined.columns:
        combined["datetime_hour"] = pd.to_datetime(combined["datetime_hour"], errors="coerce")

    if key_cols:
        missing_keys = [c for c in key_cols if c not in combined.columns]
        if missing_keys:
            raise ValueError(f"Cannot deduplicate {label}; missing key columns: {missing_keys}")
        combined = (
            combined.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = combined.reset_index(drop=True)

    sort_cols = [c for c in ["datetime_hour", "region_id"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    write_single_parquet_file_atomic(combined, path)
    log(f"Rewrote {label} as a single parquet file: +{len(new_rows):,} rows, total {len(combined):,} rows")


def cast_series_to_dtype(series: pd.Series, dtype_str: str) -> pd.Series:
    dtype_str = str(dtype_str).lower()
    try:
        if "timestamp" in dtype_str or "datetime" in dtype_str:
            return pd.to_datetime(series, errors="coerce")
        if dtype_str.startswith("date32") or dtype_str.startswith("date64") or dtype_str == "date":
            return pd.to_datetime(series, errors="coerce").dt.date
        if dtype_str in {"bool", "boolean"}:
            return series.fillna(False).astype(bool)
        if dtype_str.startswith("int") or dtype_str in {"int8", "int16", "int32", "int64"}:
            return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
        if dtype_str.startswith("uint"):
            return pd.to_numeric(series, errors="coerce").fillna(0).astype("uint64")
        if dtype_str.startswith("float") or dtype_str in {"double"}:
            return pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")
        if dtype_str in {"string", "large_string"}:
            return series.astype("string")
        return series
    except Exception:
        return series


def default_value_for_column(col: str) -> Any:
    if col.startswith("hour_conditions_simple_"):
        return False
    if col in {"alarm_active", "region_id", "day_of_week", "hour", "is_weekend", "is_night"}:
        return 0
    if re.fullmatch(r"(isw|tg)_topic_\d+", col):
        return 0.0
    if col in TG_REGION_COLS:
        return 0
    if col.endswith("_count"):
        return 0
    return 0


def align_to_schema(
    df: pd.DataFrame,
    schema: dict[str, Any],
    *,
    strict_extra: bool = False,
    strict_missing: bool = False,
) -> pd.DataFrame:
    expected_cols = schema["columns"]
    expected_dtypes = schema["dtypes"]
    out = df.copy()
    missing = [c for c in expected_cols if c not in out.columns]
    extra = [c for c in out.columns if c not in expected_cols]

    if strict_missing and missing:
        raise ValueError(f"Schema drift: missing columns {missing}")
    if strict_extra and extra:
        raise ValueError(f"Schema drift: unexpected columns {extra}")

    for col in missing:
        out[col] = default_value_for_column(col)
    if extra:
        out = out.drop(columns=extra)

    out = out[expected_cols].copy()
    for col in expected_cols:
        out[col] = cast_series_to_dtype(out[col], expected_dtypes.get(col, "object"))
    return out



def ensure_nltk_resources() -> None:
    import nltk

    needed = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, download_name in needed:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log(f"Downloading NLTK resource: {download_name}")
            nltk.download(download_name, quiet=True)


class ISWTextPreprocessor:
    def __init__(self) -> None:
        ensure_nltk_resources()
        from nltk import pos_tag
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer

        self.pos_tag = pos_tag
        self.stop_words = set(stopwords.words("english"))
        self.wordnet = wordnet
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag: str) -> str:
        if treebank_tag.startswith("J"):
            return self.wordnet.ADJ
        if treebank_tag.startswith("V"):
            return self.wordnet.VERB
        if treebank_tag.startswith("N"):
            return self.wordnet.NOUN
        if treebank_tag.startswith("R"):
            return self.wordnet.ADV
        return self.wordnet.NOUN

    @staticmethod
    def clean_isw_text(text: str) -> str:
        text = str(text)
        text = text.replace("Previous\nNext", " ")
        text = text.replace("Click\nhere", " ")
        text = text.replace("\n", " ")
        return text

    def smart_preprocess_cached(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\b(?:https|http|www)\S*\b", " ", text)
        text = re.sub(r"\b(?:twitter|facebook|telegram|youtube|instagram|isw)\S*\b", " ", text)
        text = re.sub(r"\b(previous|next|click here|dot)\b", " ", text)
        text = re.sub(r"[^a-z\s-]", " ", text)
        text = re.sub(r"\b\d+\b", "", text)
        words = [w for w in text.split() if w not in self.stop_words and len(w) > 2]
        tagged = self.pos_tag(words)
        lemmas = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in tagged]
        return " ".join(lemmas)


def lazy_import_tg_clean_deps():
    try:
        import pymorphy3
        from stop_words import get_stop_words
    except ImportError as exc:
        raise RuntimeError("Telegram preprocessing needs pymorphy3 and stop-words. Install them first.") from exc
    return pymorphy3, get_stop_words


class TelegramTextPreprocessor:
    def __init__(self) -> None:
        pymorphy3, get_stop_words = lazy_import_tg_clean_deps()
        self.morph = pymorphy3.MorphAnalyzer(lang="uk")
        self.ua_stop_words = set(get_stop_words("ukrainian"))
        self.trash_words = {"telegram", "whatsapp", "viber"}
        self.word_cache: dict[str, str] = {}

    def fast_tg_clean_optimized(self, text: str) -> str:
        text = re.sub(r"https?://\S+|@\w+", "", str(text).lower())
        text = re.sub(r"[^а-яіїєґa-z\s-]", " ", text)
        words = text.split()
        res: list[str] = []
        for w in words:
            if len(w) < 3:
                continue
            if w not in self.word_cache:
                self.word_cache[w] = self.morph.parse(w)[0].normal_form
            lemma = self.word_cache[w]
            if lemma not in self.ua_stop_words and lemma not in self.trash_words:
                res.append(lemma)
        return " ".join(res)



def prepare_weather_frame(df: pd.DataFrame, start_hour: pd.Timestamp, end_hour: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    df = df[(df["datetime_hour"] >= start_hour) & (df["datetime_hour"] <= end_hour)].copy()
    if df.empty:
        return df
    df = remove_spring_dst_hour(df, dt_col="datetime_hour")

    for col in [
        "day_solarradiation",
        "day_solarenergy",
        "day_uvindex",
        "hour_solarradiation",
        "hour_solarenergy",
        "hour_uvindex",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            med = df[col].median()
            if pd.notna(med):
                df[col] = df[col].fillna(med)
    if "hour_precip" in df.columns:
        df["hour_precip"] = pd.to_numeric(df["hour_precip"], errors="coerce").fillna(0)

    drop_now = [c for c in WEATHER_DROP_COLUMNS if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    bool_cols = [c for c in df.columns if c.startswith("hour_conditions_simple_")]
    df = coerce_bool_columns(df, bool_cols)

    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)

    if "day_datetime" not in df.columns:
        df["day_datetime"] = df["datetime_hour"].dt.strftime("%Y-%m-%d")
    else:
        df = ensure_date_string_col(df, "day_datetime")
    if "hour" not in df.columns:
        df["hour"] = df["datetime_hour"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["datetime_hour"].dt.dayofweek

    df = (
        df.sort_values(["datetime_hour", "region_id"])
        .drop_duplicates(["datetime_hour", "region_id"], keep="last")
        .reset_index(drop=True)
    )
    return df


def preprocess_weather_new_rows(store: SnapshotStore, start_hour: pd.Timestamp, end_hour: pd.Timestamp) -> pd.DataFrame:
    files = store.list_json_files("weather_historical")
    if not files:
        raise FileNotFoundError("No weather_historical snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        if not maybe_file_relevant_by_name(path_ref, start_hour, end_hour):
            continue
        payload = store.read_json(path_ref)
        for item in payload.get("regions", []):
            weather = dict(item.get("weather", {}))
            dt = pd.to_datetime(weather.get("datetime_hour"), errors="coerce")
            if pd.isna(dt) or dt < start_hour or dt > end_hour:
                continue
            weather.setdefault("region_id", item.get("region_id"))
            weather.setdefault("region_key", item.get("region"))
            rows.append(weather)

    return prepare_weather_frame(pd.DataFrame(rows), start_hour, end_hour)


def preprocess_alarms_new_rows(
    store: SnapshotStore,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    region_dim: pd.DataFrame,
) -> pd.DataFrame:
    files = store.list_json_files("alarms_hourly")
    if not files:
        raise FileNotFoundError("No alarms_hourly snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        if not maybe_file_relevant_by_name(path_ref, start_hour, end_hour):
            continue
        payload = store.read_json(path_ref)
        if isinstance(payload, list):
            for row in payload:
                dt = pd.to_datetime(row.get("datetime_hour"), errors="coerce")
                if pd.isna(dt) or dt < start_hour or dt > end_hour:
                    continue
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
        df = df[df["datetime_hour"].notna()].copy()
        df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
        df = df[df["region_id"].notna()].copy()
        df["region_id"] = df["region_id"].astype(int)
        df["alarm_active"] = pd.to_numeric(df.get("alarm_active", 0), errors="coerce").fillna(0).astype(int)
        if "alarm_minutes_in_hour" in df.columns:
            df["alarm_minutes_in_hour"] = pd.to_numeric(df["alarm_minutes_in_hour"], errors="coerce").fillna(0.0)
        else:
            df["alarm_minutes_in_hour"] = 0.0
        df = (
            df.sort_values(["datetime_hour", "region_id"])
            .drop_duplicates(["datetime_hour", "region_id"], keep="last")
            .reset_index(drop=True)
        )
    else:
        df = pd.DataFrame(columns=["datetime_hour", "region_id", "alarm_active", "alarm_minutes_in_hour"])

    all_hours = pd.date_range(start_hour, end_hour, freq="h")
    backbone = (
        pd.MultiIndex.from_product([all_hours, region_dim["region_id"]], names=["datetime_hour", "region_id"])
        .to_frame(index=False)
        .merge(region_dim, on="region_id", how="left")
    )
    out = backbone.merge(
        df[["datetime_hour", "region_id", "alarm_active", "alarm_minutes_in_hour"]],
        on=["datetime_hour", "region_id"],
        how="left",
    )
    out["alarm_active"] = pd.to_numeric(out["alarm_active"], errors="coerce").fillna(0).astype(int)
    out["alarm_minutes_in_hour"] = pd.to_numeric(out["alarm_minutes_in_hour"], errors="coerce").fillna(0.0)
    out = remove_spring_dst_hour(out, dt_col="datetime_hour")
    return out.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)


def preprocess_isw_new_rows(
    store: SnapshotStore,
    topic_columns: list[str],
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    artifacts_dir: Path,
) -> pd.DataFrame:
    files = store.list_json_files("isw_reports")
    if not files:
        raise FileNotFoundError("No isw_reports snapshot files were found.")

    start_date = start_hour.floor("D")
    end_date = end_hour.floor("D")
    rows: list[dict[str, Any]] = []
    for path_ref in files:
        if not maybe_file_relevant_by_name(path_ref, start_hour, end_hour):
            continue
        payload = store.read_json(path_ref)
        if isinstance(payload, dict):
            dt = pd.to_datetime(payload.get("date"), errors="coerce")
            if pd.notna(dt) and start_date <= dt.floor("D") <= end_date:
                rows.append(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", *topic_columns])

    for col in ISW_TEXT_TOPLEVEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].replace("None", np.nan)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["date", *topic_columns])

    pre = ISWTextPreprocessor()
    df = df.dropna(subset=["title", "url", "text"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", *topic_columns])

    df["text"] = df["text"].apply(pre.clean_isw_text)
    df["text_final"] = df["text"].apply(pre.smart_preprocess_cached)

    vectorizer_path = artifacts_dir / "isw_vectorizer.pkl"
    svd_path = artifacts_dir / "isw_svd.pkl"
    if not vectorizer_path.exists() or not svd_path.exists():
        raise FileNotFoundError("Missing ISW artifacts in data/nlp_artifacts.")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(svd_path, "rb") as f:
        svd = pickle.load(f)

    X_sparse = vectorizer.transform(df["text_final"])
    X_reduced = svd.transform(X_sparse)
    if X_reduced.shape[1] != len(topic_columns):
        raise ValueError(
            f"ISW artifact mismatch: transformed {X_reduced.shape[1]} cols, template expects {len(topic_columns)}"
        )

    topic_df = pd.DataFrame(X_reduced, columns=topic_columns, index=df.index)
    out = pd.concat([df[["date"]].copy(), topic_df], axis=1)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def normalize_telegram_dates_to_local(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert(KYIV_TZ).dt.tz_localize(None)


def read_telegram_messages_new(store: SnapshotStore, start_hour: pd.Timestamp, end_hour: pd.Timestamp) -> pd.DataFrame:
    files = store.list_json_files("telegram")
    if not files:
        raise FileNotFoundError("No telegram snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        if not maybe_file_relevant_by_name(path_ref, start_hour, end_hour):
            continue
        payload = store.read_json(path_ref)
        if isinstance(payload, list):
            rows.extend(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=TELEGRAM_TEXT_TOPLEVEL_COLUMNS)

    for col in TELEGRAM_TEXT_TOPLEVEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["date"] = normalize_telegram_dates_to_local(df["date"])
    df = df[df["date"].notna()].copy()
    df["datetime_hour"] = df["date"].dt.floor("h")
    df = df[(df["datetime_hour"] >= start_hour) & (df["datetime_hour"] <= end_hour)].copy()
    if df.empty:
        return pd.DataFrame(columns=[*TELEGRAM_TEXT_TOPLEVEL_COLUMNS, "datetime_hour"])
    df["message"] = df["message"].astype(str)
    df["channel"] = df["channel"].astype(str)
    return df.reset_index(drop=True)


def build_telegram_hourly_topics(
    telegram_raw_df: pd.DataFrame,
    topic_columns: list[str],
    artifacts_dir: Path,
) -> pd.DataFrame:
    if telegram_raw_df.empty:
        return pd.DataFrame(columns=["datetime_hour", *topic_columns])

    df = telegram_raw_df.copy()
    pre = TelegramTextPreprocessor()
    df["message_clean"] = df["message"].apply(pre.fast_tg_clean_optimized)

    vectorizer_path = artifacts_dir / "tg_vectorizer.pkl"
    svd_path = artifacts_dir / "tg_svd.pkl"
    if not vectorizer_path.exists() or not svd_path.exists():
        raise FileNotFoundError("Missing Telegram artifacts in data/nlp_artifacts.")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(svd_path, "rb") as f:
        svd = pickle.load(f)

    X_sparse = vectorizer.transform(df["message_clean"])
    X_reduced = svd.transform(X_sparse)
    if X_reduced.shape[1] != len(topic_columns):
        raise ValueError(
            f"Telegram artifact mismatch: transformed {X_reduced.shape[1]} cols, template expects {len(topic_columns)}"
        )

    topic_df = pd.DataFrame(X_reduced, columns=topic_columns, index=df.index)
    msg_features = pd.concat([df[["datetime_hour"]].copy(), topic_df], axis=1)
    hourly = (
        msg_features.groupby("datetime_hour", as_index=False)[topic_columns]
        .mean()
        .sort_values("datetime_hour")
        .reset_index(drop=True)
    )
    return hourly


def build_tg_region_features(telegram_raw_df: pd.DataFrame) -> pd.DataFrame:
    if telegram_raw_df.empty:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    df = telegram_raw_df.copy()
    df["message"] = df["message"].astype(str)
    df["has_threat"] = df["message"].str.contains(TG_THREAT_PATTERN, case=False, regex=True).astype(int)
    df["has_allclear"] = df["message"].str.contains(TG_ALLCLEAR_PATTERN, case=False, regex=True).astype(int)

    region_dfs: list[pd.DataFrame] = []
    for region_id, pattern in TG_REGION_PATTERNS.items():
        mask = df["message"].str.contains(pattern, case=False, regex=True)
        tmp = df.loc[mask, ["datetime_hour", "has_threat", "has_allclear"]].copy()
        if tmp.empty:
            continue
        tmp["region_id"] = region_id
        region_dfs.append(tmp)

    if not region_dfs:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    df_region = pd.concat(region_dfs, ignore_index=True)
    hourly = (
        df_region.groupby(["datetime_hour", "region_id"], as_index=False)
        .agg(
            tg_region_threat_count=("has_threat", "sum"),
            tg_region_allclear_count=("has_allclear", "sum"),
            tg_region_mention_count=("has_threat", "count"),
        )
        .sort_values(["datetime_hour", "region_id"])
        .reset_index(drop=True)
    )
    for col in TG_REGION_COLS:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce").fillna(0)
    return hourly



def read_region_dim(unshifted_parquet: Path) -> pd.DataFrame:
    cols = ["region_id", "region_key"]
    df = read_parquet_any(unshifted_parquet, columns=cols)
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.drop_duplicates().sort_values(["region_id", "region_key"]).reset_index(drop=True)
    if df["region_id"].nunique() != EXPECTED_REGION_COUNT:
        warn(f"Expected {EXPECTED_REGION_COUNT} regions, got {df['region_id'].nunique()} in unshifted parquet.")
    return df


def drop_unshifted_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c in UNSHIFTED_DERIVED_COLS]
    return df.drop(columns=drop_cols, errors="ignore")


def build_isw_daily_with_last_report(
    isw_new: pd.DataFrame,
    unshifted_parquet: Path,
    topic_columns: list[str],
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Build day-level ISW topics for the new hourly range.

    ISW reports are not hourly and are not guaranteed to appear every day/hour.
    For missing days we must carry forward the latest available report instead of
    filling zeros. The seed comes from merged_sources_unshifted.parquet before
    start_hour, and any new reports inside [start_hour, end_hour] overwrite the
    carried-forward value for their report date.
    """
    if not topic_columns:
        return pd.DataFrame(columns=["date"])

    start_date = pd.Timestamp(start_hour).floor("D")
    end_date = pd.Timestamp(end_hour).floor("D")
    calendar = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    calendar["date"] = calendar["date"].dt.strftime("%Y-%m-%d")

    new_reports = isw_new.copy()
    if not new_reports.empty:
        new_reports = ensure_date_string_col(new_reports, "date")
        keep_cols = ["date", *[c for c in topic_columns if c in new_reports.columns]]
        new_reports = new_reports[keep_cols].copy()
        for col in topic_columns:
            if col not in new_reports.columns:
                new_reports[col] = np.nan
        new_reports = (
            new_reports[["date", *topic_columns]]
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
    else:
        new_reports = pd.DataFrame(columns=["date", *topic_columns])


    seed_df = pd.DataFrame(columns=["date", *topic_columns])
    hist_end = pd.Timestamp(start_hour) - pd.Timedelta(hours=1)
    hist_start = hist_end - pd.Timedelta(days=lookback_days)
    try:
        hist = read_parquet_range_by_hours(
            unshifted_parquet,
            hist_start,
            hist_end,
            columns=["datetime_hour", *topic_columns],
        )
    except Exception as exc:
        warn(f"Could not read historical ISW seed from unshifted parquet: {exc}")
        hist = pd.DataFrame(columns=["datetime_hour", *topic_columns])

    if not hist.empty:
        hist["datetime_hour"] = pd.to_datetime(hist["datetime_hour"], errors="coerce")
        hist = hist[hist["datetime_hour"].notna()].copy()
        for col in topic_columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

        non_empty = hist[hist[topic_columns].abs().sum(axis=1) > 0].copy()
        if not non_empty.empty:
            last = non_empty.sort_values("datetime_hour").tail(1).copy()
            seed_df = last[topic_columns].copy()
            seed_df.insert(0, "date", (start_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    combined = calendar.merge(new_reports, on="date", how="left")
    if not seed_df.empty:
        combined = pd.concat([seed_df, combined], ignore_index=True)
    elif new_reports.empty:
        warn(
            "No new ISW report and no previous non-zero ISW seed found; "
            "isw_topic_* will fall back to zeros. This should only happen at the very beginning of history."
        )

    combined = combined.sort_values("date").reset_index(drop=True)
    combined[topic_columns] = combined[topic_columns].ffill()
    combined[topic_columns] = combined[topic_columns].fillna(0)
    combined = combined[combined["date"].isin(calendar["date"])].copy()
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    carried_days = int(combined[topic_columns].abs().sum(axis=1).gt(0).sum()) if topic_columns else 0
    log(
        f"ISW daily table for merge: {len(combined)} days, "
        f"new_reports={len(new_reports)}, non_zero_days_after_ffill={carried_days}"
    )
    return combined[["date", *topic_columns]]


def merge_new_sources(
    weather_df: pd.DataFrame,
    alarms_df: pd.DataFrame,
    isw_df: pd.DataFrame,
    tg_hourly_df: pd.DataFrame,
    tg_region_df: pd.DataFrame,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
) -> pd.DataFrame:
    if weather_df.empty:
        raise ValueError(
            f"No weather rows for {start_hour} → {end_hour}; cannot build the merged hourly backbone."
        )

    df = weather_df.copy().sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    df = df.merge(
        alarms_df[["datetime_hour", "region_id", "alarm_active", "alarm_minutes_in_hour"]],
        on=["datetime_hour", "region_id"],
        how="left",
    )
    df["alarm_active"] = pd.to_numeric(df["alarm_active"], errors="coerce").fillna(0).astype(int)
    df["alarm_minutes_in_hour"] = pd.to_numeric(df["alarm_minutes_in_hour"], errors="coerce").fillna(0.0)

    isw_merge = isw_df.copy()
    if not isw_merge.empty:
        isw_merge = ensure_date_string_col(isw_merge, "date")
        df = df.merge(isw_merge, left_on="day_datetime", right_on="date", how="left")
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    else:
        warn("ISW table for new hours is empty even after last-report carry-forward; isw_topic_* may fall back to zeros.")

    all_hours = pd.DataFrame({"datetime_hour": pd.date_range(start_hour, end_hour, freq="h")})
    all_hours = remove_spring_dst_hour(all_hours, dt_col="datetime_hour")
    if not tg_hourly_df.empty:
        tg_hourly_full = all_hours.merge(tg_hourly_df, on="datetime_hour", how="left")
    else:
        tg_hourly_full = all_hours.copy()
    tg_cols = [c for c in tg_hourly_full.columns if c.startswith("tg_topic_")]
    if tg_cols:
        tg_hourly_full[tg_cols] = tg_hourly_full[tg_cols].fillna(0)
    df = df.merge(tg_hourly_full, on="datetime_hour", how="left")

    isw_cols = [c for c in df.columns if re.fullmatch(r"isw_topic_\d+", c)]
    tg_cols = [c for c in df.columns if re.fullmatch(r"tg_topic_\d+", c)]
    if isw_cols:
        df[isw_cols] = df[isw_cols].fillna(0)
    if tg_cols:
        df[tg_cols] = df[tg_cols].fillna(0)

    if not tg_region_df.empty:
        tg_region_merge = tg_region_df[["datetime_hour", "region_id", *TG_REGION_COLS]].copy()
        tg_region_merge["datetime_hour"] = pd.to_datetime(tg_region_merge["datetime_hour"], errors="coerce")
        tg_region_merge["region_id"] = pd.to_numeric(tg_region_merge["region_id"], errors="coerce").astype(int)
        df = df.merge(tg_region_merge, on=["datetime_hour", "region_id"], how="left")
        df[TG_REGION_COLS] = df[TG_REGION_COLS].fillna(0)
    else:
        for col in TG_REGION_COLS:
            df[col] = 0

    df = df.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    return df


def apply_unshifted_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    df = df.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)
    g = df.groupby("region_id", sort=False)

    df["alarm_lag_1"] = g["alarm_active"].shift(1)
    df["alarm_lag_3"] = g["alarm_active"].shift(3)
    df["alarm_lag_6"] = g["alarm_active"].shift(6)
    df["alarm_lag_12"] = g["alarm_active"].shift(12)
    lag_cols = ["alarm_lag_1", "alarm_lag_3", "alarm_lag_6", "alarm_lag_12"]
    df[lag_cols] = df[lag_cols].fillna(0)

    df["alarms_in_last_24h"] = g["alarm_active"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).sum()
    )
    df["alarms_in_last_24h"] = df["alarms_in_last_24h"].fillna(0)

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)

    hourly_total = df.groupby("datetime_hour")["alarm_active"].sum().shift(1)
    df["total_active_alarms_lag1"] = df["datetime_hour"].map(hourly_total).fillna(0)

    alarms_matrix = df.pivot_table(index="datetime_hour", columns="region_id", values="alarm_active", fill_value=0)
    neighbour_alarm_matrix = pd.DataFrame(index=alarms_matrix.index)
    for region, neighbours in NEIGHBOURING_REGIONS.items():
        valid_neighbours = [n for n in neighbours if n in alarms_matrix.columns]
        neighbour_alarm_matrix[region] = alarms_matrix[valid_neighbours].sum(axis=1) if valid_neighbours else 0
    neighbour_alarm_matrix = neighbour_alarm_matrix.shift(1)
    neighbour_alarm_long = neighbour_alarm_matrix.stack().reset_index()
    neighbour_alarm_long.columns = ["datetime_hour", "region_id", "neighbour_alarms"]
    df = df.merge(neighbour_alarm_long, on=["datetime_hour", "region_id"], how="left")
    df["neighbour_alarms"] = df["neighbour_alarms"].fillna(0)

    df["hours_since_last_alarm"] = df.groupby("region_id")["alarm_active"].transform(hours_since_last_alarm_vectorized)

    isw_topic_cols = [c for c in df.columns if re.fullmatch(r"isw_topic_\d+", c)]
    if isw_topic_cols:
        df[isw_topic_cols] = df[isw_topic_cols].fillna(0)
        df_isw_abs = df[isw_topic_cols].abs()
        df["isw_total_intensity"] = df_isw_abs.sum(axis=1)
        df["isw_topic_std"] = df_isw_abs.std(axis=1)
        df["isw_topic_max"] = df_isw_abs.max(axis=1)
        df["isw_topic_mean"] = df_isw_abs.mean(axis=1)
        df["isw_topic_entropy"] = safe_entropy_from_abs(df_isw_abs)
        df["isw_velocity_24h"] = df.groupby("region_id")["isw_total_intensity"].diff(24).fillna(0)
        df["isw_intensity_ema"] = df.groupby("region_id")["isw_total_intensity"].transform(
            lambda x: x.shift(1).ewm(span=24).mean()
        )
        df[["isw_velocity_24h", "isw_intensity_ema", "isw_topic_entropy"]] = df[
            ["isw_velocity_24h", "isw_intensity_ema", "isw_topic_entropy"]
        ].fillna(0)

    tg_topic_cols = [c for c in df.columns if re.fullmatch(r"tg_topic_\d+", c)]
    if tg_topic_cols:
        df[tg_topic_cols] = df[tg_topic_cols].fillna(0)
        df_tg_abs = df[tg_topic_cols].abs()
        df["tg_total_intensity"] = df_tg_abs.sum(axis=1)
        df["tg_topic_std"] = df_tg_abs.std(axis=1)
        df["tg_topic_max"] = df_tg_abs.max(axis=1)
        df["tg_topic_entropy"] = safe_entropy_from_abs(df_tg_abs)
        df["tg_velocity_3h"] = df.groupby("region_id")["tg_total_intensity"].diff(3).fillna(0)
        df["tg_intensity_ema_6h"] = df.groupby("region_id")["tg_total_intensity"].transform(
            lambda x: x.ewm(span=6).mean()
        )
        df["tg_intensity_zscore"] = df.groupby("region_id")["tg_total_intensity"].transform(
            lambda x: (x - x.rolling(24, min_periods=1).mean()) / (x.rolling(24, min_periods=1).std() + 1e-9)
        )
        tg_features_cols = [c for c in df.columns if ("tg_" in c and "topic" not in c)]
        df[tg_features_cols] = df[tg_features_cols].fillna(0)

    df = ensure_date_string_col(df, "day_datetime")
    return df.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)


def continue_hours_since_last_alarm_exact(new_unshifted: pd.DataFrame, last_state: pd.DataFrame) -> pd.DataFrame:
    if new_unshifted.empty or last_state.empty or "hours_since_last_alarm" not in last_state.columns:
        return new_unshifted

    out = new_unshifted.copy().sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)
    state = {}
    for row in last_state.sort_values("datetime_hour").itertuples(index=False):
        region_id = int(getattr(row, "region_id"))
        state[region_id] = {
            "prev_alarm": int(getattr(row, "alarm_active")),
            "prev_hours": float(getattr(row, "hours_since_last_alarm")),
        }

    values = []
    for row in out[["region_id", "alarm_active"]].itertuples(index=False):
        region_id = int(row.region_id)
        prev = state.get(region_id)
        if prev is None:
            cur_hours = 0.0
        elif int(prev["prev_alarm"]) == 1:
            cur_hours = 0.0
        else:
            cur_hours = float(prev["prev_hours"]) + 1.0
        values.append(cur_hours)
        state[region_id] = {"prev_alarm": int(row.alarm_active), "prev_hours": cur_hours}

    out["hours_since_last_alarm"] = values
    return out


def apply_final_source_shifts(unshifted_df: pd.DataFrame) -> pd.DataFrame:
    df = unshifted_df.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)

    isw_cols = [c for c in df.columns if c.startswith("isw_")]
    if isw_cols:
        df[isw_cols] = df.groupby("region_id")[isw_cols].shift(24).fillna(0)

    tg_cols = [c for c in df.columns if c.startswith("tg_")]
    if tg_cols:
        df[tg_cols] = df.groupby("region_id")[tg_cols].shift(1).fillna(0)

    hour_weather_cols = [c for c in df.columns if c.startswith("hour_")]
    for col in hour_weather_cols:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df.groupby("region_id")[col].shift(1).fillna(False).astype(bool)
        else:
            df[col] = df.groupby("region_id")[col].shift(1).fillna(0)

    day_weather_cols = [
        c
        for c in df.columns
        if (c.startswith("day_") and c not in ["day_datetime", "day_sunrise", "day_sunset", "day_moonphase", "day_of_week"])
    ]
    if day_weather_cols:
        df[day_weather_cols] = df.groupby("region_id")[day_weather_cols].shift(24).fillna(0)

    if "alarm_minutes_in_hour" in df.columns:
        df["alarm_minutes_in_hour"] = df.groupby("region_id")["alarm_minutes_in_hour"].shift(1).fillna(0)

    df = ensure_date_string_col(df, "day_datetime")
    return df.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)



def load_latest_weather_forecast(store: SnapshotStore) -> tuple[pd.DataFrame, str | Path]:
    files = store.list_json_files("weather_forecast_24h")
    if not files:
        raise FileNotFoundError("No weather_forecast_24h snapshot files were found.")

    latest = files[-1]
    payload = store.read_json(latest)
    rows: list[dict[str, Any]] = []

    if not isinstance(payload, dict):
        raise ValueError("Forecast weather snapshot has unexpected structure.")
    for region_id_str, values in payload.items():
        for row in values:
            item = dict(row)
            item["region_id"] = int(item.get("region_id", region_id_str))
            rows.append(item)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, latest
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    drop_now = [c for c in WEATHER_DROP_COLUMNS if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)
    df = ensure_date_string_col(df, "day_datetime")
    bool_cols = [c for c in df.columns if c.startswith("hour_conditions_simple_")]
    df = coerce_bool_columns(df, bool_cols)
    df = (
        df.sort_values(["datetime_hour", "region_id"])
        .drop_duplicates(["datetime_hour", "region_id"], keep="last")
        .reset_index(drop=True)
    )
    return df, latest


def save_forecast_runtime_inputs(forecast_weather_df: pd.DataFrame, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = runtime_dir / "weather_forecast_processed.parquet"
    forecast_weather_df.to_parquet(forecast_path, index=False, engine="pyarrow")



def run_historical_pipeline(paths: ProjectPaths) -> None:
    store = SnapshotStore(paths.snapshots)
    completed_hour = infer_completed_hour()

    ensure_single_parquet_file(paths.unshifted_parquet, "merged_sources_unshifted.parquet")
    ensure_single_parquet_file(paths.final_parquet, "final_merged_dataset.parquet")

    unshifted_schema = parquet_schema(paths.unshifted_parquet)
    final_schema = parquet_schema(paths.final_parquet)

    existing_max = parquet_max_timestamp(paths.unshifted_parquet, "datetime_hour")
    if existing_max is None:
        raise ValueError(f"Could not infer max datetime_hour from {paths.unshifted_parquet}")

    start_hour = existing_max + pd.Timedelta(hours=1)
    end_hour = completed_hour

    log("Historical incremental step started")
    log(f"existing_max_unshifted={existing_max}")
    log(f"completed_hour cutoff={completed_hour}")

    has_new_source_hours = start_hour <= end_hour
    if not has_new_source_hours:
        log("No completed new source hours to append to merged_sources_unshifted.parquet")
        end_hour = existing_max

    region_dim = read_region_dim(paths.unshifted_parquet)
    isw_topic_cols = [c for c in unshifted_schema["columns"] if re.fullmatch(r"isw_topic_\d+", c)]
    tg_topic_cols = [c for c in unshifted_schema["columns"] if re.fullmatch(r"tg_topic_\d+", c)]

    if has_new_source_hours:
        log(f"Building new source rows for {start_hour} → {end_hour}")
        weather_new = preprocess_weather_new_rows(store, start_hour, end_hour)
        alarms_new = preprocess_alarms_new_rows(store, start_hour, end_hour, region_dim)
        isw_new = preprocess_isw_new_rows(store, isw_topic_cols, start_hour, end_hour, paths.artifacts_dir)
        isw_daily_for_merge = build_isw_daily_with_last_report(
            isw_new,
            paths.unshifted_parquet,
            isw_topic_cols,
            start_hour,
            end_hour,
        )
        telegram_raw_new = read_telegram_messages_new(store, start_hour, end_hour)
        tg_hourly_new = build_telegram_hourly_topics(telegram_raw_new, tg_topic_cols, paths.artifacts_dir)
        tg_region_new = build_tg_region_features(telegram_raw_new)

        log(
            "New rows — "
            f"weather: {len(weather_new)}, alarms backbone: {len(alarms_new)}, "
            f"ISW reports: {len(isw_new)}, ISW merge days: {len(isw_daily_for_merge)}, "
            f"telegram messages: {len(telegram_raw_new)}, "
            f"tg hourly: {len(tg_hourly_new)}, tg region: {len(tg_region_new)}"
        )

        merged_new_raw = merge_new_sources(
            weather_new,
            alarms_new,
            isw_daily_for_merge,
            tg_hourly_new,
            tg_region_new,
            start_hour,
            end_hour,
        )

        expected_rows = len(remove_spring_dst_hour(pd.DataFrame({"datetime_hour": pd.date_range(start_hour, end_hour, freq="h")}))) * region_dim["region_id"].nunique()
        if len(merged_new_raw) != expected_rows:
            warn(f"Merged new raw rows = {len(merged_new_raw):,}; expected around {expected_rows:,}. Check missing weather/regions.")

        log("Reading historical tail from unshifted parquet")
        unshifted_tail = read_parquet_tail_by_hours(paths.unshifted_parquet, existing_max, HISTORY_TAIL_HOURS_FOR_UNSHIFTED)
        last_state_cols = ["datetime_hour", "region_id", "alarm_active", "hours_since_last_alarm"]
        last_state = (
            unshifted_tail[last_state_cols]
            .sort_values(["region_id", "datetime_hour"])
            .groupby("region_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        raw_tail = drop_unshifted_derived_columns(unshifted_tail)
        combined_raw = pd.concat([raw_tail, merged_new_raw], ignore_index=True)
        combined_raw = combined_raw.sort_values(["region_id", "datetime_hour"]).drop_duplicates(
            ["datetime_hour", "region_id"], keep="last"
        )

        log("Applying unshifted feature engineering on tail + new rows")
        recomputed_unshifted = apply_unshifted_feature_engineering(combined_raw)
        new_unshifted = recomputed_unshifted[recomputed_unshifted["datetime_hour"] >= start_hour].copy()
        new_unshifted = continue_hours_since_last_alarm_exact(new_unshifted, last_state)
        new_unshifted = align_to_schema(new_unshifted, unshifted_schema, strict_extra=False, strict_missing=False)
        new_unshifted = new_unshifted.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)

        rewrite_parquet_file_with_new_rows(
            paths.unshifted_parquet,
            new_unshifted,
            "merged_sources_unshifted.parquet",
            key_cols=["datetime_hour", "region_id"],
        )

    else:
        log("Skipping source preprocessing; unshifted parquet is already up to date")

    unshifted_max_after_write = parquet_max_timestamp(paths.unshifted_parquet, "datetime_hour")
    if unshifted_max_after_write is None:
        raise ValueError(f"Could not infer max datetime_hour from {paths.unshifted_parquet} after rewrite")

    comparison_start = min(
        start_hour,
        unshifted_max_after_write - pd.Timedelta(days=90),
    )
    final_rebuild_start = find_earliest_missing_final_hour(
        paths.unshifted_parquet,
        paths.final_parquet,
        comparison_start,
        unshifted_max_after_write,
    )

    if final_rebuild_start is None:
        log("No missing final rows detected; final_merged_dataset.parquet was not rewritten")
    else:
        log(f"Building final shifted rows from {final_rebuild_start} → {unshifted_max_after_write}")
        final_input_start = final_rebuild_start - pd.Timedelta(hours=HISTORY_TAIL_HOURS_FOR_FINAL)
        final_input = read_parquet_range_by_hours(
            paths.unshifted_parquet,
            final_input_start,
            unshifted_max_after_write,
        )
        final_input = final_input.sort_values(["region_id", "datetime_hour"]).drop_duplicates(
            ["datetime_hour", "region_id"], keep="last"
        )
        shifted = apply_final_source_shifts(final_input)
        final_new = shifted[
            (shifted["datetime_hour"] >= final_rebuild_start)
            & (shifted["datetime_hour"] <= unshifted_max_after_write)
        ].copy()
        final_new = align_to_schema(final_new, final_schema, strict_extra=True, strict_missing=True)
        final_new = final_new.sort_values(["region_id", "datetime_hour"]).reset_index(drop=True)

        rewrite_parquet_file_with_new_rows(
            paths.final_parquet,
            final_new,
            "final_merged_dataset.parquet",
            key_cols=["datetime_hour", "region_id"],
        )

    log_recent_dataset_health(paths.unshifted_parquet, "merged_sources_unshifted.parquet")
    log_recent_dataset_health(paths.final_parquet, "final_merged_dataset.parquet")
    log("Historical incremental step finished")


def run_forecast_pipeline(paths: ProjectPaths) -> None:
    store = SnapshotStore(paths.snapshots)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)

    log("Preparing runtime forecast weather input")
    forecast_weather, latest_ref = load_latest_weather_forecast(store)
    log(f"Using latest forecast snapshot: {latest_ref}")
    save_forecast_runtime_inputs(forecast_weather, paths.runtime_dir)
    log("weather_forecast_processed.parquet updated")


def main() -> None:
    paths = ProjectPaths.default()
    paths.validate()

    log("Starting hourly two-parquet pipeline")
    run_historical_pipeline(paths)
    run_forecast_pipeline(paths)
    log("Hourly pipeline finished successfully")


if __name__ == "__main__":
    main()
