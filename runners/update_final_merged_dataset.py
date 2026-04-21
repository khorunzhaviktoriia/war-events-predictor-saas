"""
1) read raw snapshots
2) preprocess the new source-level rows.
3) append those rows to the old processed source tables
(final_weather.csv, war_events_processed.csv,isw_processed_svd.csv, telegram_processed_svd.csv)
4) rebuild final_merged_dataset.parquet from the updated processed sources
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


warnings.filterwarnings("ignore", category=FutureWarning)

KYIV_TZ = "Europe/Kyiv"

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
    2:  r'вінниц',
    3:  r'волин|луцьк',
    4:  r'дніпр|дніпропетровськ',
    5:  r'донецьк|донеччин',
    6:  r'житомир',
    7:  r'закарпат|ужгород',
    8:  r'запоріж',
    9:  r'івано.франків',
    10: r'київськ|київщин',
    11: r'кіровоград|кропивницьк',
    13: r'львів',
    14: r'миколаїв',
    15: r'одес',
    16: r'полтав',
    17: r'рівн',
    18: r'сум',
    19: r'тернопіл',
    20: r'харків|харківщин',
    21: r'херсон',
    22: r'хмельниц',
    23: r'черкас',
    24: r'чернівц',
    25: r'чернігів',
    26: r'київ(?!ськ)',
}

TG_THREAT_PATTERN = r'загроза|виліт|пуск|ракет|шахед|герань|бпла|дрон|балістич|крилат'
TG_ALLCLEAR_PATTERN = r'відбій'

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
    weather_csv: Path
    alarms_csv: Path
    isw_csv: Path
    telegram_csv: Path
    tg_region_csv: Path
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
            weather_csv=data_dir / "final_weather.csv",
            alarms_csv=data_dir / "war_events_processed.csv",
            isw_csv=data_dir / "isw_processed_svd.csv",
            telegram_csv=data_dir / "telegram_processed_svd.csv",
            tg_region_csv=data_dir / "tg_region_features.csv",
            final_parquet=data_dir / "final_merged_dataset.parquet",
        )

    def validate(self) -> None:
        required_files = [self.weather_csv, self.alarms_csv, self.isw_csv, self.telegram_csv, self.tg_region_csv, self.final_parquet]

        if not self.snapshots.exists() or not self.snapshots.is_dir():
            raise FileNotFoundError(
                f"Missing snapshots folder: {self.snapshots}. Expected: data/raw_snapshots/"
            )
        if not self.artifacts_dir.exists() or not self.artifacts_dir.is_dir():
            raise FileNotFoundError(
                f"Missing NLP artifacts folder: {self.artifacts_dir}. Expected: data/nlp_artifacts/"
            )
        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")


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

    def read_json(self, path_ref: str | Path) -> Any:
        with open(path_ref, "r", encoding="utf-8") as f:
            return json.load(f)



def read_parquet_any(path: str | os.PathLike[str], columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise RuntimeError(
            "Could not read parquet. Install pyarrow or fastparquet in the project environment."
        ) from exc

def write_parquet_any(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not write parquet. Install pyarrow or fastparquet in the project environment."
        ) from exc



def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)

def append_csv_rows(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def csv_max_timestamp(path: Path, column: str, is_date_only: bool = False, chunksize: int = 200_000) -> pd.Timestamp | None:
    if not path.exists():
        return None
    max_val: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
        dt = pd.to_datetime(chunk[column], errors="coerce")
        if is_date_only:
            dt = dt.dt.floor("D")
        cur = dt.max()
        if pd.notna(cur):
            max_val = cur if max_val is None else max(max_val, cur)
    return max_val

def csv_region_dim(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed source file not found: {path}")
    df = pd.read_csv(path, usecols=["region_id", "region_key"], low_memory=False)
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df = df.drop_duplicates().sort_values(["region_id", "region_key"]).reset_index(drop=True)
    return df

def load_csv_full(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)



def infer_completed_hour(tz: str = KYIV_TZ) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz).tz_localize(None).floor("h") - pd.Timedelta(hours=1)

def remove_spring_dst_hour(df: pd.DataFrame, dt_col: str = "datetime_hour") -> pd.DataFrame:
    if df.empty:
        return df
    years = sorted(pd.to_datetime(df[dt_col]).dt.year.dropna().unique())
    rows_to_remove: list[pd.Timestamp] = []
    for year in years:
        last_day = pd.Timestamp(year=year, month=3, day=31)
        while last_day.weekday() != 6:
            last_day -= pd.Timedelta(days=1)
        rows_to_remove.append(last_day.replace(hour=3, minute=0, second=0))
    mask = pd.to_datetime(df[dt_col]).isin(rows_to_remove)
    removed = int(mask.sum())
    if removed:
        warn(f"Removing {removed} rows that match the old spring-DST cleanup rule.")
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

def align_to_template_columns(df: pd.DataFrame, template_columns: list[str]) -> pd.DataFrame:
    if not template_columns:
        return df
    out = df.copy()
    for col in template_columns:
        if col not in out.columns:
            if col.startswith("hour_conditions_simple_"):
                out[col] = False
            elif col in {"alarm_active", "region_id", "day_of_week", "hour"}:
                out[col] = 0
            elif re.fullmatch(r"(isw|tg)_topic_\d+", col):
                out[col] = 0.0
            elif col in {*TG_REGION_COLS}:
                out[col] = 0
            else:
                out[col] = 0
    extras = [c for c in out.columns if c not in template_columns]
    if extras:
        out = out.drop(columns=extras)
    return out[template_columns].copy()

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
        raise RuntimeError(
            "Telegram preprocessing needs pymorphy3 and stop-words. Install them first."
        ) from exc
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



def preprocess_weather_new_rows(
    store: SnapshotStore,
    template_columns: list[str],
    completed_hour: pd.Timestamp,
    existing_max: pd.Timestamp | None,
) -> pd.DataFrame:
    files = store.list_json_files("weather_historical")
    if not files:
        raise FileNotFoundError("No weather_historical snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        payload = store.read_json(path_ref)
        for item in payload.get("regions", []):
            weather = dict(item.get("weather", {}))
            weather.setdefault("region_id", item.get("region_id"))
            weather.setdefault("region_key", item.get("region"))
            rows.append(weather)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    df = df[df["datetime_hour"] <= completed_hour].copy()
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
    df = ensure_date_string_col(df, "day_datetime")

    df = (
        df.sort_values(["datetime_hour", "region_id"])
        .drop_duplicates(subset=["datetime_hour", "region_id"], keep="last")
        .reset_index(drop=True)
    )
    if existing_max is not None:
        df = df[df["datetime_hour"] > existing_max].copy()
    df = align_to_template_columns(df, template_columns)
    return df

def preprocess_alarms_new_rows(
    store: SnapshotStore,
    template_columns: list[str],
    completed_hour: pd.Timestamp,
    existing_max: pd.Timestamp | None,
    region_dim: pd.DataFrame,
) -> pd.DataFrame:
    files = store.list_json_files("alarms_hourly")
    if not files:
        raise FileNotFoundError("No alarms_hourly snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        payload = store.read_json(path_ref)
        if isinstance(payload, list):
            rows.extend(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    df = df[df["datetime_hour"] <= completed_hour].copy()
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    df["alarm_active"] = pd.to_numeric(df["alarm_active"], errors="coerce").fillna(0).astype(int)

    df = (
        df.sort_values(["datetime_hour", "region_id"])
        .drop_duplicates(subset=["datetime_hour", "region_id"], keep="last")
        .reset_index(drop=True)
    )

    if existing_max is not None:
        start_dt = max(df["datetime_hour"].min(), existing_max + pd.Timedelta(hours=1))
    else:
        start_dt = df["datetime_hour"].min()
    end_dt = df["datetime_hour"].max()
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        return pd.DataFrame(columns=template_columns or ["datetime_hour", "region_id", "region_key", "alarm_minutes_in_hour", "alarm_active"])

    all_hours = pd.date_range(start_dt, end_dt, freq="h")
    backbone = (
        pd.MultiIndex.from_product([all_hours, region_dim["region_id"]], names=["datetime_hour", "region_id"])
        .to_frame(index=False)
        .merge(region_dim, on="region_id", how="left")
    )

    df = df[["datetime_hour", "region_id", "alarm_active"]].copy()
    out = backbone.merge(df, on=["datetime_hour", "region_id"], how="left")
    out["alarm_active"] = out["alarm_active"].fillna(0).astype(int)
    out["alarm_minutes_in_hour"] = 0.0
    out = remove_spring_dst_hour(out, dt_col="datetime_hour")
    out = out.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    out = align_to_template_columns(out, template_columns)
    return out

def preprocess_isw_new_rows(
    store: SnapshotStore,
    topic_columns: list[str],
    existing_max_date: pd.Timestamp | None,
    artifacts_dir: Path,
) -> pd.DataFrame:
    files = store.list_json_files("isw_reports")
    if not files:
        raise FileNotFoundError("No isw_reports snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        payload = store.read_json(path_ref)
        if isinstance(payload, dict):
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
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if existing_max_date is not None:
        df = df[df["date"] > existing_max_date].copy()

    if df.empty:
        out = pd.DataFrame(columns=["date", *topic_columns])
        return out

    pre = ISWTextPreprocessor()
    df = df.dropna(subset=["title", "url", "text"]).copy()
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

def preprocess_telegram_new_rows(
    store: SnapshotStore,
    topic_columns: list[str],
    existing_max_dt: pd.Timestamp | None,
    artifacts_dir: Path,
) -> pd.DataFrame:
    files = store.list_json_files("telegram")
    if not files:
        raise FileNotFoundError("No telegram snapshot files were found.")

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        payload = store.read_json(path_ref)
        if isinstance(payload, list):
            rows.extend(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "channel", *topic_columns])

    for col in TELEGRAM_TEXT_TOPLEVEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["date"] = normalize_telegram_dates_to_local(df["date"])
    df = df[df["date"].notna()].copy()
    if existing_max_dt is not None:
        df = df[df["date"] > existing_max_dt].copy()

    if df.empty:
        out = pd.DataFrame(columns=["date", "channel", *topic_columns])
        return out

    df["message"] = df["message"].astype(str)
    df["channel"] = df["channel"].astype(str)

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
    out = pd.concat([df[["date", "channel"]].copy(), topic_df], axis=1)
    out = out.drop_duplicates(keep="last").sort_values(["date", "channel"]).reset_index(drop=True)
    return out

def preprocess_tg_region_new_rows(store: SnapshotStore,existing_max_dt: pd.Timestamp | None,) -> pd.DataFrame:
    files = store.list_json_files("telegram")
    if not files:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    rows: list[dict[str, Any]] = []
    for path_ref in files:
        payload = store.read_json(path_ref)
        if isinstance(payload, list):
            rows.extend(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    for col in TELEGRAM_TEXT_TOPLEVEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["date"] = normalize_telegram_dates_to_local(df["date"])
    df = df[df["date"].notna()].copy()
    if existing_max_dt is not None:
        df = df[df["date"] > existing_max_dt].copy()

    if df.empty:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    df["message"] = df["message"].astype(str)
    df["datetime_hour"] = df["date"].dt.floor("h")
    df["has_threat"] = df["message"].str.contains(TG_THREAT_PATTERN, case=False, regex=True).astype(int)
    df["has_allclear"] = df["message"].str.contains(TG_ALLCLEAR_PATTERN, case=False, regex=True).astype(int)

    region_dfs: list[pd.DataFrame] = []
    for region_id, pattern in TG_REGION_PATTERNS.items():
        mask = df["message"].str.contains(pattern, case=False, regex=True)
        tmp = df.loc[mask, ["datetime_hour", "has_threat", "has_allclear"]].copy()
        tmp["region_id"] = region_id
        region_dfs.append(tmp)

    if not region_dfs:
        return pd.DataFrame(columns=["datetime_hour", "region_id", *TG_REGION_COLS])

    df_region = pd.concat(region_dfs, ignore_index=True)
    hourly = (
        df_region.groupby(["datetime_hour", "region_id"])
        .agg(
            tg_region_threat_count=("has_threat", "sum"),
            tg_region_allclear_count=("has_allclear", "sum"),
            tg_region_mention_count=("has_threat", "count"),
        )
        .reset_index()
    )
    for col in TG_REGION_COLS:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce").fillna(0)
    return hourly


def load_tg_region_full(path: Path, new_rows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_csv_full(path)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype("Int64")
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)
    for col in TG_REGION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if new_rows_df is not None and not new_rows_df.empty:
        df = pd.concat([df, new_rows_df], ignore_index=True)
    df = (
        df.sort_values(["datetime_hour", "region_id"])
        .drop_duplicates(["datetime_hour", "region_id"], keep="last")
        .reset_index(drop=True)
    )
    return df


def load_weather_full(path: Path, new_rows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_csv_full(path)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype(int)
    bool_cols = [c for c in df.columns if c.startswith("hour_conditions_simple_")]
    df = coerce_bool_columns(df, bool_cols)
    df = ensure_date_string_col(df, "day_datetime")
    if new_rows_df is not None and not new_rows_df.empty:
        df = pd.concat([df, new_rows_df], ignore_index=True)
    df = remove_spring_dst_hour(df, dt_col="datetime_hour")
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last").reset_index(drop=True)
    return df

def load_alarms_full(path: Path, new_rows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_csv_full(path)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").astype(int)
    df["alarm_active"] = pd.to_numeric(df["alarm_active"], errors="coerce").fillna(0).astype(int)
    df["alarm_minutes_in_hour"] = pd.to_numeric(df["alarm_minutes_in_hour"], errors="coerce").fillna(0.0)
    if new_rows_df is not None and not new_rows_df.empty:
        df = pd.concat([df, new_rows_df], ignore_index=True)
    df = remove_spring_dst_hour(df, dt_col="datetime_hour")
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last").reset_index(drop=True)
    return df

def load_isw_full(path: Path, new_rows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_csv_full(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if new_rows_df is not None and not new_rows_df.empty:
        df = pd.concat([df, new_rows_df], ignore_index=True)
    df = df.sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)
    return df

def aggregate_telegram_hourly_from_csv(
    path: Path,
    new_rows_df: pd.DataFrame | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    header_cols = read_csv_header(path)
    topic_cols = [c for c in header_cols if re.fullmatch(r"tg_topic_\d+", c)]
    if not topic_cols:
        raise ValueError("Could not infer tg_topic_* columns from telegram_processed_svd.csv")

    sum_parts: list[pd.DataFrame] = []
    count_parts: list[pd.Series] = []

    usecols = ["date", *topic_cols]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[chunk["date"].notna()].copy()
        if chunk.empty:
            continue
        chunk["datetime_hour"] = chunk["date"].dt.floor("h")
        sum_parts.append(chunk.groupby("datetime_hour", as_index=True)[topic_cols].sum())
        count_parts.append(chunk.groupby("datetime_hour").size().rename("msg_count"))

    if new_rows_df is not None and not new_rows_df.empty:
        new_rows = new_rows_df.copy()
        new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
        new_rows = new_rows[new_rows["date"].notna()].copy()
        new_rows["datetime_hour"] = new_rows["date"].dt.floor("h")
        sum_parts.append(new_rows.groupby("datetime_hour", as_index=True)[topic_cols].sum())
        count_parts.append(new_rows.groupby("datetime_hour").size().rename("msg_count"))

    if sum_parts:
        sums = pd.concat(sum_parts).groupby(level=0).sum().sort_index()
        counts = pd.concat(count_parts).groupby(level=0).sum().sort_index()
        hourly = sums.div(counts, axis=0).reset_index()
        hourly = remove_spring_dst_hour(hourly, dt_col="datetime_hour")
        hourly = hourly.sort_values("datetime_hour").reset_index(drop=True)
    else:
        hourly = pd.DataFrame(columns=["datetime_hour", *topic_cols])
    return hourly



def merge_historical_sources(
    weather_df: pd.DataFrame,
    alarms_df: pd.DataFrame,
    isw_df: pd.DataFrame,
    tg_hourly_df: pd.DataFrame,
    tg_region_df: pd.DataFrame,
) -> pd.DataFrame:

    df = weather_df.copy().sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    df = df.merge(
        alarms_df[["datetime_hour", "region_id", "alarm_active", "alarm_minutes_in_hour"]],
        on=["datetime_hour", "region_id"],
        how="left",
    )
    df["alarm_active"] = pd.to_numeric(df["alarm_active"], errors="coerce").fillna(0).astype(int)
    df["alarm_minutes_in_hour"] = pd.to_numeric(df["alarm_minutes_in_hour"], errors="coerce").fillna(0.0)

    isw_merge = isw_df.copy()
    isw_merge = ensure_date_string_col(isw_merge, "date")
    df = df.merge(isw_merge, left_on="day_datetime", right_on="date", how="left")
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    if not tg_hourly_df.empty:
        all_hours = pd.DataFrame({"datetime_hour": pd.date_range(df["datetime_hour"].min(), df["datetime_hour"].max(), freq="h")})
        all_hours = remove_spring_dst_hour(all_hours, dt_col="datetime_hour")
        tg_hourly_full = all_hours.merge(tg_hourly_df, on="datetime_hour", how="left")
        tg_cols = [c for c in tg_hourly_full.columns if c.startswith("tg_")]
        if tg_cols:
            tg_hourly_full[tg_cols] = tg_hourly_full[tg_cols].fillna(0)
        df = df.merge(tg_hourly_full, on="datetime_hour", how="left")
    else:
        warn("Telegram hourly table is empty; tg_* columns will be absent before schema alignment.")

    isw_cols = [c for c in df.columns if re.fullmatch(r"isw_topic_\d+", c)]
    tg_cols = [c for c in df.columns if re.fullmatch(r"tg_topic_\d+", c)]
    if isw_cols:
        df[isw_cols] = df[isw_cols].fillna(0)
    if tg_cols:
        df[tg_cols] = df[tg_cols].fillna(0)

    # merge tg_region_features
    if not tg_region_df.empty:
        tg_region_merge = tg_region_df[["datetime_hour", "region_id", *TG_REGION_COLS]].copy()
        tg_region_merge["datetime_hour"] = pd.to_datetime(tg_region_merge["datetime_hour"], errors="coerce")
        tg_region_merge["region_id"] = pd.to_numeric(tg_region_merge["region_id"], errors="coerce").astype(int)
        df = df.merge(tg_region_merge, on=["datetime_hour", "region_id"], how="left")
        df[TG_REGION_COLS] = df[TG_REGION_COLS].fillna(0)
    else:
        warn("tg_region_df is empty; tg_region_* columns will be zero-filled.")
        for col in TG_REGION_COLS:
            df[col] = 0

    df = df.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    g = df.groupby("region_id", sort=False)

    # alarm lags features
    alarm_block = pd.DataFrame(index=df.index)
    alarm_block["alarm_lag_1"] = g["alarm_active"].shift(1).fillna(0)
    alarm_block["alarm_lag_3"] = g["alarm_active"].shift(3).fillna(0)
    alarm_block["alarm_lag_6"] = g["alarm_active"].shift(6).fillna(0)
    alarm_block["alarm_lag_12"] = g["alarm_active"].shift(12).fillna(0)
    alarm_block["alarms_in_last_24h"] = g["alarm_active"].transform(lambda x: x.shift(1).rolling(24, min_periods=1).sum()).fillna(0)
    df = pd.concat([df, alarm_block], axis=1)

    # calendar features
    cal_block = pd.DataFrame(index=df.index)
    cal_block["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    cal_block["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)
    df = pd.concat([df, cal_block], axis=1)

    # total active alarms lagged by one hour
    hourly_total = df.groupby("datetime_hour")["alarm_active"].sum().shift(1)
    df["total_active_alarms_lag1"] = df["datetime_hour"].map(hourly_total).fillna(0)

    # neighbour alarms
    alarms_matrix = df.pivot_table(index="datetime_hour", columns="region_id", values="alarm_active", fill_value=0)
    neighbour_alarm_matrix = pd.DataFrame(index=alarms_matrix.index)
    for region, neighbours in NEIGHBOURING_REGIONS.items():
        valid_neighbours = [n for n in neighbours if n in alarms_matrix.columns]
        neighbour_alarm_matrix[region] = alarms_matrix[valid_neighbours].sum(axis=1) if valid_neighbours else 0
    neighbour_alarm_matrix = neighbour_alarm_matrix.shift(1)
    neighbour_alarm_long = neighbour_alarm_matrix.stack(dropna=False).reset_index()
    neighbour_alarm_long.columns = ["datetime_hour", "region_id", "neighbour_alarms"]
    df = df.merge(neighbour_alarm_long, on=["datetime_hour", "region_id"], how="left")
    df["neighbour_alarms"] = df["neighbour_alarms"].fillna(0)

    # hours since last alarm
    df["hours_since_last_alarm"] = g["alarm_active"].transform(hours_since_last_alarm_vectorized)

    # isw features.
    isw_topic_cols = [c for c in df.columns if re.fullmatch(r"isw_topic_\d+", c)]
    if isw_topic_cols:
        df[isw_topic_cols] = df[isw_topic_cols].fillna(0)
        df_isw_abs = df[isw_topic_cols].abs()
        isw_block = pd.DataFrame(index=df.index)
        isw_block["isw_total_intensity"] = df_isw_abs.sum(axis=1)
        isw_block["isw_topic_std"] = df_isw_abs.std(axis=1)
        isw_block["isw_topic_max"] = df_isw_abs.max(axis=1)
        isw_block["isw_topic_mean"] = df_isw_abs.mean(axis=1)
        isw_block["isw_topic_entropy"] = safe_entropy_from_abs(df_isw_abs)
        df = pd.concat([df, isw_block], axis=1)
        df["isw_velocity_24h"] = df.groupby("region_id")["isw_total_intensity"].diff(24).fillna(0)
        df["isw_intensity_ema"] = (
            df.groupby("region_id")["isw_total_intensity"].transform(lambda x: x.shift(1).ewm(span=24).mean()).fillna(0)
        )

    # telegram features
    tg_topic_cols = [c for c in df.columns if re.fullmatch(r"tg_topic_\d+", c)]
    if tg_topic_cols:
        df[tg_topic_cols] = df[tg_topic_cols].fillna(0)
        df_tg_abs = df[tg_topic_cols].abs()
        tg_block = pd.DataFrame(index=df.index)
        tg_block["tg_total_intensity"] = df_tg_abs.sum(axis=1)
        tg_block["tg_topic_std"] = df_tg_abs.std(axis=1)
        tg_block["tg_topic_max"] = df_tg_abs.max(axis=1)
        tg_block["tg_topic_entropy"] = safe_entropy_from_abs(df_tg_abs)
        df = pd.concat([df, tg_block], axis=1)
        df["tg_velocity_3h"] = df.groupby("region_id")["tg_total_intensity"].diff(3).fillna(0)
        df["tg_intensity_ema_6h"] = (
            df.groupby("region_id")["tg_total_intensity"].transform(lambda x: x.ewm(span=6).mean()).fillna(0)
        )
        df["tg_intensity_zscore"] = (
            df.groupby("region_id")["tg_total_intensity"]
            .transform(lambda x: (x - x.rolling(24, min_periods=1).mean()) / (x.rolling(24, min_periods=1).std() + 1e-9))
            .fillna(0)
        )

    # final train-ready shifts
    df_to_train = df.copy()

    isw_cols = [c for c in df_to_train.columns if c.startswith("isw_")]
    if isw_cols:
        df_to_train[isw_cols] = df_to_train.groupby("region_id")[isw_cols].shift(24).fillna(0)

    tg_cols = [c for c in df_to_train.columns if c.startswith("tg_")]
    if tg_cols:
        df_to_train[tg_cols] = df_to_train.groupby("region_id")[tg_cols].shift(1).fillna(0)

    hour_weather_cols = [c for c in df_to_train.columns if c.startswith("hour_")]
    for col in hour_weather_cols:
        if pd.api.types.is_bool_dtype(df_to_train[col]):
            df_to_train[col] = df_to_train.groupby("region_id")[col].shift(1).fillna(False).astype(bool)
        else:
            df_to_train[col] = df_to_train.groupby("region_id")[col].shift(1).fillna(0)

    day_weather_cols = [
        c for c in df_to_train.columns
        if (c.startswith("day_") and c not in ["day_datetime", "day_sunrise", "day_sunset", "day_moonphase", "day_of_week"])
    ]
    if day_weather_cols:
        df_to_train[day_weather_cols] = df_to_train.groupby("region_id")[day_weather_cols].shift(24).fillna(0)

    if "alarm_minutes_in_hour" in df_to_train.columns:
        df_to_train["alarm_minutes_in_hour"] = df_to_train.groupby("region_id")["alarm_minutes_in_hour"].shift(1).fillna(0)

    df_to_train = ensure_date_string_col(df_to_train, "day_datetime")
    df_to_train = df_to_train.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    return df_to_train


def cast_series_to_dtype(series: pd.Series, dtype_str: str) -> pd.Series:
    dtype_str = str(dtype_str)
    try:
        if "datetime" in dtype_str:
            return pd.to_datetime(series, errors="coerce")
        if dtype_str == "bool":
            return series.fillna(False).astype(bool)
        if dtype_str.startswith("int"):
            return pd.to_numeric(series, errors="coerce").fillna(0).astype(dtype_str)
        if dtype_str.startswith("float"):
            return pd.to_numeric(series, errors="coerce").astype(dtype_str)
        if dtype_str == "object":
            return series.astype(object)
        return series.astype(dtype_str)
    except Exception:
        return series

def get_existing_schema(final_parquet: Path) -> dict[str, Any] | None:
    if not final_parquet.exists():
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(final_parquet)
        columns = schema.names
        dtypes = {name: str(schema.field(name).type) for name in columns}
        return {"columns": list(columns), "dtypes": dtypes}
    except Exception:
        pass

    try:
        import fastparquet  # type: ignore

        pf = fastparquet.ParquetFile(str(final_parquet))
        return {"columns": list(pf.columns), "dtypes": {k: str(v) for k, v in pf.dtypes.items()}}
    except Exception:
        pass

    try:
        df = read_parquet_any(final_parquet)
        return {"columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
    except Exception as exc:
        warn(f"Could not read existing final parquet for schema alignment: {exc}")
        return None

def align_to_existing_schema(new_df: pd.DataFrame, existing_schema: dict[str, Any] | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if existing_schema is None:
        return new_df, {"schema_alignment": "skipped"}

    expected_cols = existing_schema["columns"]
    expected_dtypes = existing_schema["dtypes"]
    aligned = new_df.copy()
    missing_cols = [c for c in expected_cols if c not in aligned.columns]
    extra_cols = [c for c in aligned.columns if c not in expected_cols]

    if missing_cols or extra_cols:
        problems: list[str] = []
        if missing_cols:
            problems.append(f"missing columns: {missing_cols}")
        if extra_cols:
            problems.append(f"unexpected columns: {extra_cols}")
        raise ValueError(
            "Schema drift detected while rebuilding final_merged_dataset.parquet; "
            + "; ".join(problems)
        )

    aligned = aligned[expected_cols].copy()
    for col in expected_cols:
        aligned[col] = cast_series_to_dtype(aligned[col], expected_dtypes.get(col, "object"))

    return aligned, {
        "missing_columns_filled": [],
        "extra_columns_dropped": [],
        "final_column_count": len(aligned.columns),
    }


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
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    drop_now = [c for c in WEATHER_DROP_COLUMNS if c in df.columns]

    if drop_now:
        df = df.drop(columns=drop_now)

    df = ensure_date_string_col(df, "day_datetime")
    bool_cols = [c for c in df.columns if c.startswith("hour_conditions_simple_")]
    df = coerce_bool_columns(df, bool_cols)
    df = df.sort_values(["datetime_hour", "region_id"]).drop_duplicates(["datetime_hour", "region_id"], keep="last").reset_index(drop=True)

    return df, latest

def save_forecast_runtime_inputs(forecast_weather_df: pd.DataFrame, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = runtime_dir / "weather_forecast_processed.parquet"
    write_parquet_any(forecast_weather_df, forecast_path)


def run_historical_pipeline(paths: ProjectPaths) -> None:
    store = SnapshotStore(paths.snapshots)
    completed_hour_ts = infer_completed_hour()
    existing_final_schema = get_existing_schema(paths.final_parquet)

    log("Historical step started")
    log(f"completed_hour cutoff={completed_hour_ts}")

    weather_cols = read_csv_header(paths.weather_csv)
    alarms_cols = read_csv_header(paths.alarms_csv)
    isw_cols = [c for c in read_csv_header(paths.isw_csv) if re.fullmatch(r"isw_topic_\d+", c)]
    tg_cols = [c for c in read_csv_header(paths.telegram_csv) if re.fullmatch(r"tg_topic_\d+", c)]

    region_dim = csv_region_dim(paths.alarms_csv)
    source_max = {
        "weather": csv_max_timestamp(paths.weather_csv, "datetime_hour"),
        "alarms": csv_max_timestamp(paths.alarms_csv, "datetime_hour"),
        "isw": csv_max_timestamp(paths.isw_csv, "date", is_date_only=True),
        "telegram": csv_max_timestamp(paths.telegram_csv, "date"),
        "tg_region": csv_max_timestamp(paths.tg_region_csv, "datetime_hour"),
    }

    log("Processing weather snapshots")
    weather_new_rows = preprocess_weather_new_rows(store, weather_cols, completed_hour_ts, source_max["weather"])
    log(f"Weather new rows: {len(weather_new_rows)}")

    log("Processing alarms snapshots")
    alarms_new_rows = preprocess_alarms_new_rows(store, alarms_cols, completed_hour_ts, source_max["alarms"], region_dim)
    log(f"Alarms new rows: {len(alarms_new_rows)}")

    log("Processing ISW snapshots")
    isw_new_rows = preprocess_isw_new_rows(store, isw_cols, source_max["isw"], paths.artifacts_dir)
    log(f"ISW new rows: {len(isw_new_rows)}")

    log("Processing Telegram snapshots")
    telegram_new_rows = preprocess_telegram_new_rows(store, tg_cols, source_max["telegram"], paths.artifacts_dir)
    log(f"Telegram new rows: {len(telegram_new_rows)}")

    log("Processing Telegram region features")
    tg_region_new_rows = preprocess_tg_region_new_rows(store, source_max["tg_region"])
    log(f"Telegram region new rows: {len(tg_region_new_rows)}")

    append_csv_rows(paths.weather_csv, weather_new_rows)
    append_csv_rows(paths.alarms_csv, alarms_new_rows)
    append_csv_rows(paths.isw_csv, isw_new_rows)
    append_csv_rows(paths.telegram_csv, telegram_new_rows)
    append_csv_rows(paths.tg_region_csv, tg_region_new_rows)
    log("Processed source tables updated")

    log("Loading updated processed sources")
    weather_full = load_weather_full(paths.weather_csv)
    alarms_full = load_alarms_full(paths.alarms_csv)
    isw_full = load_isw_full(paths.isw_csv)
    tg_hourly_full = aggregate_telegram_hourly_from_csv(paths.telegram_csv)
    tg_region_full = load_tg_region_full(paths.tg_region_csv)

    log("Merging processed sources")
    merged = merge_historical_sources(weather_full, alarms_full, isw_full, tg_hourly_full, tg_region_full)

    log("Applying feature engineering")
    rebuilt = apply_feature_engineering(merged)
    rebuilt_aligned, _ = align_to_existing_schema(rebuilt, existing_final_schema)

    write_parquet_any(rebuilt_aligned, paths.final_parquet)
    log(f"final_merged_dataset.parquet updated: {paths.final_parquet}")


def run_forecast_pipeline(paths: ProjectPaths) -> None:
    store = SnapshotStore(paths.snapshots)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)

    log("Preparing runtime forecast inputs")
    log(f"snapshots={paths.snapshots}")
    log(f"runtime_dir={paths.runtime_dir}")

    forecast_weather, latest_ref = load_latest_weather_forecast(store)
    log(f"Using latest forecast snapshot: {latest_ref}")
    save_forecast_runtime_inputs(forecast_weather, paths.runtime_dir)
    log("weather_forecast_processed.parquet updated")


def main() -> None:
    paths = ProjectPaths.default()
    paths.validate()

    log("Starting hourly pipeline")
    run_historical_pipeline(paths)
    run_forecast_pipeline(paths)
    log("Hourly pipeline finished successfully")

if __name__ == "__main__":
    main()
