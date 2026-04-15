import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
TARGET_COL = "alarm_active"
PRIMARY_METRIC = "pr_auc"
ACCEPT_TOLERANCE = 1e-6
THRESHOLD_GRID = np.arange(0.30, 0.71, 0.05)
MODEL_FILENAME = "2__hist_gradient_boosting__v1.pkl"

DROP_COLS = [
    TARGET_COL,
    "alarm_minutes_in_hour",
    "datetime_hour",
    "day_datetime",
    "day_sunrise",
    "day_sunset",
    "city_name",
    "region_key",
    "neighbour_alarms",
    "total_active_alarms_lag1",
    "alarms_in_last_24h",
]

PARAM_GRID = [
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 100,
        "l2_regularization": 2.0,
    },
]


@dataclass
class Paths:
    project_root: Path
    data_path: Path
    model_path: Path
    logs_dir: Path
    backup_dir: Path

    @classmethod
    def default(cls) -> "Paths":
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / "models"
        return cls(
            project_root=project_root,
            data_path=project_root / "data" / "final_merged_dataset.parquet",
            model_path=models_dir / MODEL_FILENAME,
            logs_dir=models_dir / "retrain_logs",
            backup_dir=models_dir / "backups",
        )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}")


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"], errors="coerce")
    df = df[df["datetime_hour"].notna()].copy()
    df = df.sort_values(["datetime_hour", "region_id"]).reset_index(drop=True)
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X = df.drop(columns=DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(np.int8)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.empty:
        raise ValueError("Feature matrix X is empty after preprocessing.")

    return X, y


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
) -> dict[str, Any]:
    n = len(X)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    return {
        "X_train": X.iloc[:train_end].copy(),
        "y_train": y.iloc[:train_end].copy(),
        "X_valid": X.iloc[train_end:valid_end].copy(),
        "y_valid": y.iloc[train_end:valid_end].copy(),
        "X_test": X.iloc[valid_end:].copy(),
        "y_test": y.iloc[valid_end:].copy(),
        "meta_train": meta_df.iloc[:train_end].copy(),
        "meta_valid": meta_df.iloc[train_end:valid_end].copy(),
        "meta_test": meta_df.iloc[valid_end:].copy(),
    }


def fit_model_with_small_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[HistGradientBoostingClassifier, float, dict[str, Any], float]:
    tscv = TimeSeriesSplit(n_splits=3)

    best_params: dict[str, Any] | None = None
    best_cv_score = -np.inf
    best_oof_proba = pd.Series(index=X_train.index, dtype=float)

    for params in PARAM_GRID:
        fold_scores: list[float] = []
        oof_proba = pd.Series(index=X_train.index, dtype=float)

        log(f"Testing params: {params}")

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr)

            model = HistGradientBoostingClassifier(
                loss="log_loss",
                random_state=RANDOM_STATE,
                early_stopping=False,
                **params,
            )
            model.fit(X_tr, y_tr, sample_weight=sample_weight)

            val_proba = model.predict_proba(X_val)[:, 1]
            oof_proba.iloc[val_idx] = val_proba

            fold_pr_auc = average_precision_score(y_val, val_proba)
            fold_scores.append(fold_pr_auc)
            log(f"  Fold {fold}: PR-AUC={fold_pr_auc:.4f}")

        mean_score = float(np.mean(fold_scores))
        log(f"Mean CV PR-AUC={mean_score:.4f}")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_params = params
            best_oof_proba = oof_proba.copy()

    valid_mask = best_oof_proba.notna()
    y_oof = y_train.loc[valid_mask]
    proba_oof = best_oof_proba.loc[valid_mask]

    best_threshold = 0.50
    best_threshold_f1 = -np.inf
    for thr in THRESHOLD_GRID:
        pred_oof = (proba_oof >= thr).astype(int)
        score = f1_score(y_oof, pred_oof, zero_division=0)
        if score > best_threshold_f1:
            best_threshold_f1 = score
            best_threshold = float(thr)

    log(f"Best params: {best_params}")
    log(f"Best CV PR-AUC: {best_cv_score:.4f}")
    log(f"Best threshold: {best_threshold:.2f}")

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=RANDOM_STATE,
        early_stopping=False,
        **best_params,
    )
    final_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    return final_model, best_threshold, best_params, best_cv_score


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X_aligned = X.copy()
    for col in feature_names:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    X_aligned = X_aligned.reindex(columns=feature_names, fill_value=0)
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X_aligned


def evaluate_bundle(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_names = list(bundle["feature_names"])

    X_eval = align_features(X, feature_names)
    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": threshold,
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def load_existing_bundle(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    required = {"model", "threshold", "feature_names"}
    if not required.issubset(bundle.keys()):
        raise ValueError(f"Existing bundle is missing required keys: {required}")

    return bundle


def is_new_model_accepted(
    old_test_metrics: dict[str, Any] | None,
    new_test_metrics: dict[str, Any],
) -> tuple[bool, str]:
    if old_test_metrics is None:
        return True, "accepted: no previous production model found"

    old_primary = float(old_test_metrics[PRIMARY_METRIC])
    new_primary = float(new_test_metrics[PRIMARY_METRIC])

    if new_primary > old_primary + ACCEPT_TOLERANCE:
        return True, f"accepted: new {PRIMARY_METRIC} improved"

    if abs(new_primary - old_primary) <= ACCEPT_TOLERANCE:
        if float(new_test_metrics["f1"]) >= float(old_test_metrics["f1"]) - ACCEPT_TOLERANCE:
            return True, "accepted: PR-AUC tied and F1 is not worse"

    return False, f"rejected: new {PRIMARY_METRIC} is worse"


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def main() -> None:
    paths = Paths.default()
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.backup_dir.mkdir(parents=True, exist_ok=True)

    log("Starting retrain_top_model.py")
    log(f"Dataset path: {paths.data_path}")
    log(f"Model path: {paths.model_path}")

    df = load_dataset(paths.data_path)
    X, y = build_xy(df)
    split = chronological_split(X, y, df[["datetime_hour", "region_id"]])

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_valid = split["X_valid"]
    y_valid = split["y_valid"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    log(f"Rows: {len(df):,} | Features: {X.shape[1]} | Positive rate: {y.mean():.6f}")
    log(
        f"Split sizes -> train={len(X_train):,}, valid={len(X_valid):,}, test={len(X_test):,}"
    )

    new_model, new_threshold, best_params, best_cv_score = fit_model_with_small_search(X_train, y_train)
    new_bundle = {
        "model": new_model,
        "threshold": new_threshold,
        "feature_names": list(X_train.columns),
    }

    new_valid_metrics = evaluate_bundle(new_bundle, X_valid, y_valid)
    new_test_metrics = evaluate_bundle(new_bundle, X_test, y_test)
    log(f"New valid PR-AUC={new_valid_metrics['pr_auc']:.4f} | F1={new_valid_metrics['f1']:.4f}")
    log(f"New test  PR-AUC={new_test_metrics['pr_auc']:.4f} | F1={new_test_metrics['f1']:.4f}")

    old_bundle = load_existing_bundle(paths.model_path)
    old_valid_metrics = None
    old_test_metrics = None
    if old_bundle is not None:
        old_valid_metrics = evaluate_bundle(old_bundle, X_valid, y_valid)
        old_test_metrics = evaluate_bundle(old_bundle, X_test, y_test)
        log(f"Old valid PR-AUC={old_valid_metrics['pr_auc']:.4f} | F1={old_valid_metrics['f1']:.4f}")
        log(f"Old test  PR-AUC={old_test_metrics['pr_auc']:.4f} | F1={old_test_metrics['f1']:.4f}")

    accepted, reason = is_new_model_accepted(old_test_metrics, new_test_metrics)
    decision = "accepted" if accepted else "rejected"
    log(f"Decision: {decision.upper()} -> {reason}")

    backup_path = None
    if accepted:
        if paths.model_path.exists():
            backup_path = paths.backup_dir / f"backup_{paths.model_path.stem}_{now_tag()}.pkl"
            shutil.copy2(paths.model_path, backup_path)
            log(f"Backup saved: {backup_path}")

        save_bundle(new_bundle, paths.model_path)
        log("Production model updated.")
    else:
        log("Production model was not changed.")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "reason": reason,
        "primary_metric": PRIMARY_METRIC,
        "model_path": str(paths.model_path),
        "backup_path": str(backup_path) if backup_path else None,
        "best_params": best_params,
        "best_cv_pr_auc": float(best_cv_score),
        "new_threshold": float(new_threshold),
        "old_valid_metrics": old_valid_metrics,
        "old_test_metrics": old_test_metrics,
        "new_valid_metrics": new_valid_metrics,
        "new_test_metrics": new_test_metrics,
    }

    summary_path = paths.logs_dir / f"retrain_summary_{now_tag()}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"Summary saved: {summary_path}")
    log("retrain_top_model.py finished")


if __name__ == "__main__":
    main()
