from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from common import DATA_PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs, to_numeric


INPUT_FILE = DATA_PROCESSED_DIR / "metabolic_syndrome_labeled_dataset.csv"
MODEL_FILE = MODEL_DIR / "xgboost_metabolic_syndrome.joblib"
MODEL_FILE_LIGHT = MODEL_DIR / "xgboost_metabolic_syndrome_light.joblib"
METRIC_FILE = OUTPUT_DIR / "xgboost_metrics.json"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"
THRESHOLD_METRICS_FILE = OUTPUT_DIR / "xgboost_threshold_metrics.csv"
THRESHOLD_METRICS_LIGHT_FILE = OUTPUT_DIR / "xgboost_threshold_metrics_light.csv"
AB_COMPARE_FILE = OUTPUT_DIR / "xgboost_ab_comparison.csv"
AB_COMPARE_JSON_FILE = OUTPUT_DIR / "xgboost_ab_comparison.json"

BASE_FEATURE_COLS = [
    "nutrition_intake",
    "nutrition_prob_2",
    "waist_x_excess_prob",
    "sex",
    "HE_ht",
    "HE_wt",
    "HE_wc",
    "age",
]
SMOOTHING_FEATURE_CANDIDATES = [
    "pattern_energy_ratio_mean",
    "pattern_energy_ratio_std",
    "pattern_energy_ratio_iqr",
    "pattern_excess_day_ratio",
    "pattern_deficient_day_ratio",
    "pattern_balanced_day_ratio",
    "pattern_protein_density_mean",
    "pattern_fat_density_mean",
    "pattern_carb_density_mean",
    "pattern_sodium_density_mean",
]
TARGET_COL = "metabolic_syndrome"


def encode_sex(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    out_num = pd.Series(np.nan, index=series.index, dtype=float)
    out_num.loc[num == 1] = 1
    out_num.loc[num == 2] = 0

    remain = out_num.isna()
    if remain.any():
        s = series.astype(str).str.strip().str.lower()
        male_tokens = {"1", "1.0", "m", "male"}
        female_tokens = {"2", "2.0", "f", "female"}
        out_str = s.map(lambda x: 1 if x in male_tokens else (0 if x in female_tokens else np.nan))
        out_num.loc[remain] = out_str.loc[remain]
    return out_num


def scan_thresholds(y_true: pd.Series, pred_prob: np.ndarray, precision_floor: float = 0.50):
    rows = []
    for thr in np.arange(0.10, 0.91, 0.01):
        pred_thr = (pred_prob >= thr).astype(int)
        rows.append(
            {
                "threshold": float(round(thr, 2)),
                "accuracy": float(accuracy_score(y_true, pred_thr)),
                "precision": float(precision_score(y_true, pred_thr, zero_division=0)),
                "recall": float(recall_score(y_true, pred_thr, zero_division=0)),
                "f1": float(f1_score(y_true, pred_thr, zero_division=0)),
            }
        )

    best_f1_row = max(rows, key=lambda x: x["f1"])
    recall_candidates = [r for r in rows if r["precision"] >= precision_floor]
    if recall_candidates:
        best_recall_row = max(recall_candidates, key=lambda x: x["recall"])
    else:
        best_recall_row = max(rows, key=lambda x: x["recall"])

    return rows, best_f1_row, best_recall_row


def make_default_metrics(y_true: pd.Series, pred_prob: np.ndarray, threshold: float = 0.5):
    pred = (pred_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, pred_prob)),
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if "nutrition_prob_2" not in df.columns:
        raise ValueError("Missing required column: nutrition_prob_2. Re-run preprocess.py first.")
    df["nutrition_prob_2"] = pd.to_numeric(df["nutrition_prob_2"], errors="coerce")
    df["waist_x_excess_prob"] = pd.to_numeric(df["HE_wc"], errors="coerce") * df["nutrition_prob_2"]

    feature_cols = BASE_FEATURE_COLS + [c for c in SMOOTHING_FEATURE_CANDIDATES if c in df.columns]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    x = df[feature_cols].copy()
    x["sex"] = encode_sex(x["sex"])
    x = to_numeric(x, [c for c in feature_cols if c != "sex"])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = (~y.isna()).values
    x = x[valid].copy()
    y = y[valid].astype(int)

    imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
    x_imp = pd.DataFrame(imputer.fit_transform(x), columns=x.columns, index=x.index)

    x_train, x_test, y_train, y_test = train_test_split(
        x_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(x_train, y_train)
    model_full = grid_search.best_estimator_
    pred_prob_full = model_full.predict_proba(x_test)[:, 1]

    threshold_rows_full, best_f1_row_full, best_recall_row_full = scan_thresholds(
        y_true=y_test, pred_prob=pred_prob_full, precision_floor=0.50
    )
    full_default = make_default_metrics(y_true=y_test, pred_prob=pred_prob_full, threshold=0.50)

    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model_full.feature_importances_}
    ).sort_values("importance", ascending=False)

    pattern_fi = fi_df[fi_df["feature"].str.startswith("pattern_")].copy()
    top_k = 3
    top_pattern_features = pattern_fi.head(top_k)["feature"].tolist()
    light_features = BASE_FEATURE_COLS + top_pattern_features

    model_light = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        **grid_search.best_params_,
    )
    model_light.fit(x_train[light_features], y_train)
    pred_prob_light = model_light.predict_proba(x_test[light_features])[:, 1]

    threshold_rows_light, best_f1_row_light, best_recall_row_light = scan_thresholds(
        y_true=y_test, pred_prob=pred_prob_light, precision_floor=0.50
    )
    light_default = make_default_metrics(y_true=y_test, pred_prob=pred_prob_light, threshold=0.50)

    metrics = {
        **full_default,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "best_cv_roc_auc": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
        "threshold_default": 0.5,
        "threshold_best_f1": float(best_f1_row_full["threshold"]),
        "best_f1_at_threshold": float(best_f1_row_full["f1"]),
        "best_recall_at_threshold": float(best_f1_row_full["recall"]),
        "best_precision_at_threshold": float(best_f1_row_full["precision"]),
        "best_accuracy_at_threshold": float(best_f1_row_full["accuracy"]),
        "threshold_recall_priority": float(best_recall_row_full["threshold"]),
        "recall_priority_recall": float(best_recall_row_full["recall"]),
        "recall_priority_precision": float(best_recall_row_full["precision"]),
        "recall_priority_f1": float(best_recall_row_full["f1"]),
        "features": feature_cols,
        "smoothing_features_used": top_pattern_features,
    }

    ab_df = pd.DataFrame(
        [
            {
                "model": "full",
                "n_features": len(feature_cols),
                "features": "|".join(feature_cols),
                "default_accuracy": full_default["accuracy"],
                "default_precision": full_default["precision"],
                "default_recall": full_default["recall"],
                "default_f1": full_default["f1"],
                "roc_auc": full_default["roc_auc"],
                "best_f1_threshold": best_f1_row_full["threshold"],
                "best_f1": best_f1_row_full["f1"],
                "best_f1_recall": best_f1_row_full["recall"],
                "best_f1_precision": best_f1_row_full["precision"],
                "recall_priority_threshold": best_recall_row_full["threshold"],
                "recall_priority_recall": best_recall_row_full["recall"],
                "recall_priority_precision": best_recall_row_full["precision"],
                "recall_priority_f1": best_recall_row_full["f1"],
            },
            {
                "model": "light",
                "n_features": len(light_features),
                "features": "|".join(light_features),
                "default_accuracy": light_default["accuracy"],
                "default_precision": light_default["precision"],
                "default_recall": light_default["recall"],
                "default_f1": light_default["f1"],
                "roc_auc": light_default["roc_auc"],
                "best_f1_threshold": best_f1_row_light["threshold"],
                "best_f1": best_f1_row_light["f1"],
                "best_f1_recall": best_f1_row_light["recall"],
                "best_f1_precision": best_f1_row_light["precision"],
                "recall_priority_threshold": best_recall_row_light["threshold"],
                "recall_priority_recall": best_recall_row_light["recall"],
                "recall_priority_precision": best_recall_row_light["precision"],
                "recall_priority_f1": best_recall_row_light["f1"],
            },
        ]
    )

    joblib.dump({"model": model_full, "imputer": imputer, "features": feature_cols}, MODEL_FILE)
    joblib.dump({"model": model_light, "imputer": imputer, "features": light_features}, MODEL_FILE_LIGHT)
    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fi_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows_full).to_csv(THRESHOLD_METRICS_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows_light).to_csv(THRESHOLD_METRICS_LIGHT_FILE, index=False, encoding="utf-8-sig")
    ab_df.to_csv(AB_COMPARE_FILE, index=False, encoding="utf-8-sig")
    with open(AB_COMPARE_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(ab_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"[STEP4] Saved model (full): {MODEL_FILE}")
    print(f"[STEP4] Saved model (light): {MODEL_FILE_LIGHT}")
    print(f"[STEP4] Saved metrics: {METRIC_FILE}")
    print(f"[STEP4] Saved threshold metrics (full): {THRESHOLD_METRICS_FILE}")
    print(f"[STEP4] Saved threshold metrics (light): {THRESHOLD_METRICS_LIGHT_FILE}")
    print(f"[STEP4] Saved A/B comparison: {AB_COMPARE_FILE}")
    print(f"[STEP4] Metrics: {metrics}")
    print("[STEP4] A/B summary:")
    print(ab_df.to_string(index=False))


if __name__ == "__main__":
    main()

