from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from common import DATA_PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs, to_numeric


INPUT_FILE = DATA_PROCESSED_DIR / "metabolic_syndrome_labeled_dataset.csv"
MODEL_FILE = MODEL_DIR / "xgboost_metabolic_syndrome.joblib"
METRIC_FILE = OUTPUT_DIR / "xgboost_metrics.json"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"

FEATURE_COLS = ["nutrition_intake", "sex", "HE_ht", "HE_wt", "HE_wc", "age"]
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


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    x = df[FEATURE_COLS].copy()
    x["sex"] = encode_sex(x["sex"])
    x = to_numeric(x, [c for c in FEATURE_COLS if c != "sex"])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = (~y.isna()).values
    x = x[valid].copy()
    y = y[valid].astype(int)

    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imputer.fit_transform(x), columns=x.columns, index=x.index)

    x_train, x_test, y_train, y_test = train_test_split(
        x_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    pred_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, pred_prob)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": FEATURE_COLS,
    }

    joblib.dump({"model": model, "imputer": imputer, "features": FEATURE_COLS}, MODEL_FILE)
    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fi_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False, encoding="utf-8-sig")

    print(f"[STEP4] Saved model: {MODEL_FILE}")
    print(f"[STEP4] Saved metrics: {METRIC_FILE}")
    print(f"[STEP4] Metrics: {metrics}")


if __name__ == "__main__":
    main()
