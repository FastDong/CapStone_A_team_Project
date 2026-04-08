from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


RANDOM_STATE = 42

DATA_FILE = Path("data/processed/final_dataset_with_label.csv")
OUTPUT_DIR = Path("ml/outputs")
MODEL_DIR = Path("ml/models")

BASE_FEATURES = [
    "age",
    "sex",
    "HE_ht",
    "HE_wt",
    "HE_wc",
    "WHtR",
    "NF_EN",
    "NF_PROT",
    "NF_FAT",
    "NF_CHO",
    "NF_SUGAR",
    "NF_TDF",
    "NF_CHOL",
    "NF_NA",
]
REG_TARGETS = ["HE_glu", "HE_sbp", "HE_chol"]
CLS_TARGET = "MetS_Label"

NUMERIC_COLS = [
    "age",
    "HE_ht",
    "HE_wt",
    "HE_wc",
    "WHtR",
    "NF_EN",
    "NF_PROT",
    "NF_FAT",
    "NF_CHO",
    "NF_SUGAR",
    "NF_TDF",
    "NF_CHOL",
    "NF_NA",
    "HE_glu",
    "HE_sbp",
    "HE_chol",
    "HE_TG",
    "HE_HDL_st2",
    "MetS_Criteria_Count",
    "MetS_Label",
]


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _encode_sex(series: pd.Series) -> pd.Series:
    """
    Encode sex to binary:
    - male: 1
    - female: 0
    """
    num = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[num == 1] = 1.0
    out.loc[num == 2] = 0.0

    remain = out.isna()
    if remain.any():
        s = series.astype(str).str.strip().str.lower()
        male_tokens = {"1", "1.0", "m", "male"}
        female_tokens = {"2", "2.0", "f", "female"}
        out.loc[remain] = s.loc[remain].map(
            lambda v: 1.0 if v in male_tokens else (0.0 if v in female_tokens else np.nan)
        )
    return out


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["ID"] + BASE_FEATURES + REG_TARGETS + [CLS_TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["sex"] = _encode_sex(df["sex"])
    df = df.dropna(subset=[CLS_TARGET]).copy()
    df[CLS_TARGET] = df[CLS_TARGET].astype(int)
    df["row_id"] = np.arange(len(df))
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[CLS_TARGET],
    )
    return train_df.copy(), test_df.copy()


def fit_feature_imputers(train_df: pd.DataFrame) -> Tuple[SimpleImputer, float]:
    # Impute numeric features with median and sex with mode.
    num_features = [c for c in BASE_FEATURES if c != "sex"]
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(train_df[num_features])

    sex_mode = float(train_df["sex"].mode(dropna=True).iloc[0])
    return num_imputer, sex_mode


def transform_base_features(df: pd.DataFrame, num_imputer: SimpleImputer, sex_mode: float) -> pd.DataFrame:
    num_features = [c for c in BASE_FEATURES if c != "sex"]
    out = df[BASE_FEATURES].copy()
    out[num_features] = num_imputer.transform(out[num_features])
    out["sex"] = out["sex"].fillna(sex_mode)
    return out


def train_single_regressor_oof(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, XGBRegressor]:
    """
    Generate OOF predictions on train and holdout predictions on test.
    Leakage prevention:
    - Each training sample prediction is made by a model that did not train on that sample.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(x_train), dtype=float)
    test_fold_preds: List[np.ndarray] = []

    for tr_idx, val_idx in kf.split(x_train):
        x_tr, x_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        model_fold = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model_fold.fit(x_tr, y_tr)
        oof_pred[val_idx] = model_fold.predict(x_val)
        test_fold_preds.append(model_fold.predict(x_test))

    test_pred = np.mean(np.vstack(test_fold_preds), axis=0)

    final_model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(x_train, y_train)
    return oof_pred, test_pred, final_model


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _print_section("1) Load Data")
    df = load_dataset()
    print(f"Data shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    _print_section("2) Preprocess")
    df = preprocess_data(df)
    missing_summary = df.isna().sum().sort_values(ascending=False)
    print("Missing values summary (top 30):")
    print(missing_summary.head(30).to_string())

    _print_section("3) Train/Test Split")
    train_df, test_df = split_data(df)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print("MetS_Label distribution:")
    print("Train:", train_df[CLS_TARGET].value_counts(normalize=True).sort_index().to_dict())
    print("Test :", test_df[CLS_TARGET].value_counts(normalize=True).sort_index().to_dict())

    _print_section("4) Feature Imputation (train-only fit)")
    num_imputer, sex_mode = fit_feature_imputers(train_df)
    x_train = transform_base_features(train_df, num_imputer, sex_mode)
    x_test = transform_base_features(test_df, num_imputer, sex_mode)
    print(f"Feature matrix train shape: {x_train.shape}")
    print(f"Feature matrix test shape:  {x_test.shape}")

    _print_section("5) Stage-1 Regression (OOF)")
    reg_metrics_rows = []
    fi_rows = []

    stage_pred_train = pd.DataFrame({"row_id": train_df["row_id"].values, "set": "train"})
    stage_pred_test = pd.DataFrame({"row_id": test_df["row_id"].values, "set": "test"})

    for target in REG_TARGETS:
        y_train = train_df[target].copy()
        y_test = test_df[target].copy()

        train_valid_mask = y_train.notna()
        test_valid_mask = y_test.notna()

        oof_pred = np.full(len(train_df), np.nan, dtype=float)
        test_pred = np.full(len(test_df), np.nan, dtype=float)

        # Train only where regression target exists.
        x_train_valid = x_train.loc[train_valid_mask].reset_index(drop=True)
        y_train_valid = y_train.loc[train_valid_mask].reset_index(drop=True)
        x_test_valid = x_test.loc[test_valid_mask].reset_index(drop=True)

        pred_oof_valid, pred_test_valid, model = train_single_regressor_oof(
            x_train=x_train_valid,
            y_train=y_train_valid,
            x_test=x_test_valid if len(x_test_valid) > 0 else x_test.iloc[:0],
        )

        oof_pred[np.where(train_valid_mask.values)[0]] = pred_oof_valid
        if len(x_test_valid) > 0:
            test_pred[np.where(test_valid_mask.values)[0]] = pred_test_valid

        # For rows without observed regression target in test, use final model prediction.
        if (~test_valid_mask).any():
            test_pred[np.where((~test_valid_mask).values)[0]] = model.predict(
                x_test.loc[~test_valid_mask].reset_index(drop=True)
            )

        metrics = evaluate_regression(y_true=y_test.loc[test_valid_mask], y_pred=test_pred[test_valid_mask.values])
        reg_metrics_rows.append({"target": target, **metrics})

        for f_name, imp in zip(BASE_FEATURES, model.feature_importances_):
            fi_rows.append({"target": target, "feature": f_name, "importance": float(imp)})

        pred_name = f"pred_{target.replace('HE_', '')}"
        stage_pred_train[pred_name] = oof_pred
        stage_pred_test[pred_name] = test_pred

        model.save_model(str(MODEL_DIR / f"regressor_{target}.json"))
        print(
            f"{target} -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}"
        )

    _print_section("6) Save Regression Artifacts")
    reg_metrics_df = pd.DataFrame(reg_metrics_rows)
    reg_metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False, encoding="utf-8-sig")

    fi_reg_df = pd.DataFrame(fi_rows).sort_values(["target", "importance"], ascending=[True, False])
    fi_reg_df.to_csv(OUTPUT_DIR / "feature_importance_regression.csv", index=False, encoding="utf-8-sig")

    # Build stage-2 ready dataset for classification script.
    train_export = train_df[
        ["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]
    ].merge(stage_pred_train, on="row_id", how="left")
    test_export = test_df[
        ["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]
    ].merge(stage_pred_test, on="row_id", how="left")
    stage2_df = pd.concat([train_export, test_export], axis=0, ignore_index=True)
    stage2_df.to_csv(OUTPUT_DIR / "stage2_input_with_predictions.csv", index=False, encoding="utf-8-sig")

    # Persist split/imputer metadata for reproducibility.
    split_meta = {
        "random_state": RANDOM_STATE,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "base_features": BASE_FEATURES,
        "regression_targets": REG_TARGETS,
    }
    pd.Series(split_meta).to_json(OUTPUT_DIR / "split_meta.json", force_ascii=False, indent=2)
    pd.DataFrame({"num_feature": [c for c in BASE_FEATURES if c != "sex"], "median": num_imputer.statistics_}).to_csv(
        OUTPUT_DIR / "feature_medians_train.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"sex_mode": [sex_mode]}).to_csv(OUTPUT_DIR / "sex_mode_train.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_DIR / 'regression_metrics.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'feature_importance_regression.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'stage2_input_with_predictions.csv'}")


if __name__ == "__main__":
    main()
