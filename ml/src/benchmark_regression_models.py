from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train_regression import (
    ALL_FEATURES,
    CLS_TARGET,
    DATA_FILE,
    METRICS_DIR,
    RANDOM_STATE,
    REG_TARGETS,
    add_features,
    encode_sex,
)


OUTPUT_FILE = METRICS_DIR / "regression_model_benchmark.csv"


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_models() -> dict:
    return {
        "xgboost_ref": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=3.0)),
            ]
        ),
    }


def maybe_wrap(model_name: str, model):
    if model_name in {"xgboost_ref", "lightgbm"}:
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    for col in ALL_FEATURES + REG_TARGETS + [CLS_TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sex"] = encode_sex(df["sex"])
    df = add_features(df)
    df = df.dropna(subset=[CLS_TARGET]).copy()
    df[CLS_TARGET] = df[CLS_TARGET].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[CLS_TARGET],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(train_df[ALL_FEATURES]), columns=ALL_FEATURES, index=train_df.index)
    x_test = pd.DataFrame(imputer.transform(test_df[ALL_FEATURES]), columns=ALL_FEATURES, index=test_df.index)

    rows = []
    for target in REG_TARGETS:
        y_train = pd.to_numeric(train_df[target], errors="coerce")
        y_test = pd.to_numeric(test_df[target], errors="coerce")
        train_mask = y_train.notna()
        test_mask = y_test.notna()
        x_train_t = x_train.loc[train_mask]
        x_test_t = x_test.loc[test_mask]
        y_train_t = y_train.loc[train_mask]
        y_test_t = y_test.loc[test_mask]

        print(f"\n[{target}]")
        for model_name, model in build_models().items():
            estimator = maybe_wrap(model_name, model)
            estimator.fit(x_train_t, y_train_t)
            pred = estimator.predict(x_test_t)
            m = evaluate(y_test_t, pred)
            rows.append({"target": target, "model": model_name, **m})
            print(f"- {model_name}: RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    out = pd.DataFrame(rows).sort_values(["target", "RMSE", "R2"], ascending=[True, True, False])
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
