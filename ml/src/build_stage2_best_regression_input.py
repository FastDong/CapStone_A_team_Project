from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split

from benchmark_regression_models import build_models, maybe_wrap
from train_regression import (
    ALL_FEATURES,
    CLS_TARGET,
    DATA_FILE,
    DATA_OUTPUT_DIR,
    METADATA_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    REG_TARGETS,
    add_features,
    encode_sex,
    metrics,
)


BEST_STAGE2_FILE = DATA_OUTPUT_DIR / "stage2_input_with_best_models_predictions.csv"
BEST_METRICS_FILE = METRICS_DIR / "regression_predict_metrics_best_models.csv"
BEST_SELECTION_FILE = METADATA_DIR / "regression_best_model_selection.json"
BENCHMARK_FILE = METRICS_DIR / "regression_model_benchmark.csv"


def pick_best_models() -> dict[str, str]:
    benchmark = pd.read_csv(BENCHMARK_FILE)
    selected: dict[str, str] = {}
    for target in REG_TARGETS:
        target_rows = benchmark[benchmark["target"] == target].sort_values(["RMSE", "R2"], ascending=[True, False])
        selected[target] = str(target_rows.iloc[0]["model"])
    return selected


def main() -> None:
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    for col in ALL_FEATURES + REG_TARGETS + [CLS_TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sex"] = encode_sex(df["sex"])
    df = add_features(df)
    df = df.dropna(subset=[CLS_TARGET]).copy()
    df[CLS_TARGET] = df[CLS_TARGET].astype(int)
    df["row_id"] = np.arange(len(df))

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[CLS_TARGET],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    imputer = SimpleImputer(strategy="median")
    non_sex_features = [c for c in ALL_FEATURES if c != "sex"]
    imputer.fit(train_df[non_sex_features])
    sex_mode = float(train_df["sex"].mode(dropna=True).iloc[0])

    x_train = train_df[ALL_FEATURES].copy()
    x_test = test_df[ALL_FEATURES].copy()
    x_train[non_sex_features] = imputer.transform(x_train[non_sex_features])
    x_test[non_sex_features] = imputer.transform(x_test[non_sex_features])
    x_train["sex"] = x_train["sex"].fillna(sex_mode)
    x_test["sex"] = x_test["sex"].fillna(sex_mode)

    selected_models = pick_best_models()
    base_models = build_models()
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_stage2 = train_df[["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]].copy()
    test_stage2 = test_df[["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]].copy()
    train_stage2["set"] = "train"
    test_stage2["set"] = "test"

    metric_rows: list[dict[str, float | str]] = []

    for target in REG_TARGETS:
        model_name = selected_models[target]
        y_train = pd.to_numeric(train_df[target], errors="coerce")
        y_test = pd.to_numeric(test_df[target], errors="coerce")
        train_mask = y_train.notna()
        test_mask = y_test.notna()

        x_train_t = x_train.loc[train_mask].copy()
        x_test_t = x_test.loc[test_mask].copy()
        y_train_t = y_train.loc[train_mask].copy()
        y_test_t = y_test.loc[test_mask].copy()

        oof_pred = np.zeros(len(x_train_t), dtype=float)
        test_fold_preds: list[np.ndarray] = []

        for tr_idx, val_idx in kf.split(x_train_t):
            estimator = maybe_wrap(model_name, build_models()[model_name])
            estimator.fit(x_train_t.iloc[tr_idx], y_train_t.iloc[tr_idx])
            oof_pred[val_idx] = estimator.predict(x_train_t.iloc[val_idx])
            test_fold_preds.append(estimator.predict(x_test_t))

        test_pred = np.mean(np.vstack(test_fold_preds), axis=0)
        final_model = maybe_wrap(model_name, base_models[model_name])
        final_model.fit(x_train_t, y_train_t)

        train_stage2.loc[train_mask.index[train_mask], f"pred_{target.replace('HE_', '')}"] = oof_pred
        test_stage2.loc[test_mask.index[test_mask], f"pred_{target.replace('HE_', '')}"] = test_pred

        metric_rows.append(
            {
                "target": target,
                "selected_model": model_name,
                **metrics(y_test_t, test_pred),
            }
        )

    stage2 = pd.concat([train_stage2, test_stage2], ignore_index=True)
    stage2.to_csv(BEST_STAGE2_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(BEST_METRICS_FILE, index=False, encoding="utf-8-sig")
    BEST_SELECTION_FILE.write_text(json.dumps(selected_models, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {BEST_STAGE2_FILE}")
    print(f"Saved: {BEST_METRICS_FILE}")
    print(f"Saved: {BEST_SELECTION_FILE}")


if __name__ == "__main__":
    main()
