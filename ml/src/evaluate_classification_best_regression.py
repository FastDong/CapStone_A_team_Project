from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from train_classification import (
    CLS_TARGET,
    MODEL_SPECS,
    evaluate_with_threshold,
    find_best_threshold,
    get_oof_probabilities,
    _make_classifier,
)


INPUT_FILE = Path("ml/outputs/data/stage2_input_with_best_models_predictions.csv")
OUTPUT_FILE = Path("ml/outputs/metrics/classification_metrics_best_regression_models.csv")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run build_stage2_best_regression_input.py first.")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    train_df = df[df["set"] == "train"].copy()
    test_df = df[df["set"] == "test"].copy()
    train_df[CLS_TARGET] = pd.to_numeric(train_df[CLS_TARGET], errors="coerce").astype(int)
    test_df[CLS_TARGET] = pd.to_numeric(test_df[CLS_TARGET], errors="coerce").astype(int)
    y_train = train_df[CLS_TARGET]
    y_test = test_df[CLS_TARGET]

    rows: List[Dict[str, float | str]] = []

    for spec in MODEL_SPECS:
        features = spec["features"]
        x_train_raw = train_df[features].copy()
        x_test_raw = test_df[features].copy()
        for col in features:
            x_train_raw[col] = pd.to_numeric(x_train_raw[col], errors="coerce")
            x_test_raw[col] = pd.to_numeric(x_test_raw[col], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        x_train = pd.DataFrame(imputer.fit_transform(x_train_raw), columns=features, index=x_train_raw.index)
        x_test = pd.DataFrame(imputer.transform(x_test_raw), columns=features, index=x_test_raw.index)

        oof_prob = get_oof_probabilities(x_train, y_train, n_splits=5)
        best_thr, _ = find_best_threshold(y_train, oof_prob)

        model = _make_classifier(y_train)
        model.fit(x_train, y_train)
        test_prob = model.predict_proba(x_test)[:, 1]

        rows.append(
            {
                "model_name": spec["model_name"],
                "threshold_type": "default_0.5",
                **evaluate_with_threshold(y_test, test_prob, 0.5),
            }
        )
        rows.append(
            {
                "model_name": spec["model_name"],
                "threshold_type": "optimized",
                **evaluate_with_threshold(y_test, test_prob, best_thr),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
