from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import (
    DATA_INTERIM_DIR,
    DATA_RAW_DIR,
    ensure_dirs,
    knn_predict_with_softmax,
    normalize_id_series,
    to_numeric,
)


NUTRIENT_PERSON_FILE = DATA_INTERIM_DIR / "nutrient_2019_person_level_with_status.csv"
NHANES_FILE = DATA_RAW_DIR / "NHANES_2017_2023.csv"
OUT_FILE = DATA_INTERIM_DIR / "nhanes_with_nutrition_intake_label.csv"

PROFILE_BASE_COLS = ["sex", "age", "region", "incm", "ho_incm", "edu", "occp", "town_t", "apt_t", "year"]


def encode_mixed_features(train_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train_df.copy()
    x_target = target_df.copy()

    for col in x_train.columns:
        if pd.api.types.is_numeric_dtype(x_train[col]):
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
            x_target[col] = pd.to_numeric(x_target[col], errors="coerce")
        else:
            union_values = pd.concat(
                [x_train[col].astype(str).replace("nan", np.nan), x_target[col].astype(str).replace("nan", np.nan)]
            ).dropna()
            categories = pd.Index(union_values.unique())
            mapping = {v: i for i, v in enumerate(categories)}
            x_train[col] = x_train[col].astype(str).replace("nan", np.nan).map(mapping)
            x_target[col] = x_target[col].astype(str).replace("nan", np.nan).map(mapping)

    return x_train, x_target


def main() -> None:
    ensure_dirs()
    nutrient_df = pd.read_csv(NUTRIENT_PERSON_FILE, low_memory=False)
    nutrient_df["ID"] = normalize_id_series(nutrient_df["ID"])
    nutrient_df = to_numeric(nutrient_df, ["age", "incm", "ho_incm", "year"])

    nhanes_cols = pd.read_csv(NHANES_FILE, nrows=1).columns.tolist()
    nhanes_usecols = [c for c in ["ID", *PROFILE_BASE_COLS] if c in nhanes_cols]
    nhanes_df = pd.read_csv(NHANES_FILE, usecols=nhanes_usecols, low_memory=False)
    nhanes_df["ID"] = normalize_id_series(nhanes_df["ID"])
    nhanes_df = to_numeric(nhanes_df, ["age", "incm", "ho_incm", "year"])

    profile_cols = [c for c in PROFILE_BASE_COLS if c in nutrient_df.columns and c in nhanes_df.columns]
    if len(profile_cols) < 2:
        raise ValueError("Too few common profile columns between nutrient and NHANES for transfer labeling.")

    train_df = nutrient_df.dropna(subset=["nutrition_status"]).copy()
    x_train_raw = train_df[profile_cols].copy()
    y_train = train_df["nutrition_status"].astype(int).values
    x_target_raw = nhanes_df[profile_cols].copy()

    x_train, x_target = encode_mixed_features(x_train_raw, x_target_raw)

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_target_imp = imputer.transform(x_target)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_target_scaled = scaler.transform(x_target_imp)

    knn = KNeighborsClassifier(n_neighbors=31, weights="distance")
    knn.fit(x_train_scaled, y_train)

    class_values = np.array([0, 1, 2], dtype=int)
    pred = knn_predict_with_softmax(knn, x_target_scaled, class_values=class_values, temperature=1.0)

    out_df = nhanes_df.copy()
    out_df["nutrition_intake"] = pred.labels.astype(int)
    out_df["nutrition_intake_text"] = out_df["nutrition_intake"].map(
        {0: "deficient", 1: "balanced", 2: "excess"}
    )
    out_df["nutrition_prob_0"] = pred.probs[:, 0]
    out_df["nutrition_prob_1"] = pred.probs[:, 1]
    out_df["nutrition_prob_2"] = pred.probs[:, 2]

    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[STEP2] Saved: {OUT_FILE}")
    print(f"[STEP2] Rows: {len(out_df):,}")
    print(out_df["nutrition_intake"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
