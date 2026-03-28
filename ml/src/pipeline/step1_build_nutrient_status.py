from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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


NUTRIENT_FILE = DATA_RAW_DIR / "nutrient_2019.csv"
OUT_FILE = DATA_INTERIM_DIR / "nutrient_2019_person_level_with_status.csv"

PROFILE_COLS = ["sex", "age", "region", "incm", "ho_incm", "edu", "occp", "town_t", "apt_t"]
NUTRITION_COLS = ["NF_EN", "NF_PROT", "NF_FAT", "NF_CHO", "NF_TDF", "NF_NA"]


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def _target_kcal_by_profile(sex: pd.Series, age: pd.Series) -> np.ndarray:
    sex_num = pd.to_numeric(sex, errors="coerce")
    age_num = pd.to_numeric(age, errors="coerce").fillna(30)
    target_kcal = np.where(sex_num == 1, 2300, 1800).astype(float)
    target_kcal = np.where(age_num >= 50, target_kcal - 100, target_kcal)
    target_kcal = np.clip(target_kcal, 1500, 2800)
    return target_kcal


def _iqr(series: pd.Series) -> float:
    if series.dropna().empty:
        return np.nan
    return float(series.quantile(0.75) - series.quantile(0.25))


def build_smoothed_pattern_features(daily_sum: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_sum.merge(profile_df[["sex", "age"]], left_on="ID", right_index=True, how="left")
    daily["target_kcal"] = _target_kcal_by_profile(daily["sex"], daily["age"])
    daily["energy_ratio"] = daily["NF_EN"] / daily["target_kcal"]

    daily["excess_day"] = (daily["energy_ratio"] > 1.15).astype(float)
    daily["deficient_day"] = (daily["energy_ratio"] < 0.85).astype(float)
    daily["balanced_day"] = ((daily["energy_ratio"] >= 0.85) & (daily["energy_ratio"] <= 1.15)).astype(float)

    daily["protein_density"] = (daily["NF_PROT"] / daily["NF_EN"]) * 1000.0
    daily["fat_density"] = (daily["NF_FAT"] / daily["NF_EN"]) * 1000.0
    daily["carb_density"] = (daily["NF_CHO"] / daily["NF_EN"]) * 1000.0
    daily["sodium_density"] = (daily["NF_NA"] / daily["NF_EN"]) * 1000.0

    pattern_df = (
        daily.groupby("ID", dropna=True)
        .agg(
            pattern_energy_ratio_mean=("energy_ratio", "mean"),
            pattern_energy_ratio_std=("energy_ratio", "std"),
            pattern_energy_ratio_iqr=("energy_ratio", _iqr),
            pattern_excess_day_ratio=("excess_day", "mean"),
            pattern_deficient_day_ratio=("deficient_day", "mean"),
            pattern_balanced_day_ratio=("balanced_day", "mean"),
            pattern_protein_density_mean=("protein_density", "mean"),
            pattern_fat_density_mean=("fat_density", "mean"),
            pattern_carb_density_mean=("carb_density", "mean"),
            pattern_sodium_density_mean=("sodium_density", "mean"),
        )
        .reset_index()
    )
    return pattern_df


def build_person_level_with_daily_aggregation(file_path: Path) -> pd.DataFrame:
    cols = pd.read_csv(file_path, nrows=1).columns.tolist()
    day_col = "N_DAY" if "N_DAY" in cols else None
    year_col = "year" if "year" in cols else None

    usecols = [c for c in ["ID", day_col, year_col, *PROFILE_COLS, *NUTRITION_COLS] if c and c in cols]
    nutrition_cols = [c for c in NUTRITION_COLS if c in usecols]
    profile_cols = [c for c in PROFILE_COLS if c in usecols]

    if "ID" not in usecols or not nutrition_cols:
        raise ValueError("nutrient_2019.csv must include ID and NF_* nutrient columns.")

    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)
    df["ID"] = normalize_id_series(df["ID"])
    df = to_numeric(df, nutrition_cols + ["age", "incm", "ho_incm", "N_DAY", "year"])

    daily_group_cols = ["ID"]
    if year_col:
        daily_group_cols.append(year_col)
    if day_col:
        daily_group_cols.append(day_col)

    # 동일 ID의 다중 행(하루 식품별 기록)을 하루 단위 섭취량으로 합산.
    daily_sum = df.groupby(daily_group_cols, dropna=False)[nutrition_cols].sum(min_count=1).reset_index()

    # 개인별 대표값은 하루 합계의 중앙값으로 사용(특이한 과식/결식 일자 완화).
    person_nutrition = daily_sum.groupby("ID", dropna=True)[nutrition_cols].median()
    recorded_days = daily_sum.groupby("ID", dropna=True).size().rename("recorded_days")

    profile_df = df.groupby("ID", dropna=True)[profile_cols].agg(_first_non_null)
    pattern_df = build_smoothed_pattern_features(daily_sum, profile_df)
    person_df = profile_df.join(person_nutrition, how="inner").join(recorded_days, how="left").reset_index()
    person_df = person_df.merge(pattern_df, on="ID", how="left")

    missing_cols = [c for c in nutrition_cols if c in person_df.columns]
    person_df["missing_nutrition_rate"] = person_df[missing_cols].isna().mean(axis=1)

    pattern_cols = [c for c in person_df.columns if c.startswith("pattern_")]
    for col in pattern_cols:
        person_df[col] = pd.to_numeric(person_df[col], errors="coerce")
        if "std" in col or "iqr" in col:
            person_df[col] = person_df[col].fillna(0.0)

    return person_df


def create_initial_nutrition_class(df: pd.DataFrame) -> pd.Series:
    energy = df["NF_EN"].fillna(df["NF_EN"].median())
    target_kcal = _target_kcal_by_profile(df["sex"], df["age"])

    intake_ratio = energy / target_kcal
    status = np.select(
        [intake_ratio < 0.85, intake_ratio > 1.15],
        [0, 2],
        default=1,
    )
    return pd.Series(status, index=df.index, name="nutrition_status_bootstrap")


def main() -> None:
    ensure_dirs()
    person_df = build_person_level_with_daily_aggregation(NUTRIENT_FILE)

    required = ["NF_EN", "NF_PROT", "NF_FAT", "NF_CHO"]
    missing = [c for c in required if c not in person_df.columns]
    if missing:
        raise ValueError(f"Missing required nutrient columns in nutrient dataset: {missing}")

    person_df["nutrition_status_bootstrap"] = create_initial_nutrition_class(person_df)

    knn_feature_cols = [c for c in required + ["NF_TDF", "NF_NA"] if c in person_df.columns]
    x = person_df[knn_feature_cols].copy()
    x = x.fillna(x.median(numeric_only=True))
    y = person_df["nutrition_status_bootstrap"].astype(int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    knn = KNeighborsClassifier(n_neighbors=21, weights="distance")
    knn.fit(x_scaled, y)

    class_values = np.array([0, 1, 2], dtype=int)
    pred = knn_predict_with_softmax(knn, x_scaled, class_values=class_values, temperature=1.0)

    out_df = person_df.copy()
    out_df["nutrition_status"] = pred.labels.astype(int)
    out_df["nutrition_prob_0"] = pred.probs[:, 0]
    out_df["nutrition_prob_1"] = pred.probs[:, 1]
    out_df["nutrition_prob_2"] = pred.probs[:, 2]
    out_df["nutrition_status_text"] = out_df["nutrition_status"].map(
        {0: "deficient", 1: "balanced", 2: "excess"}
    )

    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[STEP1] Saved: {OUT_FILE}")
    print(f"[STEP1] Rows: {len(out_df):,}")
    print(f"[STEP1] Median recorded_days: {out_df['recorded_days'].median()}")
    print(f"[STEP1] Mean missing_nutrition_rate: {out_df['missing_nutrition_rate'].mean():.4f}")
    print(out_df["nutrition_status"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
