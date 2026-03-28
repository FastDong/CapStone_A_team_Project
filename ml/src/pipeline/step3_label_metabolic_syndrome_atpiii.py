from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_INTERIM_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, ensure_dirs, normalize_id_series, to_numeric


NHANES_FILE = DATA_RAW_DIR / "NHANES_2017_2023.csv"
NUTRITION_LABEL_FILE = DATA_INTERIM_DIR / "nhanes_with_nutrition_intake_label.csv"
OUT_FILE = DATA_PROCESSED_DIR / "metabolic_syndrome_labeled_dataset.csv"

# ATP III 기본 cut-off (연구 목적 참고용)
WAIST_MALE_CUTOFF = 90.0
WAIST_FEMALE_CUTOFF = 80.0
TG_CUTOFF = 150.0
HDL_MALE_CUTOFF = 40.0
HDL_FEMALE_CUTOFF = 50.0
SBP_CUTOFF = 130.0
DBP_CUTOFF = 85.0
GLUCOSE_CUTOFF = 100.0


def is_male(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    male_tokens = {"1", "1.0", "m", "male"}
    return s.isin(male_tokens)


def main() -> None:
    ensure_dirs()
    nutrition_df = pd.read_csv(NUTRITION_LABEL_FILE, low_memory=False)
    nutrition_df["ID"] = normalize_id_series(nutrition_df["ID"])

    need_cols = [
        "ID",
        "sex",
        "age",
        "HE_ht",
        "HE_wt",
        "HE_wc",
        "HE_TG",
        "HE_HDL_st2",
        "HE_sbp",
        "HE_dbp",
        "HE_glu",
    ]
    nhanes_cols = pd.read_csv(NHANES_FILE, nrows=1).columns.tolist()
    usecols = [c for c in need_cols if c in nhanes_cols]
    missing = [c for c in need_cols if c not in usecols]
    if missing:
        raise ValueError(f"Missing required ATP III columns in NHANES: {missing}")

    nhanes_df = pd.read_csv(NHANES_FILE, usecols=usecols, low_memory=False)
    nhanes_df["ID"] = normalize_id_series(nhanes_df["ID"])
    merged = nutrition_df.merge(nhanes_df, on="ID", how="inner", suffixes=("", "_nh"))

    numeric_cols = ["age", "HE_ht", "HE_wt", "HE_wc", "HE_TG", "HE_HDL_st2", "HE_sbp", "HE_dbp", "HE_glu"]
    merged = to_numeric(merged, numeric_cols)

    male_mask = is_male(merged["sex"])

    c_waist = np.where(male_mask, merged["HE_wc"] >= WAIST_MALE_CUTOFF, merged["HE_wc"] >= WAIST_FEMALE_CUTOFF)
    c_tg = merged["HE_TG"] >= TG_CUTOFF
    c_hdl = np.where(male_mask, merged["HE_HDL_st2"] < HDL_MALE_CUTOFF, merged["HE_HDL_st2"] < HDL_FEMALE_CUTOFF)
    c_bp = (merged["HE_sbp"] >= SBP_CUTOFF) | (merged["HE_dbp"] >= DBP_CUTOFF)
    c_glu = merged["HE_glu"] >= GLUCOSE_CUTOFF

    criteria_count = (
        c_waist.astype("float")
        + c_tg.astype("float")
        + c_hdl.astype("float")
        + c_bp.astype("float")
        + c_glu.astype("float")
    )
    merged["metabolic_criteria_count"] = criteria_count
    merged["metabolic_syndrome"] = (criteria_count >= 3).astype(int)

    keep_cols = [
        "ID",
        "nutrition_intake",
        "nutrition_intake_text",
        "nutrition_prob_0",
        "nutrition_prob_1",
        "nutrition_prob_2",
        "sex",
        "age",
        "HE_ht",
        "HE_wt",
        "HE_wc",
        "HE_TG",
        "HE_HDL_st2",
        "HE_sbp",
        "HE_dbp",
        "HE_glu",
        "metabolic_criteria_count",
        "metabolic_syndrome",
    ]
    out_df = merged[keep_cols].copy()
    out_df = out_df.dropna(
        subset=["nutrition_intake", "sex", "age", "HE_ht", "HE_wt", "HE_wc", "metabolic_syndrome"]
    )
    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[STEP3] Saved: {OUT_FILE}")
    print(f"[STEP3] Rows: {len(out_df):,}")
    print(out_df["metabolic_syndrome"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
