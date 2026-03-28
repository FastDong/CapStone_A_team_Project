from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "nutrition_intake",
    "nutrition_prob_2",
    "waist_x_excess_prob",
    "sex",
    "HE_ht",
    "HE_wt",
    "HE_wc",
    "age",
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


def assign_age_group(age: pd.Series) -> pd.Series:
    bins = [0, 20, 30, 40, 50, 60, 70, 200]
    labels = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    return pd.cut(age, bins=bins, labels=labels, right=False)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_file = root / "data" / "processed" / "metabolic_syndrome_labeled_dataset.csv"
    model_file = root / "ml" / "models" / "xgboost_metabolic_syndrome.joblib"
    metrics_file = root / "ml" / "outputs" / "xgboost_metrics.json"
    out_dir = root / "ml" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, low_memory=False)
    if "nutrition_prob_2" not in df.columns:
        raise ValueError("Missing nutrition_prob_2. Run preprocess.py first.")

    df["nutrition_prob_2"] = pd.to_numeric(df["nutrition_prob_2"], errors="coerce")
    df["waist_x_excess_prob"] = pd.to_numeric(df["HE_wc"], errors="coerce") * df["nutrition_prob_2"]
    df["sex"] = encode_sex(df["sex"])

    x = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        if col != "sex":
            x[col] = pd.to_numeric(x[col], errors="coerce")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = (~y.isna()).values
    x = x[valid].copy()
    y = y[valid].astype(int)

    # Correlation analysis (feature-feature + target).
    corr_df = x.copy()
    corr_df[TARGET_COL] = y.values
    corr = corr_df.corr(numeric_only=True)
    corr.to_csv(out_dir / "feature_correlation_matrix.csv", encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    # Reproduce holdout split and evaluate by age groups.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    bundle = joblib.load(model_file)
    model = bundle["model"]
    imputer = bundle["imputer"]
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    best_threshold = float(metrics.get("threshold_best_f1", 0.5))

    x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=FEATURE_COLS, index=x_test.index)
    prob = model.predict_proba(x_test_imp)[:, 1]
    pred = (prob >= best_threshold).astype(int)

    eval_df = pd.DataFrame(
        {
            "age": pd.to_numeric(x_test["age"], errors="coerce"),
            "y_true": y_test.values,
            "y_prob": prob,
            "y_pred": pred,
        }
    )
    eval_df["age_group"] = assign_age_group(eval_df["age"])

    rows = []
    for group, gdf in eval_df.dropna(subset=["age_group"]).groupby("age_group", observed=False):
        if len(gdf) == 0:
            continue
        row = {
            "age_group": str(group),
            "rows": int(len(gdf)),
            "positive_rate": float(gdf["y_true"].mean()),
            "accuracy": float(accuracy_score(gdf["y_true"], gdf["y_pred"])),
            "precision": float(precision_score(gdf["y_true"], gdf["y_pred"], zero_division=0)),
            "recall": float(recall_score(gdf["y_true"], gdf["y_pred"], zero_division=0)),
            "f1": float(f1_score(gdf["y_true"], gdf["y_pred"], zero_division=0)),
        }
        if gdf["y_true"].nunique() > 1:
            row["roc_auc"] = float(roc_auc_score(gdf["y_true"], gdf["y_prob"]))
        else:
            row["roc_auc"] = np.nan
        rows.append(row)

    age_perf = pd.DataFrame(rows)
    age_perf.to_csv(out_dir / "age_group_performance.csv", index=False, encoding="utf-8-sig")

    plot_df = age_perf.copy()
    plot_df = plot_df.sort_values("age_group")
    plt.figure(figsize=(11, 6))
    plt.plot(plot_df["age_group"], plot_df["f1"], marker="o", label="F1")
    plt.plot(plot_df["age_group"], plot_df["recall"], marker="o", label="Recall")
    plt.plot(plot_df["age_group"], plot_df["roc_auc"], marker="o", label="ROC-AUC")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title(f"Performance by Age Group (threshold={best_threshold:.2f})")
    plt.xlabel("Age Group")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "age_group_performance.png", dpi=300)
    plt.close()

    print(f"[ANALYZE] Saved: {out_dir / 'feature_correlation_matrix.csv'}")
    print(f"[ANALYZE] Saved: {out_dir / 'feature_correlation_heatmap.png'}")
    print(f"[ANALYZE] Saved: {out_dir / 'age_group_performance.csv'}")
    print(f"[ANALYZE] Saved: {out_dir / 'age_group_performance.png'}")
    print(age_perf.to_string(index=False))


if __name__ == "__main__":
    main()
