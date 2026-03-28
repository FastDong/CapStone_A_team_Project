from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from pipeline.common import normalize_id_series
import json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_file = root / "ml" / "models" / "xgboost_metabolic_syndrome.joblib"
    data_file = root / "data" / "processed" / "metabolic_syndrome_labeled_dataset.csv"
    metrics_file = root / "ml" / "outputs" / "xgboost_metrics.json"

    if not model_file.exists():
        print("No model file found. Run train.py first.")
        return
    if not data_file.exists():
        print("No processed dataset found. Run preprocess.py first.")
        return

    bundle = joblib.load(model_file)
    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["features"]

    df = pd.read_csv(data_file, low_memory=False)
    df["ID"] = normalize_id_series(df["ID"])

    if "nutrition_prob_2" in df.columns and "HE_wc" in df.columns:
        df["nutrition_prob_2"] = pd.to_numeric(df["nutrition_prob_2"], errors="coerce")
        df["waist_x_excess_prob"] = pd.to_numeric(df["HE_wc"], errors="coerce") * df["nutrition_prob_2"]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required prediction features: {missing_features}")

    x = df[features].copy()

    raw_sex = x["sex"].copy()
    sex_num = pd.to_numeric(raw_sex, errors="coerce")
    x["sex"] = sex_num
    x.loc[x["sex"] == 1, "sex"] = 1
    x.loc[x["sex"] == 2, "sex"] = 0
    remain = x["sex"].isna()
    if remain.any():
        sex = raw_sex.loc[remain].astype(str).str.strip().str.lower()
        x.loc[remain, "sex"] = sex.map(
            lambda v: 1 if v in {"1", "1.0", "m", "male"} else (0 if v in {"2", "2.0", "f", "female"} else None)
        )
    x_imp = pd.DataFrame(imputer.transform(x), columns=features, index=x.index)

    prob = model.predict_proba(x_imp)[:, 1]
    threshold = 0.5
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)
            threshold = float(m.get("threshold_best_f1", 0.5))

    out = df[["ID"]].copy()
    out["metabolic_syndrome_risk_prob"] = prob
    out["pred_label_by_best_f1_threshold"] = (out["metabolic_syndrome_risk_prob"] >= threshold).astype(int)
    out["used_threshold"] = threshold
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
