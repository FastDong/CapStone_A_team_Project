from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
METRICS_DIR = OUTPUT_DIR / "metrics"
REG_PLOTS_DIR = OUTPUT_DIR / "plots" / "regression"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
METADATA_DIR = OUTPUT_DIR / "metadata"

ORIGINAL_FEATURES = [
    "age", "sex", "HE_ht", "HE_wt", "HE_wc", "WHtR",
    "NF_EN", "NF_PROT", "NF_FAT", "NF_CHO", "NF_SUGAR", "NF_TDF", "NF_CHOL", "NF_NA",
]
ENGINEERED_FEATURES = [
    "BMI", "age_sq", "wt_ht_ratio", "wc_ht_interaction", "age_x_whtR",
    "protein_per_kg", "sodium_per_kcal", "sugar_carb_ratio",
    "protein_energy_ratio", "fat_energy_ratio", "carb_energy_ratio",
]
ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES
REG_TARGETS = ["HE_glu", "HE_sbp", "HE_chol"]
CLS_TARGET = "MetS_Label"

SEARCH_SPACE = {
    "HE_glu": {"transforms": ["identity", "log1p"], "objs": ["reg:squarederror", "reg:pseudohubererror"]},
    "HE_sbp": {"transforms": ["identity"], "objs": ["reg:squarederror", "reg:pseudohubererror"]},
    "HE_chol": {"transforms": ["identity", "log1p"], "objs": ["reg:squarederror", "reg:pseudohubererror"]},
}
PARAM_GRID = {
    "HE_glu": [
        {"name": "glu_a", "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3, "reg_lambda": 1.0},
        {"name": "glu_b", "n_estimators": 450, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_lambda": 1.5},
        {"name": "glu_c", "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_lambda": 2.0},
    ],
    "HE_sbp": [
        {"name": "sbp_a", "n_estimators": 250, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 4, "reg_lambda": 1.0},
        {"name": "sbp_b", "n_estimators": 350, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 3, "reg_lambda": 1.5},
        {"name": "sbp_c", "n_estimators": 250, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_lambda": 2.0},
    ],
    "HE_chol": [
        {"name": "chol_a", "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3, "reg_lambda": 1.0},
        {"name": "chol_b", "n_estimators": 450, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_lambda": 1.5},
        {"name": "chol_c", "n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_lambda": 2.0},
    ],
}


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def encode_sex(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[num == 1] = 1.0
    out.loc[num == 2] = 0.0
    remain = out.isna()
    if remain.any():
        s = series.astype(str).str.strip().str.lower()
        out.loc[remain] = s.loc[remain].map(lambda v: 1.0 if v in {"1", "1.0", "m", "male"} else (0.0 if v in {"2", "2.0", "f", "female"} else np.nan))
    return out


def safe_div(num: pd.Series, den: pd.Series | float) -> pd.Series:
    den_s = den if isinstance(den, pd.Series) else pd.Series(float(den), index=num.index)
    return num / den_s.replace(0, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h_m = safe_div(out["HE_ht"], 100.0)
    out["BMI"] = safe_div(out["HE_wt"], h_m**2)
    out["age_sq"] = out["age"] ** 2
    out["wt_ht_ratio"] = safe_div(out["HE_wt"], out["HE_ht"])
    out["wc_ht_interaction"] = out["HE_wc"] * out["WHtR"]
    out["age_x_whtR"] = out["age"] * out["WHtR"]
    out["protein_per_kg"] = safe_div(out["NF_PROT"], out["HE_wt"])
    out["sodium_per_kcal"] = safe_div(out["NF_NA"], out["NF_EN"] + 1.0)
    out["sugar_carb_ratio"] = safe_div(out["NF_SUGAR"], out["NF_CHO"] + 1.0)
    out["protein_energy_ratio"] = safe_div(out["NF_PROT"] * 4.0, out["NF_EN"] + 1.0)
    out["fat_energy_ratio"] = safe_div(out["NF_FAT"] * 9.0, out["NF_EN"] + 1.0)
    out["carb_energy_ratio"] = safe_div(out["NF_CHO"] * 4.0, out["NF_EN"] + 1.0)
    return out


def transform_y(y: pd.Series, name: str) -> pd.Series:
    return np.log1p(np.clip(y.astype(float), 0.0, None)) if name == "log1p" else y.astype(float)


def inverse_y(pred: np.ndarray, name: str) -> np.ndarray:
    if name == "log1p":
        return np.expm1(np.clip(pred, -5.0, 8.0))
    return pred


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_model(obj: str, params: dict) -> XGBRegressor:
    use = {k: v for k, v in params.items() if k != "name"}
    return XGBRegressor(objective=obj, random_state=RANDOM_STATE, n_jobs=-1, **use)


def run_cv(x_tr: pd.DataFrame, y_tr_raw: pd.Series, x_te: pd.DataFrame, sex_tr: pd.Series, sex_te: pd.Series, obj: str, params: dict, y_tf: str, group_mode: str) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_tr = transform_y(y_tr_raw, y_tf)
    oof = np.zeros(len(x_tr), dtype=float)
    te_preds = []
    for tr_idx, va_idx in kf.split(x_tr):
        x_fold_tr, x_fold_va = x_tr.iloc[tr_idx], x_tr.iloc[va_idx]
        y_fold_tr = y_tr.iloc[tr_idx]
        s_fold_tr, s_fold_va = sex_tr.iloc[tr_idx], sex_tr.iloc[va_idx]
        fallback = make_model(obj, params)
        fallback.fit(x_fold_tr, y_fold_tr)
        va_pred = inverse_y(fallback.predict(x_fold_va), y_tf)
        te_pred = inverse_y(fallback.predict(x_te), y_tf)
        if group_mode == "by_sex":
            for sex_val in sorted(s_fold_tr.dropna().unique()):
                mask_tr = s_fold_tr == sex_val
                if int(mask_tr.sum()) < 25:
                    continue
                model = make_model(obj, params)
                model.fit(x_fold_tr.loc[mask_tr], y_fold_tr.loc[mask_tr])
                mask_va = s_fold_va == sex_val
                if mask_va.any():
                    va_pred[mask_va.values] = inverse_y(model.predict(x_fold_va.loc[mask_va]), y_tf)
                mask_te = sex_te == sex_val
                if mask_te.any():
                    te_pred[mask_te.values] = inverse_y(model.predict(x_te.loc[mask_te]), y_tf)
        oof[va_idx] = va_pred
        te_preds.append(te_pred)
    return oof, np.mean(np.vstack(te_preds), axis=0)


def fit_bundle(x_tr: pd.DataFrame, y_tr_raw: pd.Series, sex_tr: pd.Series, obj: str, params: dict, y_tf: str, group_mode: str) -> dict:
    y_tr = transform_y(y_tr_raw, y_tf)
    bundle = {"group_mode": group_mode, "transform": y_tf}
    fallback = make_model(obj, params)
    fallback.fit(x_tr, y_tr)
    bundle["fallback"] = fallback
    models = {}
    if group_mode == "by_sex":
        for sex_val in sorted(sex_tr.dropna().unique()):
            mask = sex_tr == sex_val
            if int(mask.sum()) < 25:
                continue
            model = make_model(obj, params)
            model.fit(x_tr.loc[mask], y_tr.loc[mask])
            models[str(int(sex_val))] = model
    bundle["models"] = models
    bundle["params"] = {k: v for k, v in params.items() if k != "name"}
    bundle["objective"] = obj
    bundle["param_name"] = params["name"]
    return bundle


def predict_bundle(bundle: dict, x_te: pd.DataFrame, sex_te: pd.Series) -> np.ndarray:
    pred = inverse_y(bundle["fallback"].predict(x_te), bundle["transform"])
    for sex_key, model in bundle["models"].items():
        mask = sex_te == float(sex_key)
        if mask.any():
            pred[mask.values] = inverse_y(model.predict(x_te.loc[mask]), bundle["transform"])
    return pred


def feature_importance_rows(target: str, bundle: dict, sex_tr: pd.Series) -> list[dict]:
    if not bundle["models"]:
        vals = bundle["fallback"].feature_importances_
    else:
        weights, arrs = [], []
        for sex_key, model in bundle["models"].items():
            weights.append(int((sex_tr == float(sex_key)).sum()))
            arrs.append(model.feature_importances_)
        vals = np.average(np.vstack(arrs), axis=0, weights=np.asarray(weights, dtype=float))
    return [{"target": target, "feature": f, "importance": float(v)} for f, v in zip(ALL_FEATURES, vals)]


def save_bundle_models(target: str, bundle: dict) -> None:
    bundle["fallback"].save_model(str(MODEL_DIR / f"regressor_{target}_fallback.json"))
    if not bundle["models"]:
        bundle["fallback"].save_model(str(MODEL_DIR / f"regressor_{target}.json"))
    for sex_key, model in bundle["models"].items():
        model.save_model(str(MODEL_DIR / f"regressor_{target}_sex_{sex_key}.json"))


def plot_scatter(y_true: pd.Series, y_pred: np.ndarray, target: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18)
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_title(f"Actual vs Predicted ({target}, {suffix})")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REG_PLOTS_DIR / f"actual_vs_predicted_{target}.png", dpi=300)
    plt.close(fig)


def plot_residual(y_true: pd.Series, y_pred: np.ndarray, target: str, suffix: str) -> None:
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, resid, alpha=0.35, s=18)
    ax.axhline(0, linestyle="--", color="red")
    ax.set_title(f"Residual Plot ({target}, {suffix})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual - Predicted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REG_PLOTS_DIR / f"residual_plot_{target}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    for path in [OUTPUT_DIR, MODEL_DIR, METRICS_DIR, REG_PLOTS_DIR, DATA_OUTPUT_DIR, METADATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    print_section("1) Load Data")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    for col in ORIGINAL_FEATURES + REG_TARGETS + ["HE_TG", "HE_HDL_st2", "MetS_Criteria_Count", CLS_TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sex"] = encode_sex(df["sex"])
    df = add_features(df)
    df = df.dropna(subset=[CLS_TARGET]).copy()
    df[CLS_TARGET] = df[CLS_TARGET].astype(int)
    df["row_id"] = np.arange(len(df))
    print(f"Data shape: {df.shape}")
    print("Engineered features:", ENGINEERED_FEATURES)

    print_section("2) Split + Impute")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[CLS_TARGET])
    train_df, test_df = train_df.copy(), test_df.copy()
    imp = SimpleImputer(strategy="median")
    imp.fit(train_df[[c for c in ALL_FEATURES if c != "sex"]])
    sex_mode = float(train_df["sex"].mode(dropna=True).iloc[0])
    x_train = train_df[ALL_FEATURES].copy()
    x_test = test_df[ALL_FEATURES].copy()
    x_train[[c for c in ALL_FEATURES if c != "sex"]] = imp.transform(x_train[[c for c in ALL_FEATURES if c != "sex"]])
    x_test[[c for c in ALL_FEATURES if c != "sex"]] = imp.transform(x_test[[c for c in ALL_FEATURES if c != "sex"]])
    x_train["sex"] = x_train["sex"].fillna(sex_mode)
    x_test["sex"] = x_test["sex"].fillna(sex_mode)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print_section("3) Regression Search")
    baseline_params = {"name": "baseline", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9}
    best_rows, compare_rows, exp_rows, pred_rows, fi_rows = [], [], [], [], []
    selected = {}
    stage_pred_train = pd.DataFrame({"row_id": train_df["row_id"].values, "set": "train"})
    stage_pred_test = pd.DataFrame({"row_id": test_df["row_id"].values, "set": "test"})

    for target in REG_TARGETS:
        y_tr = train_df[target].reset_index(drop=True)
        y_te = test_df[target].reset_index(drop=True)
        x_tr_all = x_train.reset_index(drop=True)
        x_te_all = x_test.reset_index(drop=True)
        sex_tr = x_tr_all["sex"].copy()
        sex_te = x_te_all["sex"].copy()

        base_oof, base_te = run_cv(x_tr_all[ORIGINAL_FEATURES], y_tr, x_te_all[ORIGINAL_FEATURES], sex_tr, sex_te, "reg:squarederror", baseline_params, "identity", "global")
        base_m = metrics(y_te, base_te)
        print(f"{target} baseline -> RMSE {base_m['RMSE']:.4f}, R2 {base_m['R2']:.4f}")

        best = None
        for y_tf in SEARCH_SPACE[target]["transforms"]:
            for obj in SEARCH_SPACE[target]["objs"]:
                for group_mode in ["global", "by_sex"]:
                    for params in PARAM_GRID[target]:
                        oof, te_pred = run_cv(x_tr_all, y_tr, x_te_all, sex_tr, sex_te, obj, params, y_tf, group_mode)
                        cv_m = metrics(y_tr, oof)
                        te_m = metrics(y_te, te_pred)
                        exp_rows.append({"target": target, "group_mode": group_mode, "transform": y_tf, "objective": obj, "param_set": params["name"], "cv_MAE": cv_m["MAE"], "cv_RMSE": cv_m["RMSE"], "cv_R2": cv_m["R2"], "test_MAE": te_m["MAE"], "test_RMSE": te_m["RMSE"], "test_R2": te_m["R2"]})
                        if best is None or cv_m["RMSE"] < best["cv"]["RMSE"]:
                            best = {"group_mode": group_mode, "transform": y_tf, "objective": obj, "params": params, "cv": cv_m, "test": te_m}

        best_oof, best_te_valid = run_cv(x_tr_all, y_tr, x_te_all, sex_tr, sex_te, best["objective"], best["params"], best["transform"], best["group_mode"])
        bundle = fit_bundle(x_tr_all, y_tr, sex_tr, best["objective"], best["params"], best["transform"], best["group_mode"])
        save_bundle_models(target, bundle)
        fi_rows.extend(feature_importance_rows(target, bundle, sex_tr))
        suffix = f"{best['group_mode']}, {best['transform']}, {best['objective']}"
        plot_scatter(y_te, best_te_valid, target, suffix)
        plot_residual(y_te, best_te_valid, target, suffix)
        pred_name = f"pred_{target.replace('HE_', '')}"
        stage_pred_train[pred_name] = best_oof
        stage_pred_test[pred_name] = best_te_valid
        best_rows.append({"target": target, **best["test"]})
        compare_rows.append({"target": target, "scenario": "baseline", **base_m})
        compare_rows.append({"target": target, "scenario": "improved", **best["test"]})
        pred_rows.append(pd.DataFrame({"row_id": test_df["row_id"].values, "ID": test_df["ID"].values, "target": target, "model_variant": "baseline", "actual": y_te.values, "predicted": base_te, "residual": y_te.values - base_te}))
        pred_rows.append(pd.DataFrame({"row_id": test_df["row_id"].values, "ID": test_df["ID"].values, "target": target, "model_variant": "improved", "actual": y_te.values, "predicted": best_te_valid, "residual": y_te.values - best_te_valid}))
        selected[target] = {
            "group_mode": best["group_mode"], "transform": best["transform"], "objective": best["objective"], "param_set": best["params"]["name"],
            "params": {k: v for k, v in best["params"].items() if k != "name"},
            "baseline_test_metrics": base_m, "improved_test_metrics": best["test"],
            "improvement": {"MAE_delta": best["test"]["MAE"] - base_m["MAE"], "RMSE_delta": best["test"]["RMSE"] - base_m["RMSE"], "R2_delta": best["test"]["R2"] - base_m["R2"]},
        }
        print(f"{target} improved -> RMSE {best['test']['RMSE']:.4f}, R2 {best['test']['R2']:.4f}, config {selected[target]['param_set']} / {best['group_mode']} / {best['transform']} / {best['objective']}")

    print_section("4) Save Artifacts")
    pd.DataFrame(best_rows).to_csv(METRICS_DIR / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(compare_rows).to_csv(METRICS_DIR / "regression_baseline_vs_improved.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exp_rows).sort_values(["target", "cv_RMSE"]).to_csv(METRICS_DIR / "regression_experiment_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fi_rows).sort_values(["target", "importance"], ascending=[True, False]).to_csv(METRICS_DIR / "feature_importance_regression.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(DATA_OUTPUT_DIR / "regression_test_predictions.csv", index=False, encoding="utf-8-sig")
    stage2 = pd.concat([
        train_df[["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]].merge(stage_pred_train, on="row_id", how="left"),
        test_df[["row_id", "ID", CLS_TARGET, "age", "sex", "HE_wt", "HE_wc", "WHtR", "HE_glu", "HE_sbp", "HE_chol"]].merge(stage_pred_test, on="row_id", how="left"),
    ], ignore_index=True)
    stage2.to_csv(DATA_OUTPUT_DIR / "stage2_input_with_predictions.csv", index=False, encoding="utf-8-sig")
    pd.Series({"random_state": RANDOM_STATE, "train_size": int(len(train_df)), "test_size": int(len(test_df)), "original_features": ORIGINAL_FEATURES, "engineered_features": ENGINEERED_FEATURES, "all_features": ALL_FEATURES, "regression_targets": REG_TARGETS}).to_json(METADATA_DIR / "split_meta.json", force_ascii=False, indent=2)
    pd.DataFrame({"num_feature": [c for c in ALL_FEATURES if c != "sex"], "median": imp.statistics_}).to_csv(METADATA_DIR / "feature_medians_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"sex_mode": [sex_mode]}).to_csv(METADATA_DIR / "sex_mode_train.csv", index=False, encoding="utf-8-sig")
    with open(METADATA_DIR / "regression_selected_configs.json", "w", encoding="utf-8") as fp:
        json.dump(selected, fp, ensure_ascii=False, indent=2)
    print(f"Saved: {METRICS_DIR / 'regression_metrics.csv'}")
    print(f"Saved: {METRICS_DIR / 'regression_baseline_vs_improved.csv'}")
    print(f"Saved: {METRICS_DIR / 'regression_experiment_results.csv'}")
    print(f"Saved: {DATA_OUTPUT_DIR / 'stage2_input_with_predictions.csv'}")
    print(f"Saved: {METADATA_DIR / 'regression_selected_configs.json'}")


if __name__ == "__main__":
    main()
